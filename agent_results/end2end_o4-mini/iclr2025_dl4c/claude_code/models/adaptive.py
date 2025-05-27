import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

class PolicyNetwork(torch.nn.Module):
    """
    Policy network for the adaptive code assistant.
    
    This wraps a pre-trained language model and adds a mechanism to incorporate
    user profile embeddings into the generation process.
    """
    
    def __init__(
        self,
        base_model,
        embedding_dim: int = 64,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Define profile embedding projection layer
        # This projects the user profile embedding to be combined with model hidden states
        self.profile_projection = torch.nn.Linear(
            embedding_dim, 
            self.base_model.config.hidden_size
        ).to(device)
        
        # Value function head for PPO
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.base_model.config.hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to(device)
    
    def forward(self, input_ids, attention_mask, profile_embedding=None, return_value=False):
        """
        Forward pass through the policy network.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask for input
            profile_embedding: Optional user profile embedding
            return_value: Whether to also return the value estimate

        Returns:
            Tuple of (logits, value) if return_value=True, else just logits
        """
        # Convert profile embedding to tensor if provided
        if profile_embedding is not None and isinstance(profile_embedding, np.ndarray):
            profile_embedding = torch.tensor(
                profile_embedding,
                dtype=torch.float32
            ).to(self.device)

        # Get encoder outputs
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get the last hidden state from encoder
        last_hidden_state = encoder_outputs.last_hidden_state

        # Incorporate profile embedding if provided
        if profile_embedding is not None:
            # Project profile embedding
            projected_profile = self.profile_projection(profile_embedding)

            # Reshape for broadcasting
            # (batch_size, 1, hidden_size)
            projected_profile = projected_profile.unsqueeze(1)

            # Add to hidden states (broadcasting across sequence length)
            adapted_hidden_state = last_hidden_state + projected_profile
        else:
            adapted_hidden_state = last_hidden_state

        # Update encoder outputs with adapted hidden state
        encoder_outputs.last_hidden_state = adapted_hidden_state

        # Run the decoder with encoder outputs
        decoder_outputs = self.base_model.decoder(
            input_ids=torch.zeros_like(input_ids[:, :1]),  # Start token
            encoder_hidden_states=adapted_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )

        # Get the decoder's last hidden state
        decoder_hidden_state = decoder_outputs.last_hidden_state

        # Get logits from the language model head
        logits = self.base_model.lm_head(decoder_hidden_state)

        if return_value:
            # Extract the last token's hidden state for value estimation
            # (batch_size, hidden_size)
            last_token_hidden = decoder_hidden_state[:, -1, :]

            # Compute value estimate
            value = self.value_head(last_token_hidden)

            return logits, value

        return logits
    
    def get_value(self, input_ids, attention_mask, profile_embedding=None):
        """Get value function estimate for state."""
        # Convert profile embedding to tensor if needed
        if profile_embedding is not None and isinstance(profile_embedding, np.ndarray):
            profile_embedding = torch.tensor(
                profile_embedding,
                dtype=torch.float32
            ).to(self.device)

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Get the encoder's last hidden state
            last_hidden_state = encoder_outputs.last_hidden_state

            # Incorporate profile embedding if provided
            if profile_embedding is not None:
                # Project profile embedding
                projected_profile = self.profile_projection(profile_embedding)

                # Reshape for broadcasting
                projected_profile = projected_profile.unsqueeze(1)

                # Add to hidden states
                adapted_hidden_state = last_hidden_state + projected_profile
            else:
                adapted_hidden_state = last_hidden_state

            # Update encoder outputs with adapted hidden state
            encoder_outputs.last_hidden_state = adapted_hidden_state

            # Run the decoder with encoder outputs
            decoder_outputs = self.base_model.decoder(
                input_ids=torch.zeros_like(input_ids[:, :1]),  # Start token
                encoder_hidden_states=adapted_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )

            # Get the decoder's last hidden state
            decoder_hidden_state = decoder_outputs.last_hidden_state

            # Extract the last token's hidden state
            last_token_hidden = decoder_hidden_state[:, -1, :]

            # Compute value estimate
            value = self.value_head(last_token_hidden)

        return value

class AdaptiveCodeAssistant:
    """
    Adaptive code assistant that learns from implicit developer feedback.
    
    This model uses PPO to fine-tune a pre-trained language model based on
    implicit feedback signals from developers.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5p-220m-py",  # Smaller model for faster experiments
        device: torch.device = torch.device('cpu'),
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        embedding_dim: int = 64,
        learning_rate: float = 3e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        buffer_size: int = 1000
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.embedding_dim = embedding_dim
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special handling for pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.base_model.to(device)
        
        # Create policy network (wraps base model)
        self.policy = PolicyNetwork(
            base_model=self.base_model,
            embedding_dim=embedding_dim,
            device=device
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Create replay buffer for PPO
        self.buffer = []
    
    def train(
        self,
        train_data: Dataset,
        valid_data: Dataset,
        epochs: int = 3,
        batch_size: int = 8,
        output_dir: str = "./models/adaptive"
    ):
        """
        Train the adaptive model using simulated developer feedback.
        
        This implements a simplified version of PPO training, where we:
        1. Generate suggestions for code contexts
        2. Simulate developer feedback
        3. Use the feedback as reward signals for PPO
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            
            # Clear buffer for new experiences
            self.buffer = []
            
            # Collect experiences
            for i, data_item in enumerate(tqdm(train_data, desc="Collecting experiences")):
                # Extract context and ground truth
                context = data_item['context']
                ground_truth = data_item['solution']
                
                # Simulate a developer profile
                from utils.data_utils import DeveloperProfile
                profile_id = f"dev_{i % 10}"  # Create 10 different profiles
                developer_profile = DeveloperProfile(profile_id, embedding_dim=self.embedding_dim)
                
                # Generate suggestion
                suggestion = self.generate_suggestion(
                    context=context,
                    developer_profile=developer_profile,
                    device=self.device
                )
                
                # Simulate developer feedback
                from utils.data_utils import simulate_developer_feedback
                reward, feedback_signals = simulate_developer_feedback(
                    suggestion=suggestion,
                    developer_profile=developer_profile,
                    ground_truth=ground_truth
                )
                
                # Tokenize context
                inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
                
                # Add experience to buffer
                self.buffer.append({
                    'context': context,
                    'input_ids': inputs.input_ids,
                    'attention_mask': inputs.attention_mask,
                    'profile_embedding': developer_profile.embedding,
                    'reward': reward,
                    'suggestion': suggestion,
                    'feedback': feedback_signals
                })
                
                # Update policy with PPO after collecting a batch of experiences
                if len(self.buffer) >= batch_size:
                    self._update_policy()
                    self.buffer = []  # Clear buffer after update
            
            # Final update if there are remaining experiences
            if len(self.buffer) > 0:
                self._update_policy()
                self.buffer = []
            
            # Evaluate on validation set
            val_metrics = self._evaluate(valid_data)
            print(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            self._save_checkpoint(f"{output_dir}/epoch_{epoch+1}")
        
        # Save final model
        self._save_checkpoint(f"{output_dir}/final")
    
    def _update_policy(self):
        """Update policy network using PPO."""
        # Extract experiences from buffer
        contexts = [exp['context'] for exp in self.buffer]
        input_ids = torch.cat([exp['input_ids'] for exp in self.buffer], dim=0)
        attention_mask = torch.cat([exp['attention_mask'] for exp in self.buffer], dim=0)
        profile_embeddings = np.stack([exp['profile_embedding'] for exp in self.buffer])
        rewards = torch.tensor([exp['reward'] for exp in self.buffer], dtype=torch.float32).to(self.device)
        
        # Convert profile embeddings to tensor
        profile_embeddings = torch.tensor(profile_embeddings, dtype=torch.float32).to(self.device)
        
        # Get old logits and values
        with torch.no_grad():
            old_logits, old_values = self.policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                profile_embedding=profile_embeddings,
                return_value=True
            )
            old_logprobs = torch.nn.functional.log_softmax(old_logits, dim=-1)
            old_values = old_values.squeeze(-1)
        
        # PPO update loop
        for _ in range(self.ppo_epochs):
            # Forward pass
            logits, values = self.policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                profile_embedding=profile_embeddings,
                return_value=True
            )
            
            # Compute log probabilities and entropy
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            entropy = -(logprobs.exp() * logprobs).sum(dim=-1).mean()
            
            # Simplify: use rewards directly as advantage estimates
            # In a real implementation, you'd compute proper advantage estimates
            advantages = rewards - old_values.detach()
            
            # Get the token probabilities for the actual next token
            # Simplification: just use the last token's logprob
            next_token_ids = input_ids[:, -1]
            new_logprobs_action = torch.gather(
                logprobs[:, -1, :], 
                dim=1, 
                index=next_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            old_logprobs_action = torch.gather(
                old_logprobs[:, -1, :], 
                dim=1, 
                index=next_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Compute ratio and clipped ratio
            ratio = torch.exp(new_logprobs_action - old_logprobs_action.detach())
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            
            # Compute losses
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            values = values.squeeze(-1)
            value_loss = torch.nn.functional.mse_loss(values, rewards)
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
            
            # Perform update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def _evaluate(self, eval_data: Dataset) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.policy.eval()
        
        total_reward = 0.0
        acceptance_rate = 0.0
        total_edit_distance = 0.0
        
        with torch.no_grad():
            for i, data_item in enumerate(eval_data):
                # Extract context and ground truth
                context = data_item['context']
                ground_truth = data_item['solution']
                
                # Simulate a developer profile
                from utils.data_utils import DeveloperProfile
                profile_id = f"dev_val_{i % 5}"  # Create 5 different validation profiles
                developer_profile = DeveloperProfile(profile_id, embedding_dim=self.embedding_dim)
                
                # Generate suggestion
                suggestion = self.generate_suggestion(
                    context=context,
                    developer_profile=developer_profile,
                    device=self.device
                )
                
                # Simulate developer feedback
                from utils.data_utils import simulate_developer_feedback
                reward, feedback_signals = simulate_developer_feedback(
                    suggestion=suggestion,
                    developer_profile=developer_profile,
                    ground_truth=ground_truth
                )
                
                # Update metrics
                total_reward += reward
                acceptance_rate += feedback_signals['accept']
                total_edit_distance += feedback_signals['edit_distance']
        
        # Calculate average metrics
        num_examples = len(eval_data)
        avg_reward = total_reward / num_examples if num_examples > 0 else 0
        avg_acceptance_rate = acceptance_rate / num_examples if num_examples > 0 else 0
        avg_edit_distance = total_edit_distance / num_examples if num_examples > 0 else 0
        
        self.policy.train()
        
        return {
            'avg_reward': avg_reward,
            'acceptance_rate': avg_acceptance_rate,
            'avg_edit_distance': avg_edit_distance
        }
    
    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save base model and tokenizer
        self.base_model.save_pretrained(f"{path}/base_model")
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
        
        # Save policy network state
        torch.save({
            'profile_projection': self.policy.profile_projection.state_dict(),
            'value_head': self.policy.value_head.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"{path}/policy_network.pt")
    
    def generate_suggestion(
        self,
        context: str,
        developer_profile,
        device: Optional[torch.device] = None
    ) -> str:
        """
        Generate code suggestion for the given context, considering the developer profile.
        """
        device = device or self.device
        self.policy.eval()

        # Prepare input
        inputs = self.tokenizer(context, return_tensors="pt").to(device)

        # Get profile embedding
        profile_embedding = developer_profile.embedding
        profile_embedding_tensor = torch.tensor(profile_embedding, dtype=torch.float32).to(device)

        # Generate with simpler approach for T5
        with torch.no_grad():
            # Project profile embedding
            projected_profile = self.policy.profile_projection(profile_embedding_tensor)

            # Create encoder outputs with profile information
            encoder_outputs = self.base_model.encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True
            )

            # Modify last hidden state with profile information
            last_hidden_state = encoder_outputs.last_hidden_state
            adapted_hidden_state = last_hidden_state + projected_profile.unsqueeze(1)
            encoder_outputs.last_hidden_state = adapted_hidden_state

            # Generate with modified encoder outputs
            outputs = self.base_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode suggestion
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return suggestion