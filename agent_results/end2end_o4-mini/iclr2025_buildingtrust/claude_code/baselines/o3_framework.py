"""
Implementation of the O3 Framework baseline method for LLM unlearning.
Based on: "Practical Unlearning for Large Language Models" (Gao et al., 2024)
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


class OODDetector(nn.Module):
    """
    Out-Of-Distribution (OOD) detector to measure the similarity between
    input and unlearning data.
    """
    
    def __init__(self, hidden_size, num_layers=1, temperature=0.1):
        """
        Initialize the OOD detector.
        
        Args:
            hidden_size: Hidden size of the transformer
            num_layers: Number of layers to use for detection
            temperature: Temperature for contrastive entropy loss
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temperature = temperature
        
        # Projection for each layer
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Layer-aggregated scoring parameters
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Centroids for unlearning data and normal data
        self.unlearn_centroids = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
        
        self.normal_centroids = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size))
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states_list):
        """
        Compute OOD scores for input.
        
        Args:
            hidden_states_list: List of hidden states from different layers
            
        Returns:
            ood_scores: OOD scores (higher means more similar to unlearning data)
        """
        # Ensure we have the right number of hidden states
        assert len(hidden_states_list) >= self.num_layers, "Not enough hidden states provided"
        
        # Use the last num_layers hidden states
        if len(hidden_states_list) > self.num_layers:
            hidden_states_list = hidden_states_list[-self.num_layers:]
        
        # Compute scores for each layer
        layer_scores = []
        
        for i, hidden_states in enumerate(hidden_states_list):
            # Project hidden states
            projected = self.projections[i](hidden_states)
            
            # Compute similarity to centroids
            unlearn_sim = F.cosine_similarity(
                projected.view(-1, self.hidden_size),
                self.unlearn_centroids[i].unsqueeze(0),
                dim=1
            )
            
            normal_sim = F.cosine_similarity(
                projected.view(-1, self.hidden_size),
                self.normal_centroids[i].unsqueeze(0),
                dim=1
            )
            
            # Compute layer score
            layer_score = unlearn_sim - normal_sim
            layer_scores.append(layer_score)
        
        # Stack layer scores
        stacked_scores = torch.stack(layer_scores, dim=1)  # [batch_size, num_layers]
        
        # Compute weighted average
        softmax_weights = F.softmax(self.layer_weights, dim=0)
        ood_scores = torch.sum(stacked_scores * softmax_weights, dim=1)
        
        return ood_scores
    
    def compute_loss(self, unlearn_hidden_states, normal_hidden_states):
        """
        Compute contrastive entropy loss for training the detector.
        
        Args:
            unlearn_hidden_states: List of hidden states from unlearning data
            normal_hidden_states: List of hidden states from normal data
            
        Returns:
            loss: Contrastive entropy loss
        """
        # Ensure we have the right number of hidden states
        assert len(unlearn_hidden_states) >= self.num_layers, "Not enough unlearn hidden states"
        assert len(normal_hidden_states) >= self.num_layers, "Not enough normal hidden states"
        
        # Use the last num_layers hidden states
        if len(unlearn_hidden_states) > self.num_layers:
            unlearn_hidden_states = unlearn_hidden_states[-self.num_layers:]
            normal_hidden_states = normal_hidden_states[-self.num_layers:]
        
        # Compute loss for each layer
        layer_losses = []
        
        for i in range(self.num_layers):
            # Project hidden states
            unlearn_projected = self.projections[i](unlearn_hidden_states[i])
            normal_projected = self.projections[i](normal_hidden_states[i])
            
            # Update centroids with moving average
            with torch.no_grad():
                unlearn_mean = unlearn_projected.mean(dim=0)
                normal_mean = normal_projected.mean(dim=0)
                
                self.unlearn_centroids[i].data = 0.9 * self.unlearn_centroids[i].data + 0.1 * unlearn_mean
                self.normal_centroids[i].data = 0.9 * self.normal_centroids[i].data + 0.1 * normal_mean
            
            # Compute similarities
            unlearn_u_sim = F.cosine_similarity(
                unlearn_projected.view(-1, self.hidden_size),
                self.unlearn_centroids[i].unsqueeze(0),
                dim=1
            ) / self.temperature
            
            unlearn_n_sim = F.cosine_similarity(
                unlearn_projected.view(-1, self.hidden_size),
                self.normal_centroids[i].unsqueeze(0),
                dim=1
            ) / self.temperature
            
            normal_u_sim = F.cosine_similarity(
                normal_projected.view(-1, self.hidden_size),
                self.unlearn_centroids[i].unsqueeze(0),
                dim=1
            ) / self.temperature
            
            normal_n_sim = F.cosine_similarity(
                normal_projected.view(-1, self.hidden_size),
                self.normal_centroids[i].unsqueeze(0),
                dim=1
            ) / self.temperature
            
            # Compute contrastive entropy loss
            unlearn_loss = -torch.log(
                torch.exp(unlearn_u_sim) / (torch.exp(unlearn_u_sim) + torch.exp(unlearn_n_sim))
            ).mean()
            
            normal_loss = -torch.log(
                torch.exp(normal_n_sim) / (torch.exp(normal_u_sim) + torch.exp(normal_n_sim))
            ).mean()
            
            layer_loss = unlearn_loss + normal_loss
            layer_losses.append(layer_loss)
        
        # Compute total loss
        total_loss = sum(layer_losses) / self.num_layers
        
        return total_loss


class OrthogonalLoRA(nn.Module):
    """
    Orthogonal Low-Rank Adapter (LoRA) for unlearning specific data.
    """
    
    def __init__(self, hidden_size, adapter_dim=4, alpha=8):
        """
        Initialize the orthogonal LoRA module.
        
        Args:
            hidden_size: Hidden size of the transformer
            adapter_dim: Dimension of the low-rank adaptation
            alpha: Scaling factor for the adapter
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.alpha = alpha
        
        # LoRA weight matrices
        self.lora_A = nn.Parameter(torch.randn(adapter_dim, hidden_size))
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, adapter_dim))
        
        # Initialize A with Gaussian
        nn.init.normal_(self.lora_A, std=1.0 / adapter_dim)
        
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
        
        # Register a buffer for orthogonality regularization
        self.register_buffer('eye', torch.eye(adapter_dim))
    
    def forward(self, hidden_states):
        """
        Apply orthogonal LoRA to hidden states.
        
        Args:
            hidden_states: Hidden states from the transformer
            
        Returns:
            adjusted_hidden_states: Hidden states after LoRA application
        """
        # Apply LoRA adjustment: x + (B @ A) @ x * (alpha / adapter_dim)
        adjustment = (hidden_states @ self.lora_A.t() @ self.lora_B.t()) * (self.alpha / self.adapter_dim)
        adjusted_hidden_states = hidden_states + adjustment
        
        return adjusted_hidden_states
    
    def compute_orthogonality_loss(self):
        """
        Compute orthogonality regularization loss.
        
        Returns:
            loss: Orthogonality loss
        """
        # Compute A @ A^T - I
        A_gram = self.lora_A @ self.lora_A.t()
        orthogonality_loss = F.mse_loss(A_gram, self.eye)
        
        return orthogonality_loss


class O3Framework:
    """
    O3 Framework implementation combining OOD detector and orthogonal LoRA.
    
    The method includes an Out-Of-Distribution (OOD) detector to measure similarity
    between input and unlearning data, and an Orthogonal low-rank adapter (LoRA)
    for continuously unlearning requested data.
    """
    
    def __init__(
        self,
        model,
        optimizer_class=torch.optim.AdamW,
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=16,
        num_detector_layers=2,
        adapter_dim=4,
        ood_threshold=0.5,
        orthogonal_weight=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the O3 Framework.
        
        Args:
            model: The pretrained LLM model
            optimizer_class: Optimizer class to use
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for fine-tuning
            num_detector_layers: Number of layers for OOD detector
            adapter_dim: Dimension of the low-rank adaptation
            ood_threshold: Threshold for OOD detection
            orthogonal_weight: Weight for orthogonality regularization
            device: Device to use for computation
        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_detector_layers = num_detector_layers
        self.adapter_dim = adapter_dim
        self.ood_threshold = ood_threshold
        self.orthogonal_weight = orthogonal_weight
        self.device = device
        
        # Determine hidden size
        if hasattr(model, 'config'):
            self.hidden_size = model.config.hidden_size
        else:
            # Try to infer from the model parameters
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    self.hidden_size = param.shape[0]
                    break
            else:
                self.hidden_size = 768  # Default size if we can't determine
        
        # Storage for sequential unlearning
        self.lora_adapters = {}
        self.ood_detectors = {}
    
    def _collect_hidden_states(self, model, batch):
        """
        Collect hidden states from the model.
        
        Args:
            model: The model to extract hidden states from
            batch: Batch of examples
            
        Returns:
            hidden_states_list: List of hidden states from different layers
        """
        # Move batch to device
        if isinstance(batch, dict):
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
        else:
            # Assume batch is a tuple of (inputs, targets)
            inputs = {k: v.to(self.device) for k, v in batch[0].items()}
        
        # Get hidden states
        hidden_states_list = []
        
        # For transformers models with GPT-2 style architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # Register hooks to get hidden states
            hooks = []
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states_list.append(output[0].detach())
                else:
                    hidden_states_list.append(output.detach())
            
            # Register hooks on transformer layers
            for layer in model.transformer.h:
                hooks.append(layer.register_forward_hook(hook_fn))
                
            # Forward pass
            with torch.no_grad():
                model(**inputs)
                
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
        # For transformers models with BERT style architecture
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # Register hooks to get hidden states
            hooks = []
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states_list.append(output[0].detach())
                else:
                    hidden_states_list.append(output.detach())
            
            # Register hooks on encoder layers
            for layer in model.encoder.layer:
                hooks.append(layer.register_forward_hook(hook_fn))
                
            # Forward pass
            with torch.no_grad():
                model(**inputs)
                
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
        # For models with output.hidden_states
        else:
            # Set output_hidden_states=True
            if 'output_hidden_states' not in inputs:
                inputs['output_hidden_states'] = True
                
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Extract hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states_list = list(outputs.hidden_states)
            else:
                # Fallback
                hidden_states_list = [
                    torch.randn(
                        inputs['input_ids'].shape[0],
                        inputs['input_ids'].shape[1],
                        self.hidden_size,
                        device=self.device
                    )
                    for _ in range(self.num_detector_layers)
                ]
        
        return hidden_states_list
    
    def _insert_lora_adapter(self, model, adapter_id):
        """
        Insert LoRA adapter into the model.
        
        Args:
            model: The model to modify
            adapter_id: Identifier for the adapter
            
        Returns:
            modified_model: The modified model with LoRA adapter
        """
        # Create a copy of the model
        modified_model = copy.deepcopy(model)
        
        # Create LoRA adapter
        lora_adapter = OrthogonalLoRA(
            hidden_size=self.hidden_size,
            adapter_dim=self.adapter_dim
        ).to(self.device)
        
        # Store the adapter
        self.lora_adapters[adapter_id] = lora_adapter
        
        # For transformers models with GPT-2 style architecture
        if hasattr(modified_model, 'transformer') and hasattr(modified_model.transformer, 'h'):
            # Modify the last layer
            layer_idx = len(modified_model.transformer.h) - 1
            
            # Save the original forward method
            original_forward = modified_model.transformer.h[layer_idx].forward
            
            # Define a new forward method with LoRA adaptation
            def new_forward(hidden_states, *args, **kwargs):
                # Call the original forward method
                outputs = original_forward(hidden_states, *args, **kwargs)
                
                # Apply LoRA adapter
                if isinstance(outputs, tuple):
                    adapted_hidden_states = lora_adapter(outputs[0])
                    modified_outputs = (adapted_hidden_states,) + outputs[1:]
                else:
                    modified_outputs = lora_adapter(outputs)
                
                return modified_outputs
            
            # Replace the forward method
            modified_model.transformer.h[layer_idx].forward = new_forward
            
        # For transformers models with BERT style architecture
        elif hasattr(modified_model, 'encoder') and hasattr(modified_model.encoder, 'layer'):
            # Modify the last layer
            layer_idx = len(modified_model.encoder.layer) - 1
            
            # Save the original forward method
            original_forward = modified_model.encoder.layer[layer_idx].forward
            
            # Define a new forward method with LoRA adaptation
            def new_forward(hidden_states, *args, **kwargs):
                # Call the original forward method
                outputs = original_forward(hidden_states, *args, **kwargs)
                
                # Apply LoRA adapter
                if isinstance(outputs, tuple):
                    adapted_hidden_states = lora_adapter(outputs[0])
                    modified_outputs = (adapted_hidden_states,) + outputs[1:]
                else:
                    modified_outputs = lora_adapter(outputs)
                
                return modified_outputs
            
            # Replace the forward method
            modified_model.encoder.layer[layer_idx].forward = new_forward
        
        return modified_model
    
    def _train_ood_detector(self, deletion_set, validation_set):
        """
        Train the OOD detector.
        
        Args:
            deletion_set: Set of examples to delete
            validation_set: Set of examples to retain
            
        Returns:
            detector: Trained OOD detector
        """
        # Create OOD detector
        detector = OODDetector(
            hidden_size=self.hidden_size,
            num_layers=self.num_detector_layers,
            temperature=0.1
        ).to(self.device)
        
        # Create data loaders
        deletion_loader = DataLoader(
            deletion_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        validation_loader = DataLoader(
            validation_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create optimizer
        optimizer = self.optimizer_class(
            detector.parameters(),
            lr=self.learning_rate
        )
        
        # Train the detector
        detector.train()
        self.model.eval()
        
        for epoch in range(self.num_epochs):
            for unlearn_batch, normal_batch in zip(deletion_loader, validation_loader):
                optimizer.zero_grad()
                
                # Collect hidden states
                unlearn_hidden_states = self._collect_hidden_states(self.model, unlearn_batch)
                normal_hidden_states = self._collect_hidden_states(self.model, normal_batch)
                
                # Compute loss
                loss = detector.compute_loss(unlearn_hidden_states, normal_hidden_states)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
        
        return detector
    
    def _train_lora_adapter(self, model, detector, deletion_set, validation_set, adapter_id):
        """
        Train the LoRA adapter for unlearning.
        
        Args:
            model: The model with inserted LoRA adapter
            detector: Trained OOD detector
            deletion_set: Set of examples to delete
            validation_set: Set of examples to retain
            adapter_id: Identifier for the adapter
            
        Returns:
            trained_model: The model with trained LoRA adapter
        """
        # Get the LoRA adapter
        lora_adapter = self.lora_adapters[adapter_id]
        
        # Create data loaders
        deletion_loader = DataLoader(
            deletion_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        validation_loader = DataLoader(
            validation_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create optimizer (only for LoRA parameters)
        optimizer = self.optimizer_class(
            lora_adapter.parameters(),
            lr=self.learning_rate
        )
        
        # Train the adapter
        model.train()
        lora_adapter.train()
        detector.eval()
        
        for epoch in range(self.num_epochs):
            # Train on validation data to maintain performance
            for batch in validation_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    targets = batch['targets'].to(self.device)
                else:
                    # Assume batch is a tuple of (inputs, targets)
                    inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                    targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Compute standard cross-entropy loss
                if logits.shape[:-1] != targets.shape:
                    # Reshape for language modeling
                    shifted_logits = logits.contiguous().view(-1, logits.size(-1))
                    shifted_targets = targets.contiguous().view(-1)
                else:
                    shifted_logits = logits
                    shifted_targets = targets
                    
                ce_loss = F.cross_entropy(shifted_logits, shifted_targets)
                
                # Add orthogonality regularization
                ortho_loss = lora_adapter.compute_orthogonality_loss()
                loss = ce_loss + self.orthogonal_weight * ortho_loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
            
            # Train on deletion data to unlearn
            for batch in deletion_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    targets = batch['targets'].to(self.device)
                else:
                    # Assume batch is a tuple of (inputs, targets)
                    inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                    targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Collect hidden states
                hidden_states_list = self._collect_hidden_states(model, batch)
                
                # Compute OOD scores
                ood_scores = detector(hidden_states_list)
                
                # Compute unlearning loss (maximize cross-entropy on deletion data)
                if logits.shape[:-1] != targets.shape:
                    # Reshape for language modeling
                    shifted_logits = logits.contiguous().view(-1, logits.size(-1))
                    shifted_targets = targets.contiguous().view(-1)
                else:
                    shifted_logits = logits
                    shifted_targets = targets
                
                # We want to maximize the loss for unlearning, so negate it
                unlearn_loss = -F.cross_entropy(shifted_logits, shifted_targets)
                
                # Add orthogonality regularization
                ortho_loss = lora_adapter.compute_orthogonality_loss()
                loss = unlearn_loss + self.orthogonal_weight * ortho_loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
        
        return model
    
    def unlearn(self, validation_data, deletion_set, adapter_id=None):
        """
        Perform unlearning using the O3 Framework.
        
        Args:
            validation_data: Dataset of examples to retain
            deletion_set: Set of examples to delete
            adapter_id: Identifier for the adapter (for sequential unlearning)
            
        Returns:
            unlearned_model: The unlearned model
            metrics: Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Generate adapter ID if not provided
        if adapter_id is None:
            adapter_id = f"adapter_{len(self.lora_adapters)}"
            
        # Train OOD detector
        detector = self._train_ood_detector(deletion_set, validation_data)
        
        # Store the detector
        self.ood_detectors[adapter_id] = detector
        
        # Insert LoRA adapter
        unlearned_model = self._insert_lora_adapter(self.model, adapter_id)
        
        # Train LoRA adapter
        unlearned_model = self._train_lora_adapter(
            unlearned_model, detector, deletion_set, validation_data, adapter_id
        )
        
        # Compute metrics
        metrics = {
            'method': 'O3 Framework',
            'compute_time': time.time() - start_time,
            'adapter_id': adapter_id
        }
        
        return unlearned_model, metrics
    
    def sequential_unlearn(self, validation_data, deletion_set):
        """
        Perform sequential unlearning.
        
        Args:
            validation_data: Dataset of examples to retain
            deletion_set: Set of examples to delete
            
        Returns:
            unlearned_model: The unlearned model with all adapters
            metrics: Unlearning metrics
        """
        # Generate a new adapter ID
        adapter_id = f"adapter_{len(self.lora_adapters)}"
        
        # Perform unlearning with the new adapter
        unlearned_model, metrics = self.unlearn(validation_data, deletion_set, adapter_id)
        
        # Add sequential information to metrics
        metrics['is_sequential'] = True
        metrics['num_adapters'] = len(self.lora_adapters)
        
        return unlearned_model, metrics
    
    def apply_ood_detection(self, model, inputs):
        """
        Apply OOD detection to inputs during inference.
        
        Args:
            model: The model with LoRA adapters
            inputs: Input tensors
            
        Returns:
            outputs: Model outputs, possibly with modified logits
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Collect hidden states
        hidden_states_list = self._collect_hidden_states(model, inputs)
        
        # Apply OOD detection for each detector
        max_ood_score = float('-inf')
        detected_adapter = None
        
        for adapter_id, detector in self.ood_detectors.items():
            ood_scores = detector(hidden_states_list)
            batch_score = ood_scores.mean().item()
            
            if batch_score > max_ood_score:
                max_ood_score = batch_score
                detected_adapter = adapter_id
        
        # If OOD score exceeds threshold, use the unlearned model
        if max_ood_score > self.ood_threshold and detected_adapter is not None:
            # Forward pass with model that includes the detected adapter
            outputs = model(**inputs)
        else:
            # Forward pass with original model
            outputs = self.model(**inputs)
        
        return outputs