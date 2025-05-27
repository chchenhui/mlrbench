"""
Baseline Methods Module

This module implements baseline explainability methods for comparison.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class BaselinesRunner:
    """
    Class for running baseline explainability methods.
    
    This class provides implementations of:
    1. Attention visualization
    2. Integrated Gradients
    3. Chain-of-Thought (CoT) extraction
    4. Basic token attribution
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the baselines runner.
        
        Args:
            model: Pre-trained language model
            tokenizer: Tokenizer for the model
            device: Device to run the model on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def _prepare_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize the input prompt and prepare it for the model.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary of model inputs
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return inputs
    
    def run_attention_visualization(
        self,
        prompt: str,
        save_path: Optional[str] = None,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        top_k_heads: int = 3
    ) -> Dict[str, Any]:
        """
        Run attention visualization baseline.
        
        Args:
            prompt: Input text prompt
            save_path: Path to save the visualization
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize (None for all heads)
            top_k_heads: Number of top attention heads to visualize
            
        Returns:
            Dictionary with attention weights and visualization
        """
        logger.info(f"Running attention visualization baseline (layer={layer_idx}, head={head_idx})")
        
        # Prepare input
        inputs = self._prepare_input(prompt)
        
        # Run model
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention weights
        attentions = outputs.attentions  # Tuple of attention tensors
        
        # Select layer (default to last layer)
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx
        
        if layer_idx < 0 or layer_idx >= len(attentions):
            logger.error(f"Invalid layer index: {layer_idx}, using last layer")
            layer_idx = len(attentions) - 1
        
        layer_attention = attentions[layer_idx]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Convert tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Prepare result
        result = {
            'tokens': tokens,
            'attention_weights': {},
            'layer_idx': layer_idx
        }
        
        # Visualize attention
        num_heads = layer_attention.size(1)
        
        if head_idx is not None:
            # Visualize specific head
            if head_idx < 0 or head_idx >= num_heads:
                logger.error(f"Invalid head index: {head_idx}, using head 0")
                head_idx = 0
            
            head_attention = layer_attention[0, head_idx].cpu().numpy()
            result['attention_weights'][head_idx] = head_attention
            
            # Create visualization
            self._visualize_attention_matrix(
                head_attention,
                tokens,
                f"Layer {layer_idx}, Head {head_idx}",
                save_path
            )
        
        else:
            # Visualize top-k heads
            if top_k_heads > num_heads:
                top_k_heads = num_heads
            
            # Calculate head importance based on attention entropy
            head_importance = []
            for h in range(num_heads):
                head_attn = layer_attention[0, h].cpu().numpy()
                
                # Calculate entropy of attention weights
                # Low entropy means attention is focused on specific tokens
                head_attn = head_attn + 1e-10  # Avoid log(0)
                head_attn = head_attn / head_attn.sum(axis=1, keepdims=True)
                entropy = -np.sum(head_attn * np.log(head_attn), axis=1).mean()
                
                head_importance.append((h, entropy))
            
            # Sort heads by importance (low entropy = more focused attention)
            head_importance.sort(key=lambda x: x[1])
            top_heads = [h for h, _ in head_importance[:top_k_heads]]
            
            # Visualize top heads
            fig, axes = plt.subplots(1, top_k_heads, figsize=(5 * top_k_heads, 5))
            
            if top_k_heads == 1:
                axes = [axes]
            
            for i, h in enumerate(top_heads):
                head_attn = layer_attention[0, h].cpu().numpy()
                result['attention_weights'][h] = head_attn
                
                ax = axes[i]
                sns.heatmap(
                    head_attn,
                    ax=ax,
                    cmap="viridis",
                    xticklabels=tokens,
                    yticklabels=tokens
                )
                ax.set_title(f"Layer {layer_idx}, Head {h}")
                
                # Rotate labels for better visibility
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                result['visualization_path'] = save_path
                logger.info(f"Saved attention visualization to {save_path}")
            
            plt.close()
        
        return result
    
    def _visualize_attention_matrix(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        title: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize an attention matrix.
        
        Args:
            attention_matrix: Attention weight matrix
            tokens: List of tokens for axis labels
            title: Title for the visualization
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            attention_matrix,
            ax=ax,
            cmap="viridis",
            xticklabels=tokens,
            yticklabels=tokens
        )
        
        ax.set_title(title)
        
        # Rotate labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention matrix visualization to {save_path}")
        
        return fig
    
    def run_integrated_gradients(
        self,
        prompt: str,
        target_token_idx: Optional[int] = None,
        num_steps: int = 20,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run Integrated Gradients baseline for token attribution.
        
        Args:
            prompt: Input text prompt
            target_token_idx: Index of the target token to explain (None for last token)
            num_steps: Number of steps for Integrated Gradients
            save_path: Path to save the visualization
            
        Returns:
            Dictionary with attribution scores and visualization
        """
        logger.info(f"Running Integrated Gradients baseline (target_idx={target_token_idx}, steps={num_steps})")
        
        # Prepare input
        inputs = self._prepare_input(prompt)
        input_ids = inputs['input_ids']
        input_length = input_ids.size(1)
        
        # If target_token_idx is not specified, use the last token
        if target_token_idx is None:
            target_token_idx = input_length - 1
        
        # Ensure target_token_idx is valid
        if target_token_idx < 0 or target_token_idx >= input_length:
            logger.error(f"Invalid target token index: {target_token_idx}, using last token")
            target_token_idx = input_length - 1
        
        # Define the forward function for Integrated Gradients
        def forward_func(input_ids):
            outputs = self.model(input_ids=input_ids)
            return outputs.logits[:, target_token_idx, :]
        
        # Create the integrated gradients instance
        ig = IntegratedGradients(forward_func)
        
        # Define the baseline input (all PAD tokens)
        baseline_input = torch.ones_like(input_ids) * self.tokenizer.pad_token_id
        
        # Run attribution
        attribution = ig.attribute(
            inputs=input_ids,
            baselines=baseline_input,
            target=input_ids[0, target_token_idx].item(),
            n_steps=num_steps
        )
        
        # Extract attribution scores
        attribution_scores = attribution.sum(dim=-1)[0].detach().cpu().numpy()
        
        # Convert tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(
            range(len(tokens)),
            attribution_scores,
            color='skyblue'
        )
        
        # Highlight the target token
        bars[target_token_idx].set_color('red')
        
        # Add token labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        ax.set_title(f"Integrated Gradients Attribution (Target: '{tokens[target_token_idx]}')")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Attribution Score")
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Integrated Gradients visualization to {save_path}")
        
        # Prepare result
        result = {
            'tokens': tokens,
            'attribution_scores': attribution_scores.tolist(),
            'target_token_idx': target_token_idx,
            'target_token': tokens[target_token_idx]
        }
        
        if save_path:
            result['visualization_path'] = save_path
        
        return result
    
    def extract_cot_steps(
        self,
        model_output: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract reasoning steps from Chain-of-Thought output.
        
        Args:
            model_output: Generated text with Chain-of-Thought reasoning
            save_path: Path to save the visualization
            
        Returns:
            Dictionary with extracted reasoning steps
        """
        logger.info("Extracting Chain-of-Thought reasoning steps")
        
        # Simple heuristic: Split by newlines and/or numbered lists
        lines = model_output.strip().split('\n')
        
        # Extract non-empty lines that might be reasoning steps
        steps = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Remove step numbers if present
            if line[0].isdigit() and line[1:3] in ['. ', ') ', '- ']:
                line = line[3:].strip()
            elif line[0] == '-' or line[0] == '*':
                line = line[1:].strip()
            
            steps.append(line)
        
        # Visualize steps
        if save_path and steps:
            fig, ax = plt.subplots(figsize=(10, len(steps) * 0.5 + 2))
            
            ax.axis('off')
            ax.set_title("Chain-of-Thought Reasoning Steps")
            
            for i, step in enumerate(steps):
                ax.text(
                    0.1,
                    0.9 - (i * 0.8 / len(steps)),
                    f"{i+1}. {step}",
                    fontsize=12,
                    ha='left',
                    va='top',
                    wrap=True
                )
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Chain-of-Thought visualization to {save_path}")
            
            plt.close()
        
        # Prepare result
        result = {
            'num_steps': len(steps),
            'steps': steps
        }
        
        if save_path:
            result['visualization_path'] = save_path
        
        return result
    
    def run_token_attribution(
        self,
        prompt: str,
        generated_text: str,
        method: str = "attention",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run basic token attribution to highlight important tokens.
        
        Args:
            prompt: Input text prompt
            generated_text: Generated text from the model
            method: Attribution method ("attention" or "gradient")
            save_path: Path to save the visualization
            
        Returns:
            Dictionary with attribution scores and visualization
        """
        logger.info(f"Running token attribution baseline (method={method})")
        
        # Prepare input
        inputs = self._prepare_input(prompt)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Run appropriate attribution method
        if method == "attention":
            # Use attention weights for attribution
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            attentions = outputs.attentions  # Tuple of attention tensors
            
            # Use the average attention from the last layer
            last_layer_attention = attentions[-1][0].mean(dim=0).cpu().numpy()  # Average over heads
            
            # Sum attention received by each token
            attribution_scores = last_layer_attention.sum(axis=0)
            
        elif method == "gradient":
            # Run simple gradient-based attribution
            inputs_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
            inputs_embeds.requires_grad_(True)
            
            outputs = self.model(inputs_embeds=inputs_embeds)
            logits = outputs.logits
            
            # Use the last token logit for gradient computation
            target_logit = logits[0, -1, logits[0, -1].argmax()]
            target_logit.backward()
            
            # Get gradient with respect to input
            grads = inputs_embeds.grad[0].sum(dim=1).detach().cpu().numpy()
            attribution_scores = np.abs(grads)  # Use absolute gradient magnitude
            
        else:
            logger.error(f"Unsupported attribution method: {method}")
            attribution_scores = np.ones(len(tokens))
        
        # Normalize scores
        attribution_scores = attribution_scores / attribution_scores.max()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(
            range(len(tokens)),
            attribution_scores,
            color='skyblue'
        )
        
        # Find top tokens
        top_indices = np.argsort(attribution_scores)[-5:]
        for idx in top_indices:
            bars[idx].set_color('orange')
        
        # Add token labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        ax.set_title(f"Token Attribution Scores ({method.capitalize()} Method)")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Attribution Score")
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved token attribution visualization to {save_path}")
        
        # Prepare result
        result = {
            'tokens': tokens,
            'attribution_scores': attribution_scores.tolist(),
            'top_tokens': [(tokens[idx], attribution_scores[idx]) for idx in top_indices],
            'method': method
        }
        
        if save_path:
            result['visualization_path'] = save_path
        
        return result
    
    def run_all_baselines(
        self,
        prompt: str,
        generated_text: str,
        save_dir: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all baseline methods for comparison.
        
        Args:
            prompt: Input text prompt
            generated_text: Generated text from the model
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with results from all baseline methods
        """
        logger.info("Running all baseline methods")
        
        os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        # Run attention visualization
        attention_path = os.path.join(save_dir, "attention_vis.png")
        results['attention'] = self.run_attention_visualization(
            prompt,
            save_path=attention_path
        )
        
        # Run integrated gradients
        ig_path = os.path.join(save_dir, "integrated_gradients.png")
        results['integrated_gradients'] = self.run_integrated_gradients(
            prompt,
            save_path=ig_path
        )
        
        # Extract CoT steps
        cot_path = os.path.join(save_dir, "cot_steps.png")
        results['cot'] = self.extract_cot_steps(
            generated_text,
            save_path=cot_path
        )
        
        # Run token attribution
        attr_path = os.path.join(save_dir, "token_attribution.png")
        results['token_attribution'] = self.run_token_attribution(
            prompt,
            generated_text,
            save_path=attr_path
        )
        
        return results