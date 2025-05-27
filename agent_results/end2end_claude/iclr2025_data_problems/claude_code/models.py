#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model implementations for Attribution-Guided Training experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoModel, AutoModelForMaskedLM
import logging

logger = logging.getLogger(__name__)

class AttributionNetwork(nn.Module):
    """
    Attribution network that maps language model hidden states to source predictions.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the attribution network.
        
        Args:
            hidden_size: Size of the language model hidden states
            num_sources: Number of sources to predict
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = hidden_size
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_sources))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: Hidden states from language model [batch_size, hidden_size]
            
        Returns:
            Source logits [batch_size, num_sources]
        """
        return self.network(hidden_states)

class LayerSpecificAttributionNetwork(nn.Module):
    """
    Attribution network that uses specific layers from the language model.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        target_layer: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the layer-specific attribution network.
        
        Args:
            hidden_size: Size of the language model hidden states
            num_sources: Number of sources to predict
            target_layer: Which layer to use from the language model
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        self.target_layer = target_layer
        self.attribution_network = AttributionNetwork(
            hidden_size=hidden_size,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    
    def forward(self, hidden_states_all_layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states_all_layers: Hidden states from all layers of the language model
                Each tensor is [batch_size, seq_len, hidden_size]
            
        Returns:
            Source logits [batch_size, num_sources]
        """
        # Get hidden states from target layer
        hidden_states = hidden_states_all_layers[self.target_layer]
        
        # Use [CLS] token representation (first token)
        cls_hidden_states = hidden_states[:, 0, :]
        
        return self.attribution_network(cls_hidden_states)

class MultiLayerAttributionNetwork(nn.Module):
    """
    Attribution network that combines information from multiple layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        target_layers: List[int],
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the multi-layer attribution network.
        
        Args:
            hidden_size: Size of the language model hidden states
            num_sources: Number of sources to predict
            target_layers: Which layers to use from the language model
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        self.target_layers = target_layers
        self.input_size = hidden_size * len(target_layers)
        
        self.attribution_network = AttributionNetwork(
            hidden_size=self.input_size,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    
    def forward(self, hidden_states_all_layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states_all_layers: Hidden states from all layers of the language model
                Each tensor is [batch_size, seq_len, hidden_size]
            
        Returns:
            Source logits [batch_size, num_sources]
        """
        # Get hidden states from target layers (using [CLS] token)
        cls_hidden_states = [
            hidden_states_all_layers[layer][:, 0, :]
            for layer in self.target_layers
        ]
        
        # Concatenate hidden states from all target layers
        combined_hidden_states = torch.cat(cls_hidden_states, dim=1)
        
        return self.attribution_network(combined_hidden_states)

class AttentionBasedAttributionNetwork(nn.Module):
    """
    Attribution network that uses attention to combine information from multiple layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        target_layers: List[int],
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the attention-based attribution network.
        
        Args:
            hidden_size: Size of the language model hidden states
            num_sources: Number of sources to predict
            target_layers: Which layers to use from the language model
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        self.target_layers = target_layers
        
        # Attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.attribution_network = AttributionNetwork(
            hidden_size=hidden_size,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    
    def forward(self, hidden_states_all_layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states_all_layers: Hidden states from all layers of the language model
                Each tensor is [batch_size, seq_len, hidden_size]
            
        Returns:
            Source logits [batch_size, num_sources]
        """
        # Get hidden states from target layers (using [CLS] token)
        cls_hidden_states = [
            hidden_states_all_layers[layer][:, 0, :]
            for layer in self.target_layers
        ]
        
        # Stack to create [batch_size, num_layers, hidden_size]
        stacked_hidden_states = torch.stack(cls_hidden_states, dim=1)
        
        # Self-attention across layers
        batch_size, num_layers, hidden_size = stacked_hidden_states.shape
        
        Q = self.query(stacked_hidden_states)
        K = self.key(stacked_hidden_states)
        V = self.value(stacked_hidden_states)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_size ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum
        attended_hidden_states = torch.matmul(attention_weights, V)
        
        # Average across layers to get [batch_size, hidden_size]
        avg_attended_hidden_states = attended_hidden_states.mean(dim=1)
        
        return self.attribution_network(avg_attended_hidden_states)

class AttributionGuidedModel(nn.Module):
    """
    Full model for Attribution-Guided Training, combining a language model with an attribution network.
    """
    
    def __init__(
        self,
        model_name: str,
        num_sources: int,
        attribution_type: str = "multi_layer",
        target_layers: Optional[List[int]] = None,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        lambda_attr: float = 0.1,
        freeze_lm: bool = False
    ):
        """
        Initialize the Attribution-Guided model.
        
        Args:
            model_name: Pretrained model name
            num_sources: Number of sources to predict
            attribution_type: Type of attribution network
                Options: "layer_specific", "multi_layer", "attention"
            target_layers: Which layers to use from the LM
                If None, default layers will be chosen
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
            lambda_attr: Weight of attribution loss
            freeze_lm: Whether to freeze the language model
        """
        super().__init__()
        
        # Load pretrained model
        self.language_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_sources = num_sources
        self.lambda_attr = lambda_attr
        
        # Freeze language model if specified
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Determine target layers if not provided
        num_layers = self.config.num_hidden_layers
        if target_layers is None:
            if attribution_type == "layer_specific":
                target_layers = [num_layers - 1]  # Last layer
            else:
                # Choose first, middle, and last layers
                middle_layer = num_layers // 2
                target_layers = [0, middle_layer, num_layers - 1]
        
        # Initialize attribution network
        if attribution_type == "layer_specific":
            self.attribution_network = LayerSpecificAttributionNetwork(
                hidden_size=self.hidden_size,
                num_sources=num_sources,
                target_layer=target_layers[0],
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif attribution_type == "multi_layer":
            self.attribution_network = MultiLayerAttributionNetwork(
                hidden_size=self.hidden_size,
                num_sources=num_sources,
                target_layers=target_layers,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif attribution_type == "attention":
            self.attribution_network = AttentionBasedAttributionNetwork(
                hidden_size=self.hidden_size,
                num_sources=num_sources,
                target_layers=target_layers,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attribution_type: {attribution_type}")
        
        # Task-specific prediction head (e.g., for masked language modeling)
        # This is a simplified version - we'll use the original model's prediction head in training
        
        logger.info(f"Initialized AttributionGuidedModel with {attribution_type} attribution network")
        logger.info(f"Using target layers: {target_layers}")
        logger.info(f"Number of sources: {num_sources}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            source_idx: Source indices [batch_size] (for training)
            labels: Optional labels for task-specific prediction (e.g., MLM)
            
        Returns:
            Dictionary with model outputs including:
            - task_loss: Loss for the primary task
            - attribution_loss: Loss for source attribution
            - source_logits: Logits for source prediction
            - hidden_states: Language model hidden states
        """
        # Get language model outputs
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states for attribution
        hidden_states = lm_outputs.hidden_states
        
        # Get source predictions
        source_logits = self.attribution_network(hidden_states)
        
        # Compute losses if training
        results = {
            "source_logits": source_logits,
            "hidden_states": hidden_states
        }
        
        if source_idx is not None:
            # Compute attribution loss
            attribution_loss = F.cross_entropy(source_logits, source_idx)
            results["attribution_loss"] = attribution_loss
            
            # Total loss for training
            results["loss"] = attribution_loss
            
        return results
    
    def predict_sources(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sources for input text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (source_indices, source_probs):
            - source_indices: Top-k source indices [batch_size, k]
            - source_probs: Top-k source probabilities [batch_size, k]
        """
        # Forward pass to get source logits
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            source_logits = outputs["source_logits"]
        
        # Convert to probabilities
        source_probs = F.softmax(source_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(source_probs, k=top_k, dim=-1)
        
        return top_indices, top_probs

class AttributionGuidedMLM(nn.Module):
    """
    Attribution-Guided Masked Language Model.
    Combines masked language modeling with source attribution.
    """
    
    def __init__(
        self,
        model_name: str,
        num_sources: int,
        attribution_type: str = "multi_layer",
        target_layers: Optional[List[int]] = None,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1,
        lambda_attr: float = 0.1,
        freeze_lm: bool = False
    ):
        """
        Initialize the Attribution-Guided MLM model.
        
        Args:
            model_name: Pretrained model name
            num_sources: Number of sources to predict
            attribution_type: Type of attribution network
            target_layers: Which layers to use from the LM
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
            lambda_attr: Weight of attribution loss
            freeze_lm: Whether to freeze the language model
        """
        super().__init__()
        
        # Load pretrained MLM model
        self.language_model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_sources = num_sources
        self.lambda_attr = lambda_attr
        
        # Freeze language model if specified
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Determine target layers if not provided
        num_layers = self.config.num_hidden_layers
        if target_layers is None:
            if attribution_type == "layer_specific":
                target_layers = [num_layers - 1]  # Last layer
            else:
                # Choose first, middle, and last layers
                middle_layer = num_layers // 2
                target_layers = [0, middle_layer, num_layers - 1]
        
        # Initialize attribution network
        if attribution_type == "layer_specific":
            self.attribution_network = LayerSpecificAttributionNetwork(
                hidden_size=self.hidden_size,
                num_sources=num_sources,
                target_layer=target_layers[0],
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif attribution_type == "multi_layer":
            self.attribution_network = MultiLayerAttributionNetwork(
                hidden_size=self.hidden_size,
                num_sources=num_sources,
                target_layers=target_layers,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif attribution_type == "attention":
            self.attribution_network = AttentionBasedAttributionNetwork(
                hidden_size=self.hidden_size,
                num_sources=num_sources,
                target_layers=target_layers,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attribution_type: {attribution_type}")
        
        logger.info(f"Initialized AttributionGuidedMLM with {attribution_type} attribution network")
        logger.info(f"Using target layers: {target_layers}")
        logger.info(f"Number of sources: {num_sources}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            source_idx: Source indices [batch_size] (for training)
            labels: Labels for masked language modeling
            
        Returns:
            Dictionary with model outputs
        """
        # Get language model outputs
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get MLM loss
        mlm_loss = lm_outputs.loss
        
        # Get hidden states for attribution
        hidden_states = lm_outputs.hidden_states
        
        # Get source predictions
        source_logits = self.attribution_network(hidden_states)
        
        # Compute losses if training
        results = {
            "source_logits": source_logits,
            "hidden_states": hidden_states,
            "mlm_loss": mlm_loss,
            "logits": lm_outputs.logits
        }
        
        if source_idx is not None:
            # Compute attribution loss
            attribution_loss = F.cross_entropy(source_logits, source_idx)
            results["attribution_loss"] = attribution_loss
            
            # Total loss for training (weighted sum)
            results["loss"] = mlm_loss + self.lambda_attr * attribution_loss
            
        return results
    
    def predict_sources(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sources for input text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (source_indices, source_probs):
            - source_indices: Top-k source indices [batch_size, k]
            - source_probs: Top-k source probabilities [batch_size, k]
        """
        # Forward pass to get source logits
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            source_logits = outputs["source_logits"]
        
        # Convert to probabilities
        source_probs = F.softmax(source_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(source_probs, k=top_k, dim=-1)
        
        return top_indices, top_probs

# Baseline Models for Comparison

class PostHocAttributionModel(nn.Module):
    """
    Post-hoc attribution model that applies attribution after standard training.
    This simulates approaches like influence functions.
    """
    
    def __init__(
        self,
        model_name: str,
        num_sources: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the post-hoc attribution model.
        
        Args:
            model_name: Pretrained model name
            num_sources: Number of sources to predict
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained model
        self.language_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_sources = num_sources
        
        # Freeze language model since it's post-hoc
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Post-hoc attribution network using last layer
        self.attribution_network = AttributionNetwork(
            hidden_size=self.hidden_size,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        logger.info(f"Initialized PostHocAttributionModel")
        logger.info(f"Number of sources: {num_sources}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            source_idx: Source indices [batch_size] (for training)
            
        Returns:
            Dictionary with model outputs
        """
        # Get language model outputs
        with torch.no_grad():
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden states for attribution (last layer, CLS token)
        hidden_states = lm_outputs.last_hidden_state[:, 0, :]
        
        # Get source predictions
        source_logits = self.attribution_network(hidden_states)
        
        # Compute losses if training
        results = {
            "source_logits": source_logits
        }
        
        if source_idx is not None:
            # Compute attribution loss
            attribution_loss = F.cross_entropy(source_logits, source_idx)
            results["attribution_loss"] = attribution_loss
            results["loss"] = attribution_loss
            
        return results
    
    def predict_sources(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sources for input text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (source_indices, source_probs):
            - source_indices: Top-k source indices [batch_size, k]
            - source_probs: Top-k source probabilities [batch_size, k]
        """
        # Forward pass to get source logits
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            source_logits = outputs["source_logits"]
        
        # Convert to probabilities
        source_probs = F.softmax(source_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(source_probs, k=top_k, dim=-1)
        
        return top_indices, top_probs

class DataShapleySimulator(nn.Module):
    """
    Simulator for Data Shapley attribution method.
    Real Data Shapley requires multiple model runs, but we simulate it with a proxy task.
    """
    
    def __init__(
        self,
        model_name: str,
        num_sources: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the Data Shapley simulator.
        
        Args:
            model_name: Pretrained model name
            num_sources: Number of sources to predict
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained model
        self.language_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_sources = num_sources
        
        # Freeze base model
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Feature importance layer (simulates feature attribution)
        self.feature_importance = nn.Parameter(
            torch.ones(self.hidden_size) / self.hidden_size
        )
        
        # Attribution network
        self.attribution_network = AttributionNetwork(
            hidden_size=self.hidden_size,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        logger.info(f"Initialized DataShapleySimulator")
        logger.info(f"Number of sources: {num_sources}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            source_idx: Source indices [batch_size] (for training)
            
        Returns:
            Dictionary with model outputs
        """
        # Get language model outputs
        with torch.no_grad():
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden states for attribution (last layer, CLS token)
        hidden_states = lm_outputs.last_hidden_state[:, 0, :]
        
        # Apply feature importance weighting (simulates Shapley values)
        weighted_hidden_states = hidden_states * self.feature_importance
        
        # Get source predictions
        source_logits = self.attribution_network(weighted_hidden_states)
        
        # Compute losses if training
        results = {
            "source_logits": source_logits
        }
        
        if source_idx is not None:
            # Compute attribution loss
            attribution_loss = F.cross_entropy(source_logits, source_idx)
            
            # Add L1 regularization for sparsity (common in Shapley)
            l1_reg = 0.01 * torch.norm(self.feature_importance, p=1)
            
            results["attribution_loss"] = attribution_loss
            results["l1_reg"] = l1_reg
            results["loss"] = attribution_loss + l1_reg
            
        return results
    
    def predict_sources(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sources for input text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (source_indices, source_probs):
            - source_indices: Top-k source indices [batch_size, k]
            - source_probs: Top-k source probabilities [batch_size, k]
        """
        # Forward pass to get source logits
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            source_logits = outputs["source_logits"]
        
        # Convert to probabilities
        source_probs = F.softmax(source_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(source_probs, k=top_k, dim=-1)
        
        return top_indices, top_probs

class MinimalSubsetAttributionModel(nn.Module):
    """
    Attribution model based on the minimal interpretable subset approach.
    Simulates the LiMA method from Chen et al. (2025).
    """
    
    def __init__(
        self,
        model_name: str,
        num_sources: int,
        subset_size: int = 32,  # Size of the minimal subset
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        """
        Initialize the minimal subset attribution model.
        
        Args:
            model_name: Pretrained model name
            num_sources: Number of sources to predict
            subset_size: Size of the minimal subset
            hidden_dims: Hidden layer dimensions for the attribution network
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained model
        self.language_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_sources = num_sources
        self.subset_size = subset_size
        
        # Freeze language model
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Subset selection layer
        self.subset_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Attribution network
        self.attribution_network = AttributionNetwork(
            hidden_size=self.hidden_size,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        logger.info(f"Initialized MinimalSubsetAttributionModel")
        logger.info(f"Number of sources: {num_sources}")
        logger.info(f"Subset size: {subset_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            source_idx: Source indices [batch_size] (for training)
            
        Returns:
            Dictionary with model outputs
        """
        # Get language model outputs
        with torch.no_grad():
            lm_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get all hidden states (last layer)
        hidden_states = lm_outputs.last_hidden_state
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute importance scores for each token
        importance_scores = self.subset_selector(hidden_states).squeeze(-1)
        
        # Apply attention mask
        importance_scores = importance_scores.masked_fill(
            attention_mask == 0, -1e9
        )
        
        # Get top-k tokens by importance
        _, top_indices = torch.topk(
            importance_scores, k=min(self.subset_size, seq_len), dim=1
        )
        
        # Gather the hidden states for the top-k tokens
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(top_indices)
        top_hidden_states = hidden_states[batch_indices, top_indices]
        
        # Pool the selected token representations
        pooled_states = torch.mean(top_hidden_states, dim=1)
        
        # Get source predictions
        source_logits = self.attribution_network(pooled_states)
        
        # Compute losses if training
        results = {
            "source_logits": source_logits,
            "importance_scores": importance_scores
        }
        
        if source_idx is not None:
            # Compute attribution loss
            attribution_loss = F.cross_entropy(source_logits, source_idx)
            
            # Add sparsity regularization for importance scores
            sorted_scores, _ = torch.sort(importance_scores, dim=1, descending=True)
            threshold = sorted_scores[:, self.subset_size].unsqueeze(1)
            mask = (importance_scores > threshold).float()
            
            # L1 regularization to encourage sparsity beyond top-k
            l1_reg = 0.01 * torch.mean(importance_scores * (1 - mask))
            
            results["attribution_loss"] = attribution_loss
            results["l1_reg"] = l1_reg
            results["loss"] = attribution_loss + l1_reg
            
        return results
    
    def predict_sources(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sources for input text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (source_indices, source_probs):
            - source_indices: Top-k source indices [batch_size, k]
            - source_probs: Top-k source probabilities [batch_size, k]
        """
        # Forward pass to get source logits
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            source_logits = outputs["source_logits"]
        
        # Convert to probabilities
        source_probs = F.softmax(source_logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(source_probs, k=top_k, dim=-1)
        
        return top_indices, top_probs

if __name__ == "__main__":
    # Test model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Test with small model
    model_name = "distilroberta-base"
    num_sources = 100
    
    # Test AGT model
    agt_model = AttributionGuidedMLM(
        model_name=model_name,
        num_sources=num_sources,
        attribution_type="multi_layer"
    )
    
    logger.info(f"AttributionGuidedMLM parameters: {sum(p.numel() for p in agt_model.parameters())}")
    
    # Test baseline model
    baseline_model = PostHocAttributionModel(
        model_name=model_name,
        num_sources=num_sources
    )
    
    logger.info(f"PostHocAttributionModel parameters: {sum(p.numel() for p in baseline_model.parameters())}")
    
    logger.info("Model initialization tests passed")