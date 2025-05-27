"""
Baseline models for Neural Weight Archeology experiments.

This module defines simple baseline models for predicting model properties
from weight patterns, including:
- Simple statistical features from weights
- PCA-based dimensionality reduction of weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Tuple, List, Union, Optional

class WeightStatisticsModel(nn.Module):
    """
    Baseline model that uses simple statistical features from weights
    for property prediction.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        num_classes: Optional[Dict[str, int]] = None,
        num_regression_targets: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes or {}
        self.num_regression_targets = num_regression_targets or 0
        
        # Base network for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification heads (one per class type)
        self.classification_heads = nn.ModuleDict()
        for class_name, num_class in self.num_classes.items():
            self.classification_heads[class_name] = nn.Linear(hidden_dim, num_class)
        
        # Regression head
        if self.num_regression_targets > 0:
            self.regression_head = nn.Linear(hidden_dim, self.num_regression_targets)
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through the model
        
        Args:
            batch: Dictionary containing:
                - 'features': Tensor of shape [batch_size, input_dim]
                
        Returns:
            Dictionary containing:
                - 'classification': Dict of {class_name: logits} for each class
                - 'regression': Regression outputs if regression targets exist
        """
        # Extract features
        features = batch['features']
        x = self.feature_extractor(features)
        
        outputs = {}
        
        # Classification outputs
        if self.num_classes:
            classification_outputs = {}
            for class_name, head in self.classification_heads.items():
                classification_outputs[class_name] = head(x)
            outputs['classification'] = classification_outputs
        
        # Regression outputs
        if self.num_regression_targets > 0:
            outputs['regression'] = self.regression_head(x)
        
        return outputs
    
    def loss_function(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Compute the loss for model outputs
        
        Args:
            outputs: Dictionary from the forward pass
            batch: Dictionary containing ground truth labels
            
        Returns:
            Combined loss value
        """
        loss = 0.0
        
        # Classification loss
        if 'classification' in outputs:
            for class_name, logits in outputs['classification'].items():
                target = batch[f'class_{class_name}']
                class_loss = F.cross_entropy(logits, target)
                loss += class_loss
        
        # Regression loss
        if 'regression' in outputs:
            regression_targets = batch['regression_targets']
            regression_loss = F.mse_loss(outputs['regression'], regression_targets)
            loss += regression_loss
        
        return loss

class PCAPredictionModel(nn.Module):
    """
    Baseline model that uses PCA-based dimensionality reduction of weights
    for property prediction.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        num_classes: Optional[Dict[str, int]] = None,
        num_regression_targets: Optional[int] = None,
        dropout: float = 0.1,
        pca_components: int = 50
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes or {}
        self.num_regression_targets = num_regression_targets or 0
        self.pca_components = min(pca_components, input_dim)
        
        # PCA is not a learnable part, so we'll just use it in preprocessing
        # No parameters needed here, as PCA is fitted at runtime
        
        # Base network for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification heads (one per class type)
        self.classification_heads = nn.ModuleDict()
        for class_name, num_class in self.num_classes.items():
            self.classification_heads[class_name] = nn.Linear(hidden_dim, num_class)
        
        # Regression head
        if self.num_regression_targets > 0:
            self.regression_head = nn.Linear(hidden_dim, self.num_regression_targets)
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through the model
        
        Args:
            batch: Dictionary containing:
                - 'features': Tensor of shape [batch_size, input_dim]
                
        Returns:
            Dictionary containing:
                - 'classification': Dict of {class_name: logits} for each class
                - 'regression': Regression outputs if regression targets exist
        """
        # Extract features
        features = batch['features']
        
        # Apply PCA on the fly during training if needed
        # In practice, we would fit PCA once on the training data
        # and then transform all inputs, but for this example we just
        # pass through the input features
        
        x = self.feature_extractor(features)
        
        outputs = {}
        
        # Classification outputs
        if self.num_classes:
            classification_outputs = {}
            for class_name, head in self.classification_heads.items():
                classification_outputs[class_name] = head(x)
            outputs['classification'] = classification_outputs
        
        # Regression outputs
        if self.num_regression_targets > 0:
            outputs['regression'] = self.regression_head(x)
        
        return outputs
    
    def loss_function(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Compute the loss for model outputs
        
        Args:
            outputs: Dictionary from the forward pass
            batch: Dictionary containing ground truth labels
            
        Returns:
            Combined loss value
        """
        loss = 0.0
        
        # Classification loss
        if 'classification' in outputs:
            for class_name, logits in outputs['classification'].items():
                target = batch[f'class_{class_name}']
                class_loss = F.cross_entropy(logits, target)
                loss += class_loss
        
        # Regression loss
        if 'regression' in outputs:
            regression_targets = batch['regression_targets']
            regression_loss = F.mse_loss(outputs['regression'], regression_targets)
            loss += regression_loss
        
        return loss