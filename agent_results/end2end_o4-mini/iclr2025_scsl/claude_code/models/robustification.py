"""
Robustification methods for mitigating the effect of spurious correlations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable

from .models import SimpleImageClassifier, SimpleTextClassifier, MultimodalClassifier, Adversary


def create_adversarial_model(
    base_model: nn.Module,
    feature_dim: int,
    num_attributes: int,
    adv_hidden_dim: int = 256,
    modality: str = "multimodal"
) -> Tuple[nn.Module, nn.Module]:
    """
    Create an adversarial debiasing model by adding an adversary to a base model.
    
    Args:
        base_model: Base classifier model
        feature_dim: Dimension of the feature vector
        num_attributes: Number of attribute classes to predict
        adv_hidden_dim: Dimension of the adversary's hidden layer
        modality: Data modality ("image", "text", or "multimodal")
        
    Returns:
        Tuple of (base model, adversary)
    """
    # Create adversary
    adversary = Adversary(
        input_dim=feature_dim,
        hidden_dim=adv_hidden_dim,
        num_attributes=num_attributes
    )
    
    return base_model, adversary


class IRMModel(nn.Module):
    """
    Model wrapper for Invariant Risk Minimization (IRM).
    """
    
    def __init__(self, base_model: nn.Module, modality: str = "multimodal"):
        """
        Initialize the IRM model.
        
        Args:
            base_model: Base classifier model
            modality: Data modality ("image", "text", or "multimodal")
        """
        super().__init__()
        self.base_model = base_model
        self.modality = modality
    
    def get_features(self, *args) -> torch.Tensor:
        """
        Extract features from the input.
        
        Args:
            *args: Input tensors based on modality
            
        Returns:
            Feature tensor
        """
        if self.modality == "multimodal":
            return self.base_model.get_features(args[0], args[1])
        else:
            return self.base_model.get_features(args[0])
    
    def forward(self, *args, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            *args: Input tensors based on modality
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        if self.modality == "multimodal":
            if return_features:
                return self.base_model(args[0], args[1], return_features=True)
            else:
                return self.base_model(args[0], args[1])
        else:
            if return_features:
                return self.base_model(args[0], return_features=True)
            else:
                return self.base_model(args[0])


class GroupDROModel(nn.Module):
    """
    Model wrapper for Group Distributionally Robust Optimization (Group-DRO).
    """
    
    def __init__(self, base_model: nn.Module, modality: str = "multimodal"):
        """
        Initialize the Group-DRO model.
        
        Args:
            base_model: Base classifier model
            modality: Data modality ("image", "text", or "multimodal")
        """
        super().__init__()
        self.base_model = base_model
        self.modality = modality
    
    def get_features(self, *args) -> torch.Tensor:
        """
        Extract features from the input.
        
        Args:
            *args: Input tensors based on modality
            
        Returns:
            Feature tensor
        """
        if self.modality == "multimodal":
            return self.base_model.get_features(args[0], args[1])
        else:
            return self.base_model.get_features(args[0])
    
    def forward(self, *args, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            *args: Input tensors based on modality
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        if self.modality == "multimodal":
            if return_features:
                return self.base_model(args[0], args[1], return_features=True)
            else:
                return self.base_model(args[0], args[1])
        else:
            if return_features:
                return self.base_model(args[0], return_features=True)
            else:
                return self.base_model(args[0])


class AdversarialModel(nn.Module):
    """
    Model wrapper for Adversarial Feature Debiasing.
    """
    
    def __init__(self, base_model: nn.Module, modality: str = "multimodal"):
        """
        Initialize the Adversarial model.
        
        Args:
            base_model: Base classifier model
            modality: Data modality ("image", "text", or "multimodal")
        """
        super().__init__()
        self.base_model = base_model
        self.modality = modality
    
    def get_features(self, *args) -> torch.Tensor:
        """
        Extract features from the input.
        
        Args:
            *args: Input tensors based on modality
            
        Returns:
            Feature tensor
        """
        if self.modality == "multimodal":
            return self.base_model.get_features(args[0], args[1])
        else:
            return self.base_model.get_features(args[0])
    
    def forward(self, *args, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            *args: Input tensors based on modality
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        if self.modality == "multimodal":
            if return_features:
                outputs = self.base_model(args[0], args[1])
                features = self.get_features(args[0], args[1])
                return outputs, features
            else:
                return self.base_model(args[0], args[1])
        else:
            if return_features:
                outputs = self.base_model(args[0])
                features = self.get_features(args[0])
                return outputs, features
            else:
                return self.base_model(args[0])


class ContrastiveModel(nn.Module):
    """
    Model wrapper for Contrastive Augmentation.
    """
    
    def __init__(self, base_model: nn.Module, modality: str = "multimodal"):
        """
        Initialize the Contrastive model.
        
        Args:
            base_model: Base classifier model
            modality: Data modality ("image", "text", or "multimodal")
        """
        super().__init__()
        self.base_model = base_model
        self.modality = modality
    
    def get_features(self, *args) -> torch.Tensor:
        """
        Extract features from the input.
        
        Args:
            *args: Input tensors based on modality
            
        Returns:
            Feature tensor
        """
        if self.modality == "multimodal":
            return self.base_model.get_features(args[0], args[1])
        else:
            return self.base_model.get_features(args[0])
    
    def forward(self, *args, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            *args: Input tensors based on modality
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        if self.modality == "multimodal":
            if return_features:
                return self.base_model(args[0], args[1], return_features=True)
            else:
                return self.base_model(args[0], args[1])
        else:
            if return_features:
                return self.base_model(args[0], return_features=True)
            else:
                return self.base_model(args[0])