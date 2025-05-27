"""
Models for SpurGen benchmark experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union

class SimpleImageClassifier(nn.Module):
    """
    Simple CNN classifier for image data.
    """
    
    def __init__(self, num_classes: int = 10, feature_dim: int = 512):
        """
        Initialize the image classifier.
        
        Args:
            num_classes: Number of output classes
            feature_dim: Dimension of the feature vector
        """
        super().__init__()
        
        # Use a pre-trained ResNet as feature extractor
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension after pooling
        self.feature_dim = feature_dim
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        # Extract features
        features = self.get_features(x)
        
        # Classify
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        else:
            return outputs


class SimpleTextClassifier(nn.Module):
    """
    Simple text classifier using bag-of-words and MLP.
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 300, hidden_dim: int = 512, num_classes: int = 10):
        """
        Initialize the text classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of the word embeddings
            hidden_dim: Dimension of the hidden layer
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input.
        
        Args:
            x: Input tensor (tokenized text)
            
        Returns:
            Feature tensor
        """
        # Embed and pool
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)  # Simple mean pooling
        
        # Extract features
        features = self.feature_extractor(pooled)
        
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (tokenized text)
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        # Extract features
        features = self.get_features(x)
        
        # Classify
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        else:
            return outputs


class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier combining image and text.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        image_feature_dim: int = 512,
        text_feature_dim: int = 512,
        fusion_dim: int = 512,
        vocab_size: int = 10000,
        embed_dim: int = 300
    ):
        """
        Initialize the multimodal classifier.
        
        Args:
            num_classes: Number of output classes
            image_feature_dim: Dimension of the image feature vector
            text_feature_dim: Dimension of the text feature vector
            fusion_dim: Dimension of the fused feature vector
            vocab_size: Size of the text vocabulary
            embed_dim: Dimension of the word embeddings
        """
        super().__init__()
        
        # Image feature extractor
        resnet = models.resnet18(pretrained=True)
        self.image_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Text feature extractor
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, text_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)
    
    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Image feature tensor
        """
        features = self.image_feature_extractor(image)
        features = features.view(features.size(0), -1)
        
        return features
    
    def get_text_features(self, text: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the text.
        
        Args:
            text: Input text tensor (tokenized)
            
        Returns:
            Text feature tensor
        """
        embedded = self.embedding(text)
        pooled = torch.mean(embedded, dim=1)  # Simple mean pooling
        features = self.text_feature_extractor(pooled)
        
        return features
    
    def get_features(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse features from both modalities.
        
        Args:
            image: Input image tensor
            text: Input text tensor (tokenized)
            
        Returns:
            Fused feature tensor
        """
        # Extract features from both modalities
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(text)
        
        # Concatenate features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        return fused_features
    
    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            image: Input image tensor
            text: Input text tensor (tokenized)
            return_features: Whether to return features in addition to outputs
            
        Returns:
            Model outputs (and features if return_features=True)
        """
        # Extract and fuse features
        fused_features = self.get_features(image, text)
        
        # Classify
        outputs = self.classifier(fused_features)
        
        if return_features:
            return outputs, fused_features
        else:
            return outputs


class Adversary(nn.Module):
    """
    Adversarial discriminator for predicting spurious attributes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_attributes: int):
        """
        Initialize the adversary.
        
        Args:
            input_dim: Dimension of the input features
            hidden_dim: Dimension of the hidden layer
            num_attributes: Number of attribute classes to predict
        """
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_attributes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adversary.
        
        Args:
            x: Input feature tensor
            
        Returns:
            Predicted attribute probabilities
        """
        return self.discriminator(x)