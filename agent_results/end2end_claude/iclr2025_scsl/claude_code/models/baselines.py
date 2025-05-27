#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of baseline models for comparison with CIMRL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from transformers import AutoModel

class StandardMultiModal(nn.Module):
    """
    Standard multi-modal model without robustness interventions.
    This serves as a baseline for comparison with CIMRL.
    """
    
    def __init__(self, config):
        """
        Initialize the StandardMultiModal model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(StandardMultiModal, self).__init__()
        
        self.config = config
        
        # Modality-specific encoders (using pre-trained models)
        if 'vision' in config['modalities']:
            # Vision encoder (e.g., ViT or ResNet)
            if config['model']['vision_encoder'] == 'vit':
                self.vision_encoder = AutoModel.from_pretrained('google/vit-base-patch16-224')
                self.vision_dim = 768
            else:  # ResNet
                self.vision_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                # Remove the final classification layer
                self.vision_encoder = nn.Sequential(*list(self.vision_encoder.children())[:-1])
                self.vision_dim = 2048
        
        if 'text' in config['modalities']:
            # Text encoder (e.g., BERT or RoBERTa)
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder'])
            self.text_dim = self.text_encoder.config.hidden_size
        
        # Projection dimensions
        self.shared_dim = config['model']['shared_dim']
        
        # Define combined dimension based on available modalities
        combined_dim = 0
        if 'vision' in config['modalities']:
            combined_dim += self.vision_dim
        if 'text' in config['modalities']:
            combined_dim += self.text_dim
        
        # Fusion layer for combining modalities
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.shared_dim, self.shared_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(self.shared_dim, config['model']['num_classes'])
    
    def forward(self, batch, compute_loss=True):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input data for each modality
            compute_loss: Whether to compute the loss or just the predictions
            
        Returns:
            Dictionary containing model outputs and loss
        """
        outputs = {}
        features = []
        
        # Get features from each modality encoder
        if 'vision' in self.config['modalities']:
            vision_input = batch['vision']
            vision_features = self.vision_encoder(vision_input).pooler_output
            if len(vision_features.shape) == 3:  # For models that return sequences
                vision_features = vision_features.mean(dim=1)
            features.append(vision_features)
        
        if 'text' in self.config['modalities']:
            text_input = batch['text']
            text_features = self.text_encoder(**text_input).pooler_output
            features.append(text_features)
        
        # Combine features from all modalities
        combined_features = torch.cat(features, dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        outputs['pred'] = logits
        
        # Compute loss if required
        if compute_loss and 'labels' in batch:
            labels = batch['labels']
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class GroupDRO(nn.Module):
    """
    Group Distributionally Robust Optimization (DRO) model.
    This model uses group annotations to reweight examples during training.
    """
    
    def __init__(self, config):
        """
        Initialize the GroupDRO model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(GroupDRO, self).__init__()
        
        # Use the standard model as the base
        self.model = StandardMultiModal(config)
        
        # Group DRO specific parameters
        self.num_groups = config['model'].get('num_groups', 4)  # Default to 4 groups
        self.group_weights = nn.Parameter(torch.ones(self.num_groups), requires_grad=False)
        self.eta = config['training'].get('dro_eta', 1.0)  # Step size for group weights
    
    def forward(self, batch, compute_loss=True):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input data for each modality and group labels
            compute_loss: Whether to compute the loss or just the predictions
            
        Returns:
            Dictionary containing model outputs and loss
        """
        outputs = self.model(batch, compute_loss=False)
        
        # Compute loss if required
        if compute_loss and 'labels' in batch:
            labels = batch['labels']
            logits = outputs['pred']
            
            if 'group_labels' in batch:
                group_labels = batch['group_labels']
                
                # Compute per-example loss
                per_example_loss = F.cross_entropy(logits, labels, reduction='none')
                
                # Compute per-group loss
                group_losses = torch.zeros(self.num_groups, device=logits.device)
                group_counts = torch.zeros(self.num_groups, device=logits.device)
                
                for g in range(self.num_groups):
                    group_mask = (group_labels == g)
                    if group_mask.sum() > 0:
                        group_losses[g] = per_example_loss[group_mask].mean()
                        group_counts[g] = group_mask.sum()
                
                # Update group weights
                if self.training:
                    self.group_weights.data = self.group_weights * torch.exp(self.eta * group_losses.detach())
                    self.group_weights.data = self.group_weights / self.group_weights.sum()
                
                # Weighted group loss
                loss = (self.group_weights * group_losses).sum()
                
                outputs['group_losses'] = group_losses
                outputs['group_weights'] = self.group_weights
            else:
                # Fallback to standard cross-entropy if group labels not provided
                loss = F.cross_entropy(logits, labels)
            
            outputs['loss'] = loss
        
        return outputs


class JTT(nn.Module):
    """
    Just Train Twice (JTT) model.
    This model trains first on all data, then upweights examples misclassified by the first model.
    """
    
    def __init__(self, config):
        """
        Initialize the JTT model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(JTT, self).__init__()
        
        # Use the standard model as the base
        self.model = StandardMultiModal(config)
        
        # JTT specific parameters
        self.upweight_factor = config['training'].get('jtt_upweight_factor', 5.0)
        self.first_model_trained = config['training'].get('jtt_first_model_trained', False)
        self.error_indices = None
    
    def forward(self, batch, compute_loss=True):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input data for each modality
            compute_loss: Whether to compute the loss or just the predictions
            
        Returns:
            Dictionary containing model outputs and loss
        """
        outputs = self.model(batch, compute_loss=False)
        
        # Compute loss if required
        if compute_loss and 'labels' in batch:
            labels = batch['labels']
            logits = outputs['pred']
            
            if self.training and self.error_indices is not None and 'indices' in batch:
                # Second phase of JTT: upweight examples misclassified by the first model
                indices = batch['indices']
                weights = torch.ones(len(indices), device=logits.device)
                
                for i, idx in enumerate(indices):
                    if idx in self.error_indices:
                        weights[i] = self.upweight_factor
                
                # Compute weighted loss
                per_example_loss = F.cross_entropy(logits, labels, reduction='none')
                loss = (weights * per_example_loss).mean()
                
                outputs['upweighted'] = weights
            else:
                # Standard cross-entropy (first phase of JTT)
                loss = F.cross_entropy(logits, labels)
            
            outputs['loss'] = loss
        
        return outputs
    
    def set_error_indices(self, error_indices):
        """
        Set the indices of examples misclassified by the first model.
        
        Args:
            error_indices: Set of indices of misclassified examples
        """
        self.error_indices = error_indices
        self.first_model_trained = True


class CCRMultiModal(nn.Module):
    """
    Causally Calibrated Robust Classifier (CCR) adapted for multi-modal data.
    Based on Zhou & Zhu (2024) but extended to handle multiple modalities.
    """
    
    def __init__(self, config):
        """
        Initialize the CCRMultiModal model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(CCRMultiModal, self).__init__()
        
        self.config = config
        
        # Modality-specific encoders (using pre-trained models)
        if 'vision' in config['modalities']:
            # Vision encoder (e.g., ViT or ResNet)
            if config['model']['vision_encoder'] == 'vit':
                self.vision_encoder = AutoModel.from_pretrained('google/vit-base-patch16-224')
                self.vision_dim = 768
            else:  # ResNet
                self.vision_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                # Remove the final classification layer
                self.vision_encoder = nn.Sequential(*list(self.vision_encoder.children())[:-1])
                self.vision_dim = 2048
        
        if 'text' in config['modalities']:
            # Text encoder (e.g., BERT or RoBERTa)
            self.text_encoder = AutoModel.from_pretrained(config['model']['text_encoder'])
            self.text_dim = self.text_encoder.config.hidden_size
        
        # Projection dimensions
        self.shared_dim = config['model']['shared_dim']
        
        # Define combined dimension based on available modalities
        combined_dim = 0
        if 'vision' in config['modalities']:
            combined_dim += self.vision_dim
        if 'text' in config['modalities']:
            combined_dim += self.text_dim
        
        # Fusion layer for combining modalities
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.shared_dim, self.shared_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(self.shared_dim, config['model']['num_classes'])
        
        # CCR-specific components
        
        # Feature selector (for causal feature selection)
        self.feature_selector = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.Sigmoid()
        )
        
        # Propensity estimator (for estimating probability of spurious correlation)
        self.propensity_estimator = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim // 2),
            nn.ReLU(),
            nn.Linear(self.shared_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # CCR hyperparameters
        self.feature_reg = config['training'].get('ccr_feature_reg', 0.01)
        self.propensity_threshold = config['training'].get('ccr_propensity_threshold', 0.1)
    
    def forward(self, batch, compute_loss=True):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input data for each modality
            compute_loss: Whether to compute the loss or just the predictions
            
        Returns:
            Dictionary containing model outputs and loss
        """
        outputs = {}
        features = []
        
        # Get features from each modality encoder
        if 'vision' in self.config['modalities']:
            vision_input = batch['vision']
            vision_features = self.vision_encoder(vision_input).pooler_output
            if len(vision_features.shape) == 3:  # For models that return sequences
                vision_features = vision_features.mean(dim=1)
            features.append(vision_features)
        
        if 'text' in self.config['modalities']:
            text_input = batch['text']
            text_features = self.text_encoder(**text_input).pooler_output
            features.append(text_features)
        
        # Combine features from all modalities
        combined_features = torch.cat(features, dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion(combined_features)
        
        # Apply feature selection
        feature_weights = self.feature_selector(fused_features)
        selected_features = fused_features * feature_weights
        
        # Estimate propensity scores (probability of spurious correlation)
        propensity_scores = self.propensity_estimator(fused_features)
        
        # Classification
        logits = self.classifier(selected_features)
        outputs['pred'] = logits
        outputs['feature_weights'] = feature_weights
        outputs['propensity_scores'] = propensity_scores
        
        # Compute loss if required
        if compute_loss and 'labels' in batch:
            labels = batch['labels']
            
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            
            # Inverse propensity weighting (IPW)
            weights = torch.ones_like(propensity_scores)
            low_propensity_mask = (propensity_scores < self.propensity_threshold).squeeze()
            
            if low_propensity_mask.sum() > 0:
                # Upweight examples with low propensity scores
                weights[low_propensity_mask] = 1.0 / torch.clamp(propensity_scores[low_propensity_mask], min=0.05)
            
            # Normalize weights
            weights = weights / weights.mean()
            
            # Weighted cross-entropy loss
            weighted_loss = (weights.squeeze() * ce_loss).mean()
            
            # Feature sparsity regularization
            feature_reg_loss = torch.norm(feature_weights, p=1, dim=1).mean()
            
            # Total loss
            loss = weighted_loss + self.feature_reg * feature_reg_loss
            
            outputs['ce_loss'] = ce_loss.mean()
            outputs['weighted_loss'] = weighted_loss
            outputs['feature_reg_loss'] = feature_reg_loss
            outputs['loss'] = loss
        
        return outputs