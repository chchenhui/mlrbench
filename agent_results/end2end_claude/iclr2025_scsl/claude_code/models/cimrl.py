#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Causally-Informed Multi-Modal Representation Learning (CIMRL) model
for mitigating shortcut learning in multi-modal models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from transformers import AutoModel

class CIMRL(nn.Module):
    """
    Causally-Informed Multi-Modal Representation Learning (CIMRL) model
    for mitigating shortcut learning in multi-modal models.
    """
    
    def __init__(self, config):
        """
        Initialize the CIMRL model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super(CIMRL, self).__init__()
        
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
        
        # Shared encoder for cross-modal features
        modality_dims = []
        if 'vision' in config['modalities']:
            modality_dims.append(self.vision_dim)
        if 'text' in config['modalities']:
            modality_dims.append(self.text_dim)
        
        # Define dimensions for the shared and modality-specific projections
        self.shared_dim = config['model']['shared_dim']
        self.modality_specific_dim = config['model']['modality_specific_dim']
        
        # Projection layers
        if 'vision' in config['modalities']:
            self.vision_projection = nn.Sequential(
                nn.Linear(self.vision_dim, self.modality_specific_dim),
                nn.LayerNorm(self.modality_specific_dim),
                nn.ReLU()
            )
        
        if 'text' in config['modalities']:
            self.text_projection = nn.Sequential(
                nn.Linear(self.text_dim, self.modality_specific_dim),
                nn.LayerNorm(self.modality_specific_dim),
                nn.ReLU()
            )
        
        # Shared encoder with cross-attention
        self.shared_encoder = CrossModalEncoder(
            modality_dims,
            self.shared_dim,
            num_layers=config['model']['num_shared_layers'],
            num_heads=config['model']['num_attention_heads'],
            dropout=config['model']['dropout']
        )
        
        # Prediction heads
        if 'vision' in config['modalities']:
            self.vision_head = nn.Linear(self.modality_specific_dim, config['model']['num_classes'])
        
        if 'text' in config['modalities']:
            self.text_head = nn.Linear(self.modality_specific_dim, config['model']['num_classes'])
        
        self.shared_head = nn.Linear(self.shared_dim, config['model']['num_classes'])
        
        # Combined prediction head
        combined_dim = self.shared_dim
        if 'vision' in config['modalities']:
            combined_dim += self.modality_specific_dim
        if 'text' in config['modalities']:
            combined_dim += self.modality_specific_dim
        
        self.combined_head = nn.Linear(combined_dim, config['model']['num_classes'])
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(config['model'].get('temperature', 0.07)))
        
        # Orthogonality regularization weight
        self.ortho_weight = config['training'].get('ortho_weight', 0.1)
        
        # Weights for different loss components
        self.contrastive_weight = config['training'].get('contrastive_weight', 1.0)
        self.modality_disentanglement_weight = config['training'].get('md_weight', 1.0)
        self.intervention_weight = config['training'].get('intervention_weight', 1.0)
    
    def forward(self, batch, compute_loss=True, intervention=None):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input data for each modality
            compute_loss: Whether to compute the loss or just the predictions
            intervention: Optional intervention on features for counterfactual analysis
            
        Returns:
            Dictionary containing model outputs, representations, and loss
        """
        outputs = {}
        representations = {}
        
        # Get representations from each modality encoder
        if 'vision' in self.config['modalities']:
            vision_input = batch['vision']
            vision_features = self.vision_encoder(vision_input).pooler_output 
            if len(vision_features.shape) == 3:  # For models that return sequences
                vision_features = vision_features.mean(dim=1)
            vision_specific = self.vision_projection(vision_features)
            representations['vision_specific'] = vision_specific
        
        if 'text' in self.config['modalities']:
            text_input = batch['text']
            text_features = self.text_encoder(**text_input).pooler_output
            text_specific = self.text_projection(text_features)
            representations['text_specific'] = text_specific
        
        # Get shared representation using cross-modal encoder
        modality_features = []
        if 'vision' in self.config['modalities']:
            modality_features.append(vision_features)
        if 'text' in self.config['modalities']:
            modality_features.append(text_features)
        
        shared_features = self.shared_encoder(modality_features)
        representations['shared'] = shared_features
        
        # Apply intervention if specified (for counterfactual analysis)
        if intervention is not None:
            if intervention.get('modality') == 'vision' and 'vision' in self.config['modalities']:
                vision_specific = intervention['features']
                representations['vision_specific'] = vision_specific
            elif intervention.get('modality') == 'text' and 'text' in self.config['modalities']:
                text_specific = intervention['features']
                representations['text_specific'] = text_specific
            elif intervention.get('modality') == 'shared':
                shared_features = intervention['features']
                representations['shared'] = shared_features
        
        # Modality-specific predictions
        if 'vision' in self.config['modalities']:
            vision_pred = self.vision_head(vision_specific)
            outputs['vision_pred'] = vision_pred
        
        if 'text' in self.config['modalities']:
            text_pred = self.text_head(text_specific)
            outputs['text_pred'] = text_pred
        
        # Shared representation prediction
        shared_pred = self.shared_head(shared_features)
        outputs['shared_pred'] = shared_pred
        
        # Combined prediction
        combined_features = [shared_features]
        if 'vision' in self.config['modalities']:
            combined_features.append(vision_specific)
        if 'text' in self.config['modalities']:
            combined_features.append(text_specific)
        
        combined_features = torch.cat(combined_features, dim=1)
        combined_pred = self.combined_head(combined_features)
        outputs['combined_pred'] = combined_pred
        
        # Main prediction output
        outputs['pred'] = combined_pred
        
        # Compute loss if required
        if compute_loss and 'labels' in batch:
            labels = batch['labels']
            
            # Classification losses for each prediction head
            losses = {}
            
            if 'vision' in self.config['modalities']:
                losses['vision_loss'] = F.cross_entropy(vision_pred, labels)
            
            if 'text' in self.config['modalities']:
                losses['text_loss'] = F.cross_entropy(text_pred, labels)
            
            losses['shared_loss'] = F.cross_entropy(shared_pred, labels)
            losses['combined_loss'] = F.cross_entropy(combined_pred, labels)
            
            # Main classification loss
            losses['ce_loss'] = losses['combined_loss']
            
            # Contrastive invariance loss (if perturbed samples provided)
            if 'perturbed' in batch:
                contrastive_loss = 0
                
                # Forward pass on perturbed inputs
                with torch.no_grad():
                    perturbed_outputs = {}
                    perturbed_representations = {}
                    
                    if 'vision' in self.config['modalities'] and 'vision' in batch['perturbed']:
                        perturbed_vision = batch['perturbed']['vision']
                        perturbed_vision_features = self.vision_encoder(perturbed_vision).pooler_output
                        if len(perturbed_vision_features.shape) == 3:
                            perturbed_vision_features = perturbed_vision_features.mean(dim=1)
                        perturbed_vision_specific = self.vision_projection(perturbed_vision_features)
                        perturbed_representations['vision_specific'] = perturbed_vision_specific
                    
                    if 'text' in self.config['modalities'] and 'text' in batch['perturbed']:
                        perturbed_text = batch['perturbed']['text']
                        perturbed_text_features = self.text_encoder(**perturbed_text).pooler_output
                        perturbed_text_specific = self.text_projection(perturbed_text_features)
                        perturbed_representations['text_specific'] = perturbed_text_specific
                    
                    perturbed_modality_features = []
                    if 'vision' in self.config['modalities'] and 'vision' in batch['perturbed']:
                        perturbed_modality_features.append(perturbed_vision_features)
                    elif 'vision' in self.config['modalities']:
                        perturbed_modality_features.append(vision_features)
                    
                    if 'text' in self.config['modalities'] and 'text' in batch['perturbed']:
                        perturbed_modality_features.append(perturbed_text_features)
                    elif 'text' in self.config['modalities']:
                        perturbed_modality_features.append(text_features)
                    
                    perturbed_shared_features = self.shared_encoder(perturbed_modality_features)
                    perturbed_representations['shared'] = perturbed_shared_features
                
                # Compute contrastive loss between original and perturbed shared representations
                z_i = F.normalize(shared_features, dim=1)
                z_j = F.normalize(perturbed_representations['shared'], dim=1)
                
                # Cosine similarity matrix
                sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
                
                # Labels for contrastive learning (identity matrix)
                labels_contrastive = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                
                # Contrastive loss (InfoNCE)
                contrastive_loss = F.cross_entropy(sim_matrix, labels_contrastive)
                losses['contrastive_loss'] = contrastive_loss
            else:
                losses['contrastive_loss'] = torch.tensor(0.0, device=combined_pred.device)
            
            # Modality disentanglement loss (orthogonality constraint)
            disentanglement_loss = 0
            
            if 'vision' in self.config['modalities'] and 'text' in self.config['modalities']:
                # Orthogonality between modality-specific representations
                vision_norm = F.normalize(vision_specific, dim=1)
                text_norm = F.normalize(text_specific, dim=1)
                modal_ortho = torch.matmul(vision_norm, text_norm.T).pow(2).mean()
                disentanglement_loss += modal_ortho
            
            if 'vision' in self.config['modalities']:
                # Orthogonality between vision-specific and shared representations
                vision_norm = F.normalize(vision_specific, dim=1)
                shared_norm = F.normalize(shared_features, dim=1)
                vision_shared_ortho = torch.matmul(vision_norm, shared_norm.T).pow(2).mean()
                disentanglement_loss += vision_shared_ortho
            
            if 'text' in self.config['modalities']:
                # Orthogonality between text-specific and shared representations
                text_norm = F.normalize(text_specific, dim=1)
                shared_norm = F.normalize(shared_features, dim=1)
                text_shared_ortho = torch.matmul(text_norm, shared_norm.T).pow(2).mean()
                disentanglement_loss += text_shared_ortho
            
            losses['disentanglement_loss'] = disentanglement_loss * self.ortho_weight
            
            # Intervention-based fine-tuning loss (KL divergence for counterfactual consistency)
            intervention_loss = 0
            
            if 'counterfactual' in batch:
                # Get predictions for counterfactual samples
                cf_outputs = self.forward(batch['counterfactual'], compute_loss=False)
                cf_pred = cf_outputs['pred']
                
                # KL divergence between original and counterfactual predictions
                log_prob = F.log_softmax(combined_pred, dim=1)
                cf_prob = F.softmax(cf_pred, dim=1)
                intervention_loss = F.kl_div(log_prob, cf_prob, reduction='batchmean')
                
                losses['intervention_loss'] = intervention_loss
            else:
                losses['intervention_loss'] = torch.tensor(0.0, device=combined_pred.device)
            
            # Total loss
            losses['total_loss'] = (
                losses['ce_loss'] + 
                self.contrastive_weight * losses['contrastive_loss'] + 
                self.modality_disentanglement_weight * losses['disentanglement_loss'] + 
                self.intervention_weight * losses['intervention_loss']
            )
            
            outputs['losses'] = losses
        
        outputs['representations'] = representations
        return outputs

    def get_grad_cam(self, x, target_layer, class_idx=None):
        """
        Generate Grad-CAM visualizations to understand which features the model focuses on.
        
        Args:
            x: Input data
            target_layer: Layer to compute gradients for
            class_idx: Target class index to compute gradients for
            
        Returns:
            heatmap: Grad-CAM heatmap
        """
        # Implement Grad-CAM visualization
        # This is a placeholder for actual implementation
        pass
    
    def compute_counterfactual(self, x, intervention_type, intervention_strength=0.1):
        """
        Compute counterfactual samples by intervening on features.
        
        Args:
            x: Input data
            intervention_type: Type of intervention ('spurious' or 'causal')
            intervention_strength: Strength of the intervention
            
        Returns:
            counterfactual_x: Counterfactual sample
        """
        # Implement counterfactual computation
        # This is a placeholder for actual implementation
        pass


class CrossModalEncoder(nn.Module):
    """
    Cross-modal encoder for learning shared representations across modalities.
    """
    
    def __init__(self, modality_dims, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        """
        Initialize the cross-modal encoder.
        
        Args:
            modality_dims: List of dimensions for each modality
            output_dim: Dimension of the output shared representation
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossModalEncoder, self).__init__()
        
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, output_dim)
            for dim in modality_dims
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, modality_features):
        """
        Forward pass through the cross-modal encoder.
        
        Args:
            modality_features: List of features for each modality
            
        Returns:
            shared_features: Shared representation across modalities
        """
        # Project each modality to a common dimension
        projected_features = [
            proj(feat) for proj, feat in zip(self.modality_projections, modality_features)
        ]
        
        # Combine modality features (stack along sequence dimension)
        combined_features = torch.stack(projected_features, dim=1)
        
        # Apply transformer encoder
        encoded_features = self.transformer_encoder(combined_features)
        
        # Mean pooling across modalities
        pooled_features = encoded_features.mean(dim=1)
        
        # Final projection
        shared_features = self.output_projection(pooled_features)
        
        return shared_features