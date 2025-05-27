"""
Model definitions for the LASS framework and baseline methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertConfig
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger("LASS.models")

class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self):
        super().__init__()
        
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings (features) from the model.
        
        Args:
            x: Input data.
            
        Returns:
            embeddings: Feature embeddings.
        """
        raise NotImplementedError
    
    def save_embeddings(self, dataloader: torch.utils.data.DataLoader, 
                       device: torch.device, save_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Save embeddings for all samples in a dataloader.
        
        Args:
            dataloader: DataLoader containing samples.
            device: Device to run model on.
            save_path: Path to save embeddings.
            
        Returns:
            embeddings: Feature embeddings for all samples.
            labels: Labels for all samples.
            group_labels: Group labels for all samples.
        """
        self.eval()
        all_embeddings = []
        all_labels = []
        all_groups = []
        
        with torch.no_grad():
            for inputs, labels, groups in dataloader:
                inputs = inputs.to(device)
                embeddings = self.get_embeddings(inputs)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                all_groups.append(groups)
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_groups = torch.cat(all_groups, dim=0)
        
        # Save to disk
        torch.save({
            'embeddings': all_embeddings,
            'labels': all_labels,
            'groups': all_groups
        }, save_path)
        
        return all_embeddings, all_labels, all_groups

class ImageClassifier(BaseModel):
    """
    Image classifier based on pre-trained CNN backbones.
    """
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50', 
                pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the image classifier.
        
        Args:
            num_classes: Number of output classes.
            backbone: Backbone CNN architecture.
            pretrained: Whether to use pre-trained weights.
            freeze_backbone: Whether to freeze the backbone parameters.
        """
        super().__init__()
        
        # Create backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            logits: Output logits.
        """
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings.
        
        Args:
            x: Input tensor.
            
        Returns:
            embeddings: Feature embeddings.
        """
        return self.backbone(x)

class TextClassifier(BaseModel):
    """
    Text classifier based on pre-trained BERT.
    """
    
    def __init__(self, num_classes: int, bert_model: str = 'bert-base-uncased',
                pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize the text classifier.
        
        Args:
            num_classes: Number of output classes.
            bert_model: BERT model name or path.
            pretrained: Whether to use pre-trained weights.
            freeze_backbone: Whether to freeze the backbone parameters.
        """
        super().__init__()
        
        # Create BERT model
        if pretrained:
            self.bert = BertModel.from_pretrained(bert_model)
        else:
            config = BertConfig.from_pretrained(bert_model)
            self.bert = BertModel(config)
        
        self.feature_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
              token_type_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            
        Returns:
            logits: Output logits.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use CLS token representation
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits
    
    def get_embeddings(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get feature embeddings.
        
        Args:
            input_dict: Dictionary of input tensors.
            
        Returns:
            embeddings: Feature embeddings.
        """
        outputs = self.bert(
            input_ids=input_dict['input_ids'],
            attention_mask=input_dict.get('attention_mask'),
            token_type_ids=input_dict.get('token_type_ids')
        )
        return outputs.pooler_output

class GroupDRO(BaseModel):
    """
    Group Distributionally Robust Optimization (Group-DRO) model.
    
    Based on the paper:
    "Distributionally Robust Neural Networks for Group Shifts: On the Importance of
    Regularization for Worst-Case Generalization" (Sagawa et al., 2019)
    """
    
    def __init__(self, base_model: BaseModel, num_groups: int,
                eta: float = 1.0, alpha: float = 0.1):
        """
        Initialize Group-DRO model.
        
        Args:
            base_model: Base model to wrap.
            num_groups: Number of groups in the dataset.
            eta: Step size for group weights update.
            alpha: Regularization strength.
        """
        super().__init__()
        self.base_model = base_model
        self.num_groups = num_groups
        self.eta = eta
        self.alpha = alpha
        
        # Initialize group weights
        self.register_buffer('group_weights', torch.ones(num_groups) / num_groups)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            logits: Output logits.
        """
        return self.base_model(x)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings.
        
        Args:
            x: Input tensor.
            
        Returns:
            embeddings: Feature embeddings.
        """
        return self.base_model.get_embeddings(x)
    
    def update_group_weights(self, group_losses: torch.Tensor) -> None:
        """
        Update group weights based on losses.
        
        Args:
            group_losses: Per-group losses.
        """
        # Update group weights
        self.group_weights = self.group_weights * torch.exp(self.eta * group_losses)
        self.group_weights = self.group_weights / self.group_weights.sum()
    
    def compute_group_dro_loss(self, outputs: torch.Tensor, labels: torch.Tensor,
                             groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Group-DRO loss.
        
        Args:
            outputs: Model outputs.
            labels: True labels.
            groups: Group labels.
            
        Returns:
            loss: Group-DRO loss.
            group_losses: Per-group losses.
        """
        per_sample_loss = F.cross_entropy(outputs, labels, reduction='none')
        
        # Compute per-group losses
        group_losses = torch.zeros(self.num_groups, device=outputs.device)
        group_counts = torch.zeros(self.num_groups, device=outputs.device)
        
        # Aggregate losses by group
        for g in range(self.num_groups):
            g_mask = (groups == g)
            if g_mask.sum() > 0:
                group_losses[g] = per_sample_loss[g_mask].mean()
                group_counts[g] = g_mask.sum()
            else:
                group_losses[g] = 0
                
        # Group-DRO loss
        loss = (self.group_weights * group_losses).sum()
        
        # Add weight regularization
        if hasattr(self.base_model, 'fc'):
            loss += self.alpha * torch.norm(self.base_model.fc.weight)**2
        
        # Update group weights
        self.update_group_weights(group_losses.detach())
        
        return loss, group_losses

class ULEModel(BaseModel):
    """
    UnLearning from Experience (ULE) model.
    
    Based on the paper:
    "UnLearning from Experience to Avoid Spurious Correlations" (Mitchell et al., 2024)
    """
    
    def __init__(self, num_classes: int, backbone: str = 'resnet50',
                lambda_unlearn: float = 0.1):
        """
        Initialize ULE model with student and teacher networks.
        
        Args:
            num_classes: Number of output classes.
            backbone: Backbone CNN architecture.
            lambda_unlearn: Weight for unlearning loss.
        """
        super().__init__()
        
        # Student model (learns both causal and spurious features)
        self.student = ImageClassifier(num_classes, backbone)
        
        # Teacher model (learns only causal features by unlearning what student learns)
        self.teacher = ImageClassifier(num_classes, backbone)
        
        self.lambda_unlearn = lambda_unlearn
        self.feature_dim = self.student.feature_dim
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both student and teacher.
        
        Args:
            x: Input tensor.
            
        Returns:
            outputs: Dictionary with student and teacher outputs.
        """
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)
        
        return {
            'student': student_logits,
            'teacher': teacher_logits
        }
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get teacher embeddings.
        
        Args:
            x: Input tensor.
            
        Returns:
            embeddings: Teacher embeddings.
        """
        return self.teacher.get_embeddings(x)
    
    def compute_ule_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
                       x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ULE loss components.
        
        Args:
            outputs: Model outputs dictionary.
            labels: True labels.
            x: Input tensor.
            
        Returns:
            losses: Dictionary of loss components.
        """
        student_logits = outputs['student']
        teacher_logits = outputs['teacher']
        
        # Standard cross-entropy losses
        student_loss = F.cross_entropy(student_logits, labels)
        teacher_loss = F.cross_entropy(teacher_logits, labels)
        
        # Unlearning loss
        with torch.enable_grad():
            # Compute gradient of student output with respect to input
            x_student = x.clone().detach().requires_grad_(True)
            student_out = self.student(x_student)
            student_pred = student_out.argmax(dim=1)
            
            # Gradient of student predictions w.r.t input
            student_grad = torch.autograd.grad(
                outputs=student_out.gather(1, student_pred.unsqueeze(1)).sum(),
                inputs=x_student,
                create_graph=True
            )[0]
            
            # Teacher should avoid features student relies on
            x_teacher = x.clone().detach()
            teacher_out = self.teacher(x_teacher)
            
            # Encourage teacher to use different features
            similarity = (teacher_out * student_out).sum(dim=1).mean()
            
            unlearn_loss = self.lambda_unlearn * similarity
        
        return {
            'student_loss': student_loss,
            'teacher_loss': teacher_loss,
            'unlearn_loss': unlearn_loss,
            'total_loss': student_loss + teacher_loss + unlearn_loss
        }

class LLMAugmentedModel(BaseModel):
    """
    Model that incorporates LLM-generated hypotheses for robustification.
    This is the core model for the LASS framework.
    """
    
    def __init__(self, base_model: BaseModel, num_classes: int, num_groups: int,
                aux_head: bool = True, reweighting: bool = True,
                lambda_aux: float = 0.1):
        """
        Initialize LLM-augmented model.
        
        Args:
            base_model: Base model to augment.
            num_classes: Number of output classes.
            num_groups: Number of groups in the dataset.
            aux_head: Whether to use auxiliary head for spurious feature prediction.
            reweighting: Whether to use sample reweighting.
            lambda_aux: Weight for auxiliary loss.
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.use_aux_head = aux_head
        self.use_reweighting = reweighting
        self.lambda_aux = lambda_aux
        
        # Extract feature dimension from base model
        if hasattr(base_model, 'feature_dim'):
            self.feature_dim = base_model.feature_dim
        elif hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'feature_dim'):
            self.feature_dim = base_model.base_model.feature_dim
        else:
            logger.warning("Could not determine feature dimension, defaulting to 2048")
            self.feature_dim = 2048
        
        # Create auxiliary head for spurious feature prediction
        if aux_head:
            self.aux_head = nn.Linear(self.feature_dim, 2)  # Binary spurious feature
        
        # Initialize sample weights
        if reweighting:
            self.register_buffer('sample_weights', torch.ones(num_groups))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            outputs: Dictionary with model outputs.
        """
        features = self.base_model.get_embeddings(x)
        logits = self.base_model(x)
        
        outputs = {'logits': logits}
        
        if self.use_aux_head:
            aux_logits = self.aux_head(features)
            outputs['aux_logits'] = aux_logits
        
        return outputs
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings.
        
        Args:
            x: Input tensor.
            
        Returns:
            embeddings: Feature embeddings.
        """
        return self.base_model.get_embeddings(x)
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor,
                   groups: torch.Tensor, spurious_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute model loss components.
        
        Args:
            outputs: Model outputs dictionary.
            labels: True class labels.
            groups: Group labels.
            spurious_labels: Labels for spurious features (if available).
            
        Returns:
            losses: Dictionary of loss components.
        """
        main_loss = F.cross_entropy(outputs['logits'], labels, reduction='none')
        
        # Apply sample reweighting if enabled
        if self.use_reweighting:
            # Map group indices to weights
            weights = torch.ones_like(main_loss)
            for g in range(self.num_groups):
                g_mask = (groups == g)
                weights[g_mask] = self.sample_weights[g]
            
            # Normalize weights
            weights = weights / weights.sum() * len(weights)
            
            # Apply weights to loss
            main_loss = (main_loss * weights).mean()
        else:
            main_loss = main_loss.mean()
        
        losses = {'main_loss': main_loss}
        
        # Add auxiliary loss if enabled and spurious labels are provided
        if self.use_aux_head and spurious_labels is not None:
            aux_loss = F.cross_entropy(outputs['aux_logits'], spurious_labels)
            losses['aux_loss'] = aux_loss
            losses['total_loss'] = main_loss + self.lambda_aux * aux_loss
        else:
            losses['total_loss'] = main_loss
        
        return losses
    
    def update_sample_weights(self, group_losses: torch.Tensor, eta: float = 0.1) -> None:
        """
        Update sample weights based on group losses.
        
        Args:
            group_losses: Per-group losses.
            eta: Step size for weight updates.
        """
        if self.use_reweighting:
            # Update weights using exponential update rule
            self.sample_weights = self.sample_weights * torch.exp(eta * group_losses)
            self.sample_weights = self.sample_weights / self.sample_weights.sum() * self.num_groups