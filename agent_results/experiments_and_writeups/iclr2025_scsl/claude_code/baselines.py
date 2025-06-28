"""
Baseline methods for comparison with AIFS.

This module implements baseline methods for addressing spurious correlations,
including standard ERM and group-based approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class StandardModel(nn.Module):
    """
    Standard model for classification without any specific mechanisms for
    handling spurious correlations. Uses standard Empirical Risk Minimization (ERM).
    """
    
    def __init__(
        self, 
        num_classes: int, 
        feature_dim: int = 512,
        pretrained: bool = True,
        model_type: str = "resnet18",
        input_dim: Optional[int] = None
    ):
        """
        Initialize the standard model.
        
        Args:
            num_classes: Number of output classes
            feature_dim: Dimension of features before classification layer
            pretrained: Whether to use pretrained weights for encoder
            model_type: Model architecture to use ('resnet18', 'resnet50', 'mlp')
            input_dim: Input dimension for MLP, required if model_type='mlp'
        """
        super().__init__()
        
        self.model_type = model_type
        
        # Initialize feature extractor
        if model_type == "resnet18":
            self.features = models.resnet18(pretrained=pretrained)
            # Replace the fully connected layer
            self.features.fc = nn.Linear(self.features.fc.in_features, feature_dim)
            self.classifier = nn.Linear(feature_dim, num_classes)
        
        elif model_type == "resnet50":
            self.features = models.resnet50(pretrained=pretrained)
            self.features.fc = nn.Linear(self.features.fc.in_features, feature_dim)
            self.classifier = nn.Linear(feature_dim, num_classes)
        
        elif model_type == "mlp":
            if input_dim is None:
                raise ValueError("input_dim must be specified for MLP model")
            
            self.features = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim),
                nn.ReLU()
            )
            self.classifier = nn.Linear(feature_dim, num_classes)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.features(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.features(x)


class GroupDRO(nn.Module):
    """
    Group Distributionally Robust Optimization (Group DRO) model.
    
    This model minimizes the worst-case risk over groups, which helps
    with robustness to group shift and spurious correlations when
    group annotations are available.
    
    Based on: "Distributionally Robust Neural Networks for Group Shifts: 
              On the Importance of Regularization for Worst-Case Generalization"
              Sagawa et al. 2020
    """
    
    def __init__(
        self, 
        num_classes: int, 
        feature_dim: int = 512,
        pretrained: bool = True,
        model_type: str = "resnet18",
        num_groups: int = 2,
        input_dim: Optional[int] = None
    ):
        """
        Initialize the Group DRO model.
        
        Args:
            num_classes: Number of output classes
            feature_dim: Dimension of features before classification layer
            pretrained: Whether to use pretrained weights for encoder
            model_type: Model architecture to use ('resnet18', 'resnet50', 'mlp')
            num_groups: Number of groups in the data
            input_dim: Input dimension for MLP, required if model_type='mlp'
        """
        super().__init__()
        
        # Use the same backbone as StandardModel
        self.base_model = StandardModel(
            num_classes=num_classes,
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_type=model_type,
            input_dim=input_dim
        )
        
        # Group DRO hyperparameters
        self.num_groups = num_groups
        self.group_weights = nn.Parameter(
            torch.ones(num_groups) / num_groups,
            requires_grad=False
        )
        self.group_adj = torch.ones(num_groups).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.base_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.base_model.get_features(x)
    
    def compute_group_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        groups: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss for each group.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            groups: Group assignments
            
        Returns:
            Tuple of (total loss, group losses)
        """
        # Compute per-sample loss
        per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Compute loss for each group
        group_losses = torch.zeros(self.num_groups).to(device)
        group_counts = torch.zeros(self.num_groups).to(device)
        
        for i in range(self.num_groups):
            group_mask = (groups == i)
            if group_mask.sum() > 0:
                group_losses[i] = per_sample_loss[group_mask].mean()
                group_counts[i] = group_mask.sum()
            else:
                group_losses[i] = 0
                group_counts[i] = 0
        
        # Adjust group weights
        if group_counts.sum() > 0:
            self.group_adj = (group_counts > 0).float()
            
            # Use group weights for weighted loss
            total_loss = (self.group_weights * group_losses).sum()
        else:
            # Fall back to standard loss if no groups have samples
            total_loss = per_sample_loss.mean()
        
        return total_loss, group_losses
    
    def update_group_weights(
        self, 
        group_losses: torch.Tensor, 
        step_size: float = 0.01
    ):
        """
        Update group weights based on losses.
        
        Args:
            group_losses: Loss for each group
            step_size: Step size for weight updates
        """
        # Skip if no groups have samples
        if self.group_adj.sum() == 0:
            return
        
        # Adjust group weights based on loss
        adjusted_losses = group_losses * self.group_adj
        
        # Multiplicative update
        self.group_weights.data = self.group_weights * torch.exp(step_size * adjusted_losses)
        
        # Normalize weights
        self.group_weights.data = self.group_weights / self.group_weights.sum()


class DomainAdversarialModel(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) model.
    
    This model aims to learn domain-invariant features by using a domain 
    discriminator and adversarial training, which can help with robustness 
    to domain shift and spurious correlations.
    
    Based on: "Domain-Adversarial Training of Neural Networks"
              Ganin et al. 2016
    """
    
    def __init__(
        self, 
        num_classes: int, 
        feature_dim: int = 512,
        pretrained: bool = True,
        model_type: str = "resnet18",
        input_dim: Optional[int] = None,
        num_domains: int = 2  # Number of domains (typically 2 for spurious correlation)
    ):
        """
        Initialize the Domain Adversarial model.
        
        Args:
            num_classes: Number of output classes
            feature_dim: Dimension of features before classification layer
            pretrained: Whether to use pretrained weights for encoder
            model_type: Model architecture to use ('resnet18', 'resnet50', 'mlp')
            input_dim: Input dimension for MLP, required if model_type='mlp'
            num_domains: Number of domains in the data
        """
        super().__init__()
        
        # Use the same backbone as StandardModel for feature extraction
        self.feature_extractor = StandardModel(
            num_classes=feature_dim,  # Use feature_dim as output
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_type=model_type,
            input_dim=input_dim
        )
        
        # Remove the classifier from the feature extractor
        if hasattr(self.feature_extractor, 'classifier'):
            del self.feature_extractor.classifier
        
        # Label predictor (classifier)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Domain discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_domains)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        alpha: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            alpha: Gradient reversal scaling parameter
            
        Returns:
            Tuple of (class logits, domain logits)
        """
        # Extract features
        features = self.feature_extractor.get_features(x)
        
        # Task classifier
        class_logits = self.classifier(features)
        
        # Domain discriminator with gradient reversal
        if self.training:
            # Apply gradient reversal using a custom autograd function
            reversed_features = GradientReversalFunction.apply(features, alpha)
            domain_logits = self.domain_classifier(reversed_features)
        else:
            domain_logits = self.domain_classifier(features)
        
        return class_logits, domain_logits


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training.
    
    Forward pass is identity, but backward pass reverses gradients.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.copy()
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ReweightingModel(nn.Module):
    """
    Reweighting model for addressing class imbalance and group disparities.
    
    This model applies sample reweighting based on class and group membership,
    which helps with robustness to group imbalance and spurious correlations.
    """
    
    def __init__(
        self, 
        num_classes: int, 
        feature_dim: int = 512,
        pretrained: bool = True,
        model_type: str = "resnet18",
        input_dim: Optional[int] = None,
        num_groups: int = 2
    ):
        """
        Initialize the Reweighting model.
        
        Args:
            num_classes: Number of output classes
            feature_dim: Dimension of features before classification layer
            pretrained: Whether to use pretrained weights for encoder
            model_type: Model architecture to use ('resnet18', 'resnet50', 'mlp')
            input_dim: Input dimension for MLP, required if model_type='mlp'
            num_groups: Number of groups in the data
        """
        super().__init__()
        
        # Use the same backbone as StandardModel
        self.base_model = StandardModel(
            num_classes=num_classes,
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_type=model_type,
            input_dim=input_dim
        )
        
        # Reweighting parameters
        self.num_classes = num_classes
        self.num_groups = num_groups
        
        # Initialize weight matrix for class-group combinations
        self.weights = torch.ones(num_classes, num_groups).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.base_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.base_model.get_features(x)
    
    def update_weights(self, labels: torch.Tensor, groups: torch.Tensor):
        """
        Update sample weights based on performance.
        
        Args:
            labels: Ground truth labels
            groups: Group assignments
        """
        # Count samples in each class-group combination
        class_group_counts = torch.zeros(self.num_classes, self.num_groups).to(device)
        
        for i in range(len(labels)):
            class_idx = labels[i].item()
            group_idx = groups[i].item()
            class_group_counts[class_idx, group_idx] += 1
        
        # Compute inverse frequency for reweighting
        # Add a small constant to avoid division by zero
        class_group_counts = class_group_counts + 1e-6
        self.weights = torch.mean(class_group_counts) / class_group_counts
    
    def compute_weighted_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        groups: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            groups: Group assignments
            
        Returns:
            Weighted loss
        """
        # Compute per-sample loss
        per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Get weights for each sample
        sample_weights = torch.zeros(len(labels)).to(device)
        
        for i in range(len(labels)):
            class_idx = labels[i].item()
            group_idx = groups[i].item()
            sample_weights[i] = self.weights[class_idx, group_idx]
        
        # Normalize weights
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        # Apply weights
        weighted_loss = (per_sample_loss * sample_weights).mean()
        
        return weighted_loss


# Training functions for the baseline models

def train_standard_model(
    model: StandardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 5,
    save_path: str = "standard_model.pt"
) -> Dict[str, List[float]]:
    """
    Train a standard model using Empirical Risk Minimization (ERM).
    
    Args:
        model: Standard model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay factor
        patience: Early stopping patience
        save_path: Path to save the best model
        
    Returns:
        Dictionary of training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # For early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # For tracking metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            # Handle data with or without group labels
            if len(batch) == 3:
                inputs, labels, _ = batch  # Ignore group labels
            else:
                inputs, labels = batch
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle data with or without group labels
                if len(batch) == 3:
                    inputs, labels, _ = batch  # Ignore group labels
                else:
                    inputs, labels = batch
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_accuracy)
        metrics["val_acc"].append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return metrics


def train_group_dro(
    model: GroupDRO,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 5,
    step_size: float = 0.01,
    save_path: str = "group_dro_model.pt"
) -> Dict[str, List[float]]:
    """
    Train a Group DRO model to minimize worst-case risk over groups.
    
    Args:
        model: Group DRO model to train
        train_loader: DataLoader for training data with group labels
        val_loader: DataLoader for validation data with group labels
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay factor
        patience: Early stopping patience
        step_size: Step size for group weight updates
        save_path: Path to save the best model
        
    Returns:
        Dictionary of training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # For early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # For tracking metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "group_losses": [],
        "group_weights": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_group_losses = []
        
        for batch in train_loader:
            # Group DRO requires group labels
            if len(batch) != 3:
                raise ValueError("Group DRO requires data with group labels")
            
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Compute group losses
            loss, group_losses = model.compute_group_loss(logits, labels, groups)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update group weights
            model.update_group_weights(group_losses, step_size=step_size)
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            epoch_group_losses.append(group_losses.cpu().detach().numpy())
        
        # Calculate training metrics
        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Average group losses for this epoch
        avg_group_losses = np.mean(np.array(epoch_group_losses), axis=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) != 3:
                    raise ValueError("Group DRO requires data with group labels")
                
                inputs, labels, groups = batch
                inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
                
                # Forward pass
                logits = model(inputs)
                
                # Compute group losses
                loss, _ = model.compute_group_loss(logits, labels, groups)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_accuracy)
        metrics["val_acc"].append(val_accuracy)
        metrics["group_losses"].append(avg_group_losses)
        metrics["group_weights"].append(model.group_weights.cpu().detach().numpy())
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Group Weights: {model.group_weights.cpu().numpy()}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return metrics


def train_domain_adversarial(
    model: DomainAdversarialModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 5,
    lambda_adversarial: float = 0.1,
    save_path: str = "domain_adversarial_model.pt"
) -> Dict[str, List[float]]:
    """
    Train a Domain Adversarial Neural Network (DANN) model.
    
    Args:
        model: DANN model to train
        train_loader: DataLoader for training data with domain labels
        val_loader: DataLoader for validation data with domain labels
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay factor
        patience: Early stopping patience
        lambda_adversarial: Weight for the adversarial loss
        save_path: Path to save the best model
        
    Returns:
        Dictionary of training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # For early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # For tracking metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "domain_acc": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        domain_correct = 0
        domain_total = 0
        
        # Gradually increase adversarial weight (like in DANN paper)
        p = min(1.0, float(epoch) / (num_epochs * 0.75))
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        for batch in train_loader:
            # DANN requires domain labels (using group labels as domains)
            if len(batch) != 3:
                raise ValueError("DANN requires data with group labels as domains")
            
            inputs, labels, domains = batch
            inputs, labels, domains = inputs.to(device), labels.to(device), domains.to(device)
            
            # Forward pass
            class_logits, domain_logits = model(inputs, alpha=alpha)
            
            # Compute class loss
            class_loss = criterion(class_logits, labels)
            
            # Compute domain loss
            domain_loss = criterion(domain_logits, domains)
            
            # Total loss (weighted sum)
            loss = class_loss + lambda_adversarial * domain_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            
            # Class accuracy
            _, predicted_class = torch.max(class_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted_class == labels).sum().item()
            
            # Domain accuracy
            _, predicted_domain = torch.max(domain_logits, 1)
            domain_total += domains.size(0)
            domain_correct += (predicted_domain == domains).sum().item()
        
        # Calculate training metrics
        train_accuracy = 100 * train_correct / train_total
        domain_accuracy = 100 * domain_correct / domain_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, domains = batch
                inputs, labels, domains = inputs.to(device), labels.to(device), domains.to(device)
                
                # Forward pass
                class_logits, domain_logits = model(inputs, alpha=0.0)  # No domain adaptation during val
                
                # Compute class loss (only for val metrics)
                loss = criterion(class_logits, labels)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = torch.max(class_logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_accuracy)
        metrics["val_acc"].append(val_accuracy)
        metrics["domain_acc"].append(domain_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
              f"Domain Acc: {domain_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return metrics


def train_reweighting_model(
    model: ReweightingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 5,
    update_weights_every: int = 5,
    save_path: str = "reweighting_model.pt"
) -> Dict[str, List[float]]:
    """
    Train a Reweighting model for imbalanced classification.
    
    Args:
        model: Reweighting model to train
        train_loader: DataLoader for training data with group labels
        val_loader: DataLoader for validation data with group labels
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay factor
        patience: Early stopping patience
        update_weights_every: Update sample weights every N epochs
        save_path: Path to save the best model
        
    Returns:
        Dictionary of training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # For early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # For tracking metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    # Initialize weights based on class and group frequencies
    for batch in train_loader:
        if len(batch) != 3:
            raise ValueError("Reweighting model requires data with group labels")
        
        inputs, labels, groups = batch
        model.update_weights(labels, groups)
        break
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Update weights every N epochs
        if epoch % update_weights_every == 0:
            print("Updating sample weights...")
            all_labels = []
            all_groups = []
            
            # Collect labels and groups from training data
            for batch in train_loader:
                inputs, labels, groups = batch
                all_labels.extend(labels)
                all_groups.extend(groups)
            
            # Convert to tensors
            all_labels = torch.tensor(all_labels).to(device)
            all_groups = torch.tensor(all_groups).to(device)
            
            # Update weights
            model.update_weights(all_labels, all_groups)
        
        # Training loop
        for batch in train_loader:
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Compute weighted loss
            loss = model.compute_weighted_loss(logits, labels, groups)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, groups = batch
                inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
                
                # Forward pass
                logits = model(inputs)
                
                # Compute loss (regular CE for validation)
                loss = F.cross_entropy(logits, labels)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_accuracy)
        metrics["val_acc"].append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return metrics


# Example usage:
# model = StandardModel(num_classes=10)
# metrics = train_standard_model(model, train_loader, val_loader)