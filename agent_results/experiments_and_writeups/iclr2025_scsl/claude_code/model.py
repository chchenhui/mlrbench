"""
AIFS: Adaptive Invariant Feature Extraction using Synthetic Interventions

This module implements the AIFS method as described in the proposal.
The method integrates a generative intervention loop into model training to
automatically discover and neutralize hidden spurious factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class InterventionModule(nn.Module):
    """
    A lightweight intervention module that applies randomized "style" perturbations 
    in selected latent subspaces.
    """
    def __init__(self, latent_dim: int, num_masks: int = 5, mask_ratio: float = 0.2):
        """
        Initialize the intervention module.
        
        Args:
            latent_dim: Dimension of the latent space
            num_masks: Number of different intervention masks to learn
            mask_ratio: Proportion of dimensions to include in each mask
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_masks = num_masks
        self.mask_ratio = mask_ratio
        
        # Initialize learnable masks for interventions
        # Each mask will focus on specific latent dimensions
        self.intervention_masks = nn.Parameter(
            torch.zeros(num_masks, latent_dim)
        )
        self._initialize_masks()
        
        # Scaling factor for interventions
        self.scale_factor = nn.Parameter(torch.tensor(1.0))
    
    def _initialize_masks(self):
        """Initialize intervention masks with some dimensions activated."""
        for i in range(self.num_masks):
            # Randomly select dimensions to include in the mask
            num_active = int(self.latent_dim * self.mask_ratio)
            active_indices = np.random.choice(
                self.latent_dim, size=num_active, replace=False
            )
            self.intervention_masks.data[i, active_indices] = 1.0
    
    def apply_intervention(self, latent: torch.Tensor, mask_idx: Optional[int] = None) -> torch.Tensor:
        """
        Apply intervention to the latent representation.
        
        Args:
            latent: Latent representation [batch_size, latent_dim]
            mask_idx: Index of the mask to use. If None, a random mask is selected.
            
        Returns:
            Perturbed latent representation
        """
        batch_size = latent.size(0)
        
        # Select mask (either specified or random)
        if mask_idx is None:
            mask_idx = torch.randint(0, self.num_masks, (1,)).item()
        
        mask = self.intervention_masks[mask_idx]
        
        # Generate random noise for intervention
        noise = torch.randn_like(latent) * self.scale_factor
        
        # Apply mask to noise (element-wise multiplication)
        masked_noise = noise * mask.view(1, -1)
        
        # Apply intervention
        perturbed_latent = latent + masked_noise
        
        return perturbed_latent, mask_idx
    
    def update_mask_importance(self, gradients: torch.Tensor, mask_idx: int, learning_rate: float = 0.01):
        """
        Update the importance of mask dimensions based on gradients.
        
        Args:
            gradients: Gradients of loss with respect to latent [batch_size, latent_dim]
            mask_idx: Index of the mask that was used
            learning_rate: Learning rate for the update
        """
        # Average gradients across batch
        avg_grad = gradients.abs().mean(dim=0)
        
        # Update mask values based on gradient magnitudes
        with torch.no_grad():
            # Increase values for dimensions with high gradients
            self.intervention_masks.data[mask_idx] += learning_rate * avg_grad
            
            # Normalize mask to maintain the same proportion of active dimensions
            sorted_mask, _ = torch.sort(self.intervention_masks.data[mask_idx], descending=True)
            threshold = sorted_mask[int(self.latent_dim * self.mask_ratio)]
            self.intervention_masks.data[mask_idx] = (
                self.intervention_masks.data[mask_idx] > threshold
            ).float()


class AIFS(nn.Module):
    """
    Adaptive Invariant Feature Extraction using Synthetic Interventions.
    
    This model integrates a generative intervention loop into training to
    automatically discover and neutralize hidden spurious factors.
    """
    
    def __init__(
        self, 
        num_classes: int, 
        latent_dim: int = 512,
        pretrained: bool = True,
        encoder_name: str = "resnet18",
        num_masks: int = 5,
        mask_ratio: float = 0.2
    ):
        """
        Initialize the AIFS model.
        
        Args:
            num_classes: Number of output classes
            latent_dim: Dimension of the latent space
            pretrained: Whether to use pretrained encoder
            encoder_name: Name of the encoder model
            num_masks: Number of different intervention masks
            mask_ratio: Proportion of dimensions to include in each mask
        """
        super().__init__()
        
        # Initialize encoder (feature extractor)
        self.encoder = self._get_encoder(encoder_name, pretrained, latent_dim)
        
        # Intervention module
        self.intervention = InterventionModule(
            latent_dim=latent_dim,
            num_masks=num_masks,
            mask_ratio=mask_ratio
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def _get_encoder(self, name: str, pretrained: bool, latent_dim: int) -> nn.Module:
        """
        Get the encoder model.
        
        Args:
            name: Name of the encoder model
            pretrained: Whether to use pretrained weights
            latent_dim: Dimension of the latent space
            
        Returns:
            Encoder model
        """
        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            # Replace the final fully connected layer
            model.fc = nn.Linear(model.fc.in_features, latent_dim)
        elif name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, latent_dim)
        else:
            raise ValueError(f"Unsupported encoder: {name}")
        
        return model
    
    def forward(
        self, 
        x: torch.Tensor, 
        apply_intervention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            apply_intervention: Whether to apply intervention
            
        Returns:
            Tuple of (logits, perturbed_latent, mask_idx) where the latter two are None
            if apply_intervention is False
        """
        # Extract features
        latent = self.encoder(x)
        
        # Apply intervention if requested
        perturbed_latent = None
        mask_idx = None
        
        if apply_intervention:
            perturbed_latent, mask_idx = self.intervention.apply_intervention(latent)
            # Predict using perturbed latent
            logits = self.classifier(perturbed_latent)
        else:
            # Standard prediction
            logits = self.classifier(latent)
        
        return logits, perturbed_latent, mask_idx
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for a given input."""
        return self.encoder(x)


class AIFSLoss(nn.Module):
    """
    Loss function for AIFS.
    
    Combines an invariance loss (classification loss on perturbed inputs)
    and a sensitivity loss (penalty for relying on perturbed dimensions).
    """
    
    def __init__(self, lambda_sens: float = 0.1):
        """
        Initialize the AIFS loss.
        
        Args:
            lambda_sens: Weight for the sensitivity loss
        """
        super().__init__()
        self.lambda_sens = lambda_sens
        self.base_criterion = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        logits_original: torch.Tensor, 
        logits_perturbed: torch.Tensor, 
        labels: torch.Tensor,
        latent_perturbed: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the AIFS loss.
        
        Args:
            logits_original: Logits from the original input
            logits_perturbed: Logits from the perturbed input
            labels: Ground truth labels
            latent_perturbed: Perturbed latent representation
            mask: Intervention mask used
            
        Returns:
            Tuple of (total loss, loss components)
        """
        # Classification loss on original input
        loss_orig = self.base_criterion(logits_original, labels)
        
        # Invariance loss (classification loss on perturbed input)
        loss_inv = self.base_criterion(logits_perturbed, labels)
        
        # Compute sensitivity loss (L2 norm of gradients w.r.t. perturbed dimensions)
        # This penalizes reliance on perturbed dimensions
        latent_perturbed.requires_grad_(True)
        logits_sens = F.softmax(logits_perturbed, dim=1)
        
        # Compute gradient of logits w.r.t. perturbed latent
        selected_logits = logits_sens[torch.arange(logits_sens.size(0)), labels]
        gradients = torch.autograd.grad(
            selected_logits.sum(), 
            latent_perturbed,
            create_graph=True
        )[0]
        
        # Focus only on perturbed dimensions (apply mask)
        masked_gradients = gradients * mask.view(1, -1)
        
        # Sensitivity loss is the L2 norm of masked gradients
        loss_sens = torch.norm(masked_gradients, p=2, dim=1).mean()
        
        # Total loss
        total_loss = loss_orig + loss_inv + self.lambda_sens * loss_sens
        
        # Return total loss and components
        loss_components = {
            "orig": loss_orig,
            "inv": loss_inv,
            "sens": loss_sens
        }
        
        return total_loss, loss_components, gradients


def train_aifs(
    model: AIFS,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lambda_sens: float = 0.1,
    lr: float = 0.001,
    num_epochs: int = 50,
    update_masks_every: int = 5,
    intervention_prob: float = 0.5,
    patience: int = 10,
    save_path: str = "aifs_model.pt"
) -> Dict[str, List[float]]:
    """
    Train the AIFS model.
    
    Args:
        model: AIFS model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        lambda_sens: Weight for sensitivity loss
        lr: Learning rate
        num_epochs: Number of epochs
        update_masks_every: Update intervention masks every N batches
        intervention_prob: Probability of applying intervention during training
        patience: Early stopping patience
        save_path: Path to save the best model
        
    Returns:
        Dictionary of training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = AIFSLoss(lambda_sens=lambda_sens)
    
    # For early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # For tracking metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "worst_group_acc": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass with original inputs
            logits_orig, _, _ = model(inputs, apply_intervention=False)
            
            # Decide whether to apply intervention for this batch
            if torch.rand(1).item() < intervention_prob:
                # Forward pass with intervention
                logits_perturbed, latent_perturbed, mask_idx = model(inputs, apply_intervention=True)
                
                # Get the mask used
                mask = model.intervention.intervention_masks[mask_idx]
                
                # Compute loss
                loss, loss_components, gradients = criterion(
                    logits_orig, logits_perturbed, labels, latent_perturbed, mask
                )
                
                # Update mask importance every N batches
                if batch_count % update_masks_every == 0:
                    model.intervention.update_mask_importance(gradients.detach(), mask_idx)
            else:
                # No intervention, just standard classification loss
                loss = F.cross_entropy(logits_orig, labels)
                loss_components = {"orig": loss, "inv": torch.tensor(0.0), "sens": torch.tensor(0.0)}
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits_orig.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            batch_count += 1
        
        # Calculate training metrics
        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass without intervention for validation
                logits, _, _ = model(inputs, apply_intervention=False)
                
                # Compute loss
                loss = F.cross_entropy(logits, labels)
                
                # Track metrics
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Calculate worst group accuracy (placeholder - will be implemented in evaluation)
        worst_group_acc = 0.0  # This will be properly implemented in evaluation
        
        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_accuracy)
        metrics["val_acc"].append(val_accuracy)
        metrics["worst_group_acc"].append(worst_group_acc)
        
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


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    group_aware: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        group_aware: Whether the test data has group labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    test_correct = 0
    test_total = 0
    group_correct = {}
    group_total = {}
    
    with torch.no_grad():
        for batch in test_loader:
            if group_aware and len(batch) == 3:
                inputs, labels, groups = batch
                inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
                
                # Initialize group counters if needed
                for g in groups.unique():
                    g_item = g.item()
                    if g_item not in group_correct:
                        group_correct[g_item] = 0
                        group_total[g_item] = 0
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                groups = None
            
            # Forward pass
            if isinstance(model, AIFS):
                outputs, _, _ = model(inputs, apply_intervention=False)
            else:
                outputs = model(inputs)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Calculate group-wise accuracy if groups are available
            if groups is not None:
                for i, g in enumerate(groups):
                    g_item = g.item()
                    group_total[g_item] += 1
                    if predicted[i] == labels[i]:
                        group_correct[g_item] += 1
    
    # Calculate metrics
    accuracy = 100 * test_correct / test_total
    
    metrics = {"accuracy": accuracy}
    
    # Add group-wise metrics if available
    if group_aware and len(group_correct) > 0:
        group_accuracies = {}
        for g in group_correct:
            if group_total[g] > 0:
                group_accuracies[g] = 100 * group_correct[g] / group_total[g]
            else:
                group_accuracies[g] = 0.0
        
        # Worst group accuracy
        worst_group_acc = min(group_accuracies.values())
        metrics["worst_group_accuracy"] = worst_group_acc
        metrics["group_accuracies"] = group_accuracies
    
    return metrics


# Example usage:
# model = AIFS(num_classes=10)
# metrics = train_aifs(model, train_loader, val_loader)
# eval_metrics = evaluate_model(model, test_loader)