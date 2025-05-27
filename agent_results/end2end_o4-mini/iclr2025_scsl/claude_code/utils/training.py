"""
Training utilities for training models with different robustification methods.
"""

import os
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Callable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from .metrics import compute_accuracy, compute_worst_group_accuracy

# Setup logging
logger = logging.getLogger(__name__)


class Trainer:
    """
    Base trainer class for training models with different robustification methods.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs: int = 10,
        modality: str = "multimodal",
        save_dir: str = "checkpoints",
        exp_name: str = "baseline",
        metadata: Optional[Dict] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            num_epochs: Number of training epochs
            modality: Data modality ("image", "text", or "multimodal")
            save_dir: Directory to save checkpoints and logs
            exp_name: Experiment name for saving files
            metadata: Dataset metadata for computing group accuracy
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.modality = modality
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.metadata = metadata
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_acc": 0.0,
            "worst_group_acc": 0.0,
            "best_epoch": 0,
            "training_time": 0.0
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch in self.train_loader:
            # Extract data based on modality
            if self.modality == "image":
                images, labels = batch
                inputs = images.to(self.device)
            elif self.modality == "text":
                texts, labels = batch
                inputs = texts.to(self.device)
            else:  # multimodal
                images, texts, labels = batch
                inputs = (images.to(self.device), texts.to(self.device))
                
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                if self.modality == "multimodal":
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs)
                    
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on the given loader.
        
        Args:
            loader: DataLoader to validate on
            
        Returns:
            Tuple of (average loss, accuracy) for the validation
        """
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                # Extract data based on modality
                if self.modality == "image":
                    images, labels = batch
                    inputs = images.to(self.device)
                elif self.modality == "text":
                    texts, labels = batch
                    inputs = texts.to(self.device)
                else:  # multimodal
                    images, texts, labels = batch
                    inputs = (images.to(self.device), texts.to(self.device))
                    
                labels = labels.to(self.device)
                
                # Forward pass
                if self.modality == "multimodal":
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs)
                    
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels).item()
                total_samples += labels.size(0)
        
        # Calculate metrics
        avg_loss = running_loss / total_samples
        accuracy = running_corrects / total_samples
        
        return avg_loss, accuracy
    
    def train(self) -> Dict:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        # Timer for training duration
        start_time = time.time()
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(self.num_epochs):
            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            # Print epoch summary
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check if this is the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                # Save the best model
                model_path = os.path.join(self.save_dir, f"{self.exp_name}_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, model_path)
                
                logger.info(f"Saved best model at epoch {epoch+1} with validation accuracy {val_acc:.4f}")
        
        # Training time
        training_time = time.time() - start_time
        self.history["training_time"] = training_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        model_path = os.path.join(self.save_dir, f"{self.exp_name}_final.pth")
        torch.save({
            "epoch": self.num_epochs - 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }, model_path)
        
        # Load best model for evaluation
        best_model_path = os.path.join(self.save_dir, f"{self.exp_name}_best.pth")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate on test set
        _, test_acc = self.validate(self.test_loader)
        self.history["test_acc"] = test_acc
        self.history["best_epoch"] = best_epoch
        
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Compute worst-group accuracy if metadata is available
        if self.metadata is not None and hasattr(self.test_loader.dataset, "return_attributes"):
            # Create a new test loader with attribute return
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data.dataset import SpurGenDataset
            test_dataset = SpurGenDataset(
                data_dir=self.test_loader.dataset.data_dir,
                split="test",
                modality=self.modality,
                transform=self.test_loader.dataset.transform,
                return_attributes=True
            )
            
            test_loader_with_attrs = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.test_loader.batch_size,
                shuffle=False,
                num_workers=self.test_loader.num_workers
            )
            
            # Compute worst-group accuracy
            group_metrics = compute_worst_group_accuracy(
                self.model,
                test_loader_with_attrs,
                self.device,
                self.metadata["num_classes"],
                self.metadata["spurious_channels"],
                self.modality
            )
            
            self.history["worst_group_acc"] = group_metrics["worst_group_accuracy"]
            self.history["worst_group"] = group_metrics["worst_group"]
            self.history["group_accuracies"] = group_metrics["group_accuracies"]
            
            logger.info(f"Worst-group accuracy: {self.history['worst_group_acc']:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.history
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.exp_name} - Loss Curves")
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Accuracy")
        plt.plot(self.history["val_acc"], label="Validation Accuracy")
        plt.axhline(y=self.history["test_acc"], color="r", linestyle="--", label="Test Accuracy")
        if "worst_group_acc" in self.history:
            plt.axhline(y=self.history["worst_group_acc"], color="g", linestyle="--", label="Worst-group Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{self.exp_name} - Accuracy Curves")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.save_dir, f"{self.exp_name}_training_curves.png"))
        plt.close()


class IRMTrainer(Trainer):
    """
    Trainer for Invariant Risk Minimization (IRM).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs: int = 10,
        modality: str = "multimodal",
        save_dir: str = "checkpoints",
        exp_name: str = "irm",
        metadata: Optional[Dict] = None,
        irm_lambda: float = 1.0,
        irm_penalty_anneal_iters: int = 500
    ):
        """
        Initialize the IRM trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            num_epochs: Number of training epochs
            modality: Data modality ("image", "text", or "multimodal")
            save_dir: Directory to save checkpoints and logs
            exp_name: Experiment name for saving files
            metadata: Dataset metadata for computing group accuracy
            irm_lambda: Weight for the IRM penalty
            irm_penalty_anneal_iters: Number of iterations to anneal the IRM penalty
        """
        super().__init__(
            model, train_loader, val_loader, test_loader, criterion, optimizer,
            scheduler, device, num_epochs, modality, save_dir, exp_name, metadata
        )
        
        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.step = 0
    
    def irm_penalty(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the IRM penalty as the norm of the gradient of the loss
        with respect to a dummy classifier layer.
        
        Args:
            logits: Model outputs
            y: Ground truth labels
            
        Returns:
            IRM penalty term
        """
        scale = torch.ones((1, logits.size(-1))).to(self.device).requires_grad_()
        loss = self.criterion(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch with IRM.
        
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_irm_penalty = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch in self.train_loader:
            # Extract data based on modality
            if self.modality == "image":
                images, labels = batch
                inputs = images.to(self.device)
            elif self.modality == "text":
                texts, labels = batch
                inputs = texts.to(self.device)
            else:  # multimodal
                images, texts, labels = batch
                inputs = (images.to(self.device), texts.to(self.device))
                
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                if self.modality == "multimodal":
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs)
                    
                # ERM loss
                erm_loss = self.criterion(outputs, labels)
                
                # IRM penalty
                irm_penalty = self.irm_penalty(outputs, labels)
                
                # Penalty weight (annealing)
                penalty_weight = self.irm_lambda if self.step >= self.irm_penalty_anneal_iters else 0.0
                
                # Total loss
                loss = erm_loss + penalty_weight * irm_penalty
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                self.step += 1
            
            # Statistics
            running_loss += erm_loss.item() * labels.size(0)  # Track only ERM loss for comparison
            running_irm_penalty += irm_penalty.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_irm_penalty = running_irm_penalty / total_samples
        epoch_acc = running_corrects / total_samples
        
        logger.info(f"IRM Penalty: {epoch_irm_penalty:.4f}, Penalty Weight: {penalty_weight:.4f}")
        
        return epoch_loss, epoch_acc


class GroupDROTrainer(Trainer):
    """
    Trainer for Group Distributionally Robust Optimization (Group-DRO).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs: int = 10,
        modality: str = "multimodal",
        save_dir: str = "checkpoints",
        exp_name: str = "group_dro",
        metadata: Optional[Dict] = None,
        group_weights_step_size: float = 0.01,
        group_weights_lr: float = 0.1
    ):
        """
        Initialize the Group-DRO trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data with return_attributes=True
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            num_epochs: Number of training epochs
            modality: Data modality ("image", "text", or "multimodal")
            save_dir: Directory to save checkpoints and logs
            exp_name: Experiment name for saving files
            metadata: Dataset metadata for computing group accuracy
            group_weights_step_size: Step size for group weights update
            group_weights_lr: Learning rate for group weights
        """
        super().__init__(
            model, train_loader, val_loader, test_loader, criterion, optimizer,
            scheduler, device, num_epochs, modality, save_dir, exp_name, metadata
        )
        
        self.group_weights_step_size = group_weights_step_size
        self.group_weights_lr = group_weights_lr
        
        # Initialize group weights
        self.num_groups = self._get_num_groups()
        self.group_weights = torch.ones(self.num_groups).to(self.device) / self.num_groups
    
    def _get_num_groups(self) -> int:
        """
        Determine the number of groups based on dataset metadata.
        
        Returns:
            Number of groups
        """
        if self.metadata is None:
            raise ValueError("Metadata is required for Group-DRO training")
        
        num_classes = self.metadata["num_classes"]
        
        # Count number of possible attribute combinations
        num_combinations = 1
        for channel in self.metadata["spurious_channels"].values():
            num_combinations *= len(channel["attributes"])
            
        # Total number of groups is classes * attribute combinations
        return num_classes * num_combinations
    
    def _get_group_idx(self, class_idx: int, attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Map each sample to its group index based on class and attributes.
        
        Args:
            class_idx: Class indices (batch)
            attributes: Dictionary of attribute indices for each channel (batch)
            
        Returns:
            Group indices (batch)
        """
        batch_size = class_idx.size(0)
        
        # Number of attribute combinations
        num_combinations = 1
        for channel in self.metadata["spurious_channels"].values():
            num_combinations *= len(channel["attributes"])
        
        # Compute group indices
        # First, get base indices from class (each class has num_combinations groups)
        group_idx = class_idx * num_combinations
        
        # Then add offset from attributes
        # For each channel, multiply previous channels' sizes and add attribute index
        offset = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        multiplier = 1
        
        channels = sorted(self.metadata["spurious_channels"].keys())  # Ensure consistent order
        for channel in channels:
            attr_idx = attributes[channel]
            offset += attr_idx * multiplier
            multiplier *= len(self.metadata["spurious_channels"][channel]["attributes"])
            
        group_idx = group_idx + offset
        
        return group_idx
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch with Group-DRO.
        
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Track losses for each group
        group_losses = torch.zeros(self.num_groups).to(self.device)
        group_counts = torch.zeros(self.num_groups).to(self.device)
        
        for batch in self.train_loader:
            # Extract data based on modality
            if self.modality == "image":
                images, labels, attributes = batch
                inputs = images.to(self.device)
            elif self.modality == "text":
                texts, labels, attributes = batch
                inputs = texts.to(self.device)
            else:  # multimodal
                images, texts, labels, attributes = batch
                inputs = (images.to(self.device), texts.to(self.device))
                
            labels = labels.to(self.device)
            for k, v in attributes.items():
                attributes[k] = v.to(self.device)
            
            # Get group indices
            group_idx = self._get_group_idx(labels, attributes)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                if self.modality == "multimodal":
                    outputs = self.model(inputs[0], inputs[1])
                else:
                    outputs = self.model(inputs)
                
                # Compute per-sample losses
                per_sample_losses = F.cross_entropy(outputs, labels, reduction="none")
                
                # Update group losses
                for i, g in enumerate(group_idx):
                    group_losses[g] += per_sample_losses[i].item()
                    group_counts[g] += 1
                
                # Compute weighted loss
                group_weights = F.softmax(self.group_weights, dim=0)
                loss = 0
                for i, g in enumerate(group_idx):
                    loss += group_weights[g] * per_sample_losses[i]
                loss = loss / labels.size(0)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        # Update group weights based on group losses
        if torch.sum(group_counts) > 0:
            # Compute average loss per group
            avg_group_losses = torch.zeros_like(group_losses)
            for g in range(self.num_groups):
                if group_counts[g] > 0:
                    avg_group_losses[g] = group_losses[g] / group_counts[g]
                else:
                    avg_group_losses[g] = 0.0
            
            # Update group weights
            self.group_weights = self.group_weights + self.group_weights_step_size * (
                avg_group_losses - self.group_weights_lr * torch.log(self.group_weights + 1e-8)
            )
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc


class AdversarialDebiasing(Trainer):
    """
    Trainer for Adversarial Feature Debiasing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs: int = 10,
        modality: str = "multimodal",
        save_dir: str = "checkpoints",
        exp_name: str = "adversarial",
        metadata: Optional[Dict] = None,
        adversary: torch.nn.Module = None,
        adv_optimizer: torch.optim.Optimizer = None,
        lambda_adv: float = 1.0,
        adv_channel: str = "background"
    ):
        """
        Initialize the Adversarial Debiasing trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data with return_attributes=True
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            num_epochs: Number of training epochs
            modality: Data modality ("image", "text", or "multimodal")
            save_dir: Directory to save checkpoints and logs
            exp_name: Experiment name for saving files
            metadata: Dataset metadata for computing group accuracy
            adversary: Adversarial discriminator model
            adv_optimizer: Optimizer for the adversary
            lambda_adv: Weight for adversarial loss
            adv_channel: Spurious channel to debias
        """
        super().__init__(
            model, train_loader, val_loader, test_loader, criterion, optimizer,
            scheduler, device, num_epochs, modality, save_dir, exp_name, metadata
        )
        
        self.adversary = adversary
        self.adv_optimizer = adv_optimizer
        self.lambda_adv = lambda_adv
        self.adv_channel = adv_channel
        
        # Check if adversary is provided
        if self.adversary is None:
            raise ValueError("Adversary model is required for Adversarial Debiasing")
            
        # Move adversary to device
        self.adversary.to(self.device)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch with Adversarial Debiasing.
        
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        self.adversary.train()
        
        running_loss = 0.0
        running_adv_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch in self.train_loader:
            # Extract data based on modality
            if self.modality == "image":
                images, labels, attributes = batch
                inputs = images.to(self.device)
            elif self.modality == "text":
                texts, labels, attributes = batch
                inputs = texts.to(self.device)
            else:  # multimodal
                images, texts, labels, attributes = batch
                inputs = (images.to(self.device), texts.to(self.device))
                
            labels = labels.to(self.device)
            
            # Get spurious attribute labels for the specified channel
            spurious_labels = attributes[self.adv_channel].to(self.device)
            
            # Step 1: Train adversary on fixed features
            self.adv_optimizer.zero_grad()
            
            # Forward pass to get features
            with torch.no_grad():
                if self.modality == "multimodal":
                    features = self.model.get_features(inputs[0], inputs[1])
                else:
                    features = self.model.get_features(inputs)
            
            # Forward pass through adversary
            adv_outputs = self.adversary(features.detach())
            adv_loss = F.cross_entropy(adv_outputs, spurious_labels.squeeze())
            
            # Backward pass for adversary
            adv_loss.backward()
            self.adv_optimizer.step()
            
            # Step 2: Train main model with adversarial loss
            self.optimizer.zero_grad()
            
            # Forward pass through main model
            if self.modality == "multimodal":
                outputs, features = self.model(inputs[0], inputs[1], return_features=True)
            else:
                outputs, features = self.model(inputs, return_features=True)
                
            # Classification loss
            cls_loss = self.criterion(outputs, labels)
            
            # Adversarial loss
            adv_outputs = self.adversary(features)
            adv_loss = F.cross_entropy(adv_outputs, spurious_labels.squeeze())
            
            # Total loss: minimize classification loss, maximize adversarial loss
            loss = cls_loss - self.lambda_adv * adv_loss
            
            # Backward pass for main model
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += cls_loss.item() * labels.size(0)  # Track only classification loss
            running_adv_loss += adv_loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_adv_loss = running_adv_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        logger.info(f"Adversarial Loss: {epoch_adv_loss:.4f}")
        
        return epoch_loss, epoch_acc


class ContrastiveTrainer(Trainer):
    """
    Trainer for Contrastive Augmentation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs: int = 10,
        modality: str = "multimodal",
        save_dir: str = "checkpoints",
        exp_name: str = "contrastive",
        metadata: Optional[Dict] = None,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.5
    ):
        """
        Initialize the Contrastive Augmentation trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data with shuffled pairs
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            num_epochs: Number of training epochs
            modality: Data modality ("image", "text", or "multimodal")
            save_dir: Directory to save checkpoints and logs
            exp_name: Experiment name for saving files
            metadata: Dataset metadata for computing group accuracy
            lambda_contrastive: Weight for contrastive loss
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__(
            model, train_loader, val_loader, test_loader, criterion, optimizer,
            scheduler, device, num_epochs, modality, save_dir, exp_name, metadata
        )
        
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
    
    def contrastive_loss(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between pairs of features.
        
        Args:
            features1: Features from original samples
            features2: Features from shuffled samples
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features1, features2.t()) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(similarity.size(0)).to(self.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.t(), labels)
        
        return loss / 2.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch with Contrastive Augmentation.
        
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch in self.train_loader:
            # Extract data based on modality
            if self.modality == "image":
                orig_images, shuffled_images, labels = batch
                orig_inputs = orig_images.to(self.device)
                shuffled_inputs = shuffled_images.to(self.device)
            elif self.modality == "text":
                orig_texts, shuffled_texts, labels = batch
                orig_inputs = orig_texts.to(self.device)
                shuffled_inputs = shuffled_texts.to(self.device)
            else:  # multimodal
                orig_images, orig_texts, shuffled_images, shuffled_texts, labels = batch
                orig_inputs = (orig_images.to(self.device), orig_texts.to(self.device))
                shuffled_inputs = (shuffled_images.to(self.device), shuffled_texts.to(self.device))
                
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                # Forward pass with original inputs
                if self.modality == "multimodal":
                    orig_outputs, orig_features = self.model(orig_inputs[0], orig_inputs[1], return_features=True)
                else:
                    orig_outputs, orig_features = self.model(orig_inputs, return_features=True)
                    
                # Forward pass with shuffled inputs
                if self.modality == "multimodal":
                    shuffled_outputs, shuffled_features = self.model(shuffled_inputs[0], shuffled_inputs[1], return_features=True)
                else:
                    shuffled_outputs, shuffled_features = self.model(shuffled_inputs, return_features=True)
                    
                # Classification loss
                cls_loss = self.criterion(orig_outputs, labels)
                
                # Contrastive loss
                cont_loss = self.contrastive_loss(orig_features, shuffled_features)
                
                # Total loss
                loss = cls_loss + self.lambda_contrastive * cont_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += cls_loss.item() * labels.size(0)  # Track only classification loss
            running_contrastive_loss += cont_loss.item() * labels.size(0)
            _, preds = torch.max(orig_outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_contrastive_loss = running_contrastive_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        logger.info(f"Contrastive Loss: {epoch_contrastive_loss:.4f}")
        
        return epoch_loss, epoch_acc