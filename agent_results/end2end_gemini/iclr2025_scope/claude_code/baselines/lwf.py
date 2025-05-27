"""
Learning without Forgetting (LwF) implementation for continual learning.
LwF uses knowledge distillation to preserve performance on old tasks when learning new ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm


class LwFLoss(nn.Module):
    """
    Learning without Forgetting loss.
    Uses knowledge distillation to preserve old task performance.
    """
    def __init__(self, temperature=2.0, lambda_old=1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_old = lambda_old
        self.old_task_models = {}  # Stores models for previous tasks
    
    def register_task(self, task_id, model):
        """
        Store model state for a task after learning it.
        
        Args:
            task_id: ID of the task
            model: Model that has learned the task
        """
        # Store a copy of the current model
        self.old_task_models[task_id] = deepcopy(model)
        # Set model to eval mode to ensure consistent outputs
        self.old_task_models[task_id].eval()
    
    def forward(self, model, inputs, current_task_id=None):
        """
        Compute LwF distillation loss.
        
        Args:
            model: Current model
            inputs: Input data (without labels)
            current_task_id: ID of the current task (optional)
        
        Returns:
            LwF distillation loss
        """
        if not self.old_task_models:
            # No previous tasks registered
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        distillation_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Get current model logits
        model.eval()  # Temporarily set to eval mode for consistent outputs
        with torch.no_grad():
            current_logits = model(**inputs).logits
        model.train()  # Set back to training mode
        
        # Sum over all previous tasks
        for task_id, old_model in self.old_task_models.items():
            if current_task_id is not None and task_id == current_task_id:
                # Skip current task
                continue
            
            # Get old model logits
            with torch.no_grad():
                old_logits = old_model(**inputs).logits
            
            # Compute distillation loss using soft targets
            distillation_loss += F.kl_div(
                F.log_softmax(current_logits / self.temperature, dim=1),
                F.softmax(old_logits / self.temperature, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
        
        return self.lambda_old * distillation_loss


class LwFAdapterTrainer:
    """
    Trainer for adapter-based continual learning with Learning without Forgetting.
    """
    def __init__(
        self,
        model,
        device,
        temperature=2.0,
        lambda_old=1.0,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None
    ):
        self.model = model
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {"lr": 0.001}
        
        # Initialize LwF loss
        self.lwf_loss = LwFLoss(temperature=temperature, lambda_old=lambda_old)
    
    def train_task(
        self,
        task_id,
        adapter_name,
        train_dataloader,
        val_dataloader=None,
        n_epochs=10,
        early_stopping_patience=5
    ):
        """
        Train an adapter on a task with LwF loss.
        
        Args:
            task_id: ID of the task
            adapter_name: Name of the adapter to train
            train_dataloader: DataLoader for task training data
            val_dataloader: Optional DataLoader for task validation data
            n_epochs: Number of training epochs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
        
        Returns:
            Dictionary with task-specific training metrics
        """
        # Ensure adapter exists
        self.model.model.add_adapter(adapter_name)
        
        # Get adapter parameters
        adapter_params = self.model.model.get_adapter_parameters(adapter_name)
        
        # Create optimizer
        optimizer = self.optimizer_class(adapter_params, **self.optimizer_kwargs)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_losses = []
            
            with tqdm(train_dataloader, desc=f"Task {task_id}, Epoch {epoch+1}/{n_epochs}") as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    optimizer.zero_grad()
                    inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(inputs, adapter_name=adapter_name)
                    
                    # Task loss
                    task_loss = F.cross_entropy(outputs.logits, batch['labels'])
                    
                    # LwF distillation loss
                    distillation_loss = self.lwf_loss(self.model, inputs, current_task_id=task_id)
                    
                    # Total loss
                    loss = task_loss + distillation_loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    pbar.set_postfix({"loss": loss.item()})
            
            # Calculate average epoch loss
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_train_loss)
            
            # Validation if provided
            if val_dataloader is not None:
                avg_val_loss, val_accuracy = self._validate(adapter_name, val_dataloader)
                val_losses.append(avg_val_loss)
                
                print(
                    f"Task {task_id}, Epoch {epoch+1}/{n_epochs}: "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(
                            f"Early stopping for task {task_id} after {epoch+1} epochs. "
                            f"Best validation loss: {best_val_loss:.6f}"
                        )
                        break
            else:
                print(f"Task {task_id}, Epoch {epoch+1}/{n_epochs}: Train Loss: {avg_train_loss:.6f}")
        
        # After training, register the adapter for this task
        self.lwf_loss.register_task(task_id, self.model)
        
        # Return metrics
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss if val_dataloader is not None else None
        }
        
        return metrics
    
    def _validate(self, adapter_name, val_dataloader):
        """
        Validate the model with a specific adapter on a validation set.
        
        Args:
            adapter_name: Name of the adapter to use
            val_dataloader: Validation DataLoader
            
        Returns:
            Tuple of (average validation loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                outputs = self.model(inputs, adapter_name=adapter_name)
                loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                # Track loss and accuracy
                total_loss += loss.item() * batch['labels'].size(0)
                pred = torch.argmax(outputs.logits, dim=1)
                correct += (pred == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy