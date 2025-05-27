"""
Elastic Weight Consolidation (EWC) implementation for continual learning.
EWC is a regularization-based method that prevents catastrophic forgetting
by penalizing changes to parameters that are important for previously learned tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class EWCLoss(nn.Module):
    """
    EWC Loss implementation.
    Penalizes changes to important parameters for previous tasks.
    """
    def __init__(self, lambda_ewc=5000.0):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.fisher_matrices = {}  # Task-specific Fisher information matrices
        self.optimal_parameters = {}  # Task-specific optimal parameters
    
    def register_task(self, task_id, model, dataloader, device):
        """
        Compute and store Fisher information matrix and optimal parameters for a task.
        
        Args:
            task_id: ID of the task
            model: Model that has learned the task
            dataloader: DataLoader with task data
            device: Device to run computation on
        """
        # Store current parameter values as optimal
        self.optimal_parameters[task_id] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_parameters[task_id][name] = param.data.clone()
        
        # Compute Fisher information matrix
        fisher_matrix = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_matrix[name] = torch.zeros_like(param.data)
        
        model.eval()
        # Accumulate Fisher information
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            
            model.zero_grad()
            outputs = model(**inputs)
            
            # Compute log probability of predicted class
            log_probs = F.log_softmax(outputs.logits, dim=1)
            # Sample actions from the model's distribution
            sampled_classes = torch.multinomial(
                F.softmax(outputs.logits, dim=1),
                num_samples=1
            ).squeeze()
            # Compute loss for sampled actions
            loss = torch.mean(-log_probs.gather(1, sampled_classes.unsqueeze(1)))
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_matrix[name] += param.grad.data ** 2
        
        # Normalize by number of samples
        n_samples = len(dataloader.dataset)
        for name in fisher_matrix.keys():
            fisher_matrix[name] /= n_samples
        
        self.fisher_matrices[task_id] = fisher_matrix
    
    def forward(self, model, current_task_id=None):
        """
        Compute EWC loss.
        
        Args:
            model: Current model
            current_task_id: ID of the current task (optional)
        
        Returns:
            EWC regularization loss
        """
        if not self.fisher_matrices:
            # No previous tasks registered
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Sum over all previous tasks
        for task_id, fisher_matrix in self.fisher_matrices.items():
            if current_task_id is not None and task_id == current_task_id:
                # Skip current task
                continue
            
            # Compute weighted squared distance from optimal parameters
            for name, param in model.named_parameters():
                if name in fisher_matrix and name in self.optimal_parameters[task_id]:
                    optimal_param = self.optimal_parameters[task_id][name]
                    ewc_loss += (fisher_matrix[name] * (param - optimal_param) ** 2).sum()
        
        return self.lambda_ewc * ewc_loss


class EWCAdapterTrainer:
    """
    Trainer for adapter-based continual learning with EWC.
    """
    def __init__(
        self,
        model,
        device,
        lambda_ewc=5000.0,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None
    ):
        self.model = model
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {"lr": 0.001}
        
        # Initialize EWC loss
        self.ewc_loss = EWCLoss(lambda_ewc=lambda_ewc)
    
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
        Train an adapter on a task with EWC regularization.
        
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
            
            for batch in train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                outputs = self.model(inputs, adapter_name=adapter_name)
                
                # Task loss
                task_loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                # EWC regularization
                ewc_regularization = self.ewc_loss(self.model, current_task_id=task_id)
                
                # Total loss
                loss = task_loss + ewc_regularization
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
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
        
        # After training, compute and store Fisher information matrix
        self.ewc_loss.register_task(task_id, self.model, train_dataloader, self.device)
        
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