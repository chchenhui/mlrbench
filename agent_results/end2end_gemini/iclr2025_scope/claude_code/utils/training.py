"""
Training utilities for MeLPA and baseline methods.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Union


class MetaTrainer:
    """
    Trainer for meta-learning algorithms.
    """
    def __init__(
        self,
        model,
        device,
        meta_optimizer,
        logger=None,
        log_interval=10,
        save_dir=None
    ):
        self.model = model
        self.device = device
        self.meta_optimizer = meta_optimizer
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.log_interval = log_interval
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.meta_train_losses = []
        self.meta_valid_losses = []
    
    def meta_train_step(self, task_batch):
        """
        Execute a single meta-training step.
        
        Args:
            task_batch: Batch containing support and query sets for a task
        
        Returns:
            Meta-learning loss
        """
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        # Extract support and query data
        support_batch = {
            k: v.to(self.device) for k, v in task_batch["support"].items()
        }
        query_batch = {
            k: v.to(self.device) for k, v in task_batch["query"].items()
        }
        
        # Create a temporary adapter for this task
        adapter_name = f"temp_adapter_{int(time.time())}"
        self.model.model.add_adapter(adapter_name)
        
        # Meta-learning step
        meta_loss = self.model.meta_train_step(support_batch, query_batch, adapter_name)
        
        # Backpropagate through the meta-learning process
        meta_loss.backward()
        
        # Clip gradients to avoid explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update meta-parameters
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def meta_validate(self, val_tasks, n_tasks=10, inner_steps=5):
        """
        Validate the meta-learning model on validation tasks.
        
        Args:
            val_tasks: Validation tasks dataset
            n_tasks: Number of tasks to evaluate on
            inner_steps: Number of inner loop optimization steps
        
        Returns:
            Average meta-validation loss
        """
        self.model.eval()
        
        # Sample validation tasks
        if n_tasks > len(val_tasks):
            n_tasks = len(val_tasks)
        
        val_indices = np.random.choice(len(val_tasks), n_tasks, replace=False)
        val_losses = []
        
        for idx in val_indices:
            task_batch = val_tasks[idx]
            
            # Extract support and query data
            support_batch = {
                k: v.to(self.device) for k, v in task_batch["support"].items()
            }
            query_batch = {
                k: v.to(self.device) for k, v in task_batch["query"].items()
            }
            
            # Create a temporary adapter for this task
            adapter_name = f"temp_val_adapter_{int(time.time())}"
            self.model.model.add_adapter(adapter_name)
            
            # Meta-validation (no gradient tracking)
            with torch.no_grad():
                meta_loss = self.model.meta_train_step(support_batch, query_batch, adapter_name)
                val_losses.append(meta_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss
    
    def meta_train(
        self,
        train_tasks_dataloader,
        val_tasks=None,
        n_meta_epochs=1000,
        early_stopping_patience=20,
        save_best=True
    ):
        """
        Train the meta-learning model.
        
        Args:
            train_tasks_dataloader: DataLoader providing meta-training tasks
            val_tasks: Optional validation tasks dataset
            n_meta_epochs: Number of meta-training epochs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            save_best: Whether to save the best model
        
        Returns:
            Dictionary with training metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_meta_epochs):
            epoch_losses = []
            
            # Meta-training epoch
            with tqdm(train_tasks_dataloader, desc=f"Meta-epoch {epoch+1}/{n_meta_epochs}") as pbar:
                for i, task_batch in enumerate(pbar):
                    loss = self.meta_train_step(task_batch)
                    epoch_losses.append(loss)
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss})
                    
                    # Log periodically
                    if (i + 1) % self.log_interval == 0:
                        self.logger.info(f"Meta-step {i+1}, Loss: {loss:.6f}")
            
            # Calculate average epoch loss
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.meta_train_losses.append(avg_epoch_loss)
            
            # Meta-validation if validation tasks provided
            if val_tasks is not None:
                avg_val_loss = self.meta_validate(val_tasks)
                self.meta_valid_losses.append(avg_val_loss)
                self.logger.info(
                    f"Meta-epoch {epoch+1}: Train Loss: {avg_epoch_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}"
                )
                
                # Early stopping and model saving
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    if save_best and self.save_dir:
                        self._save_model(os.path.join(self.save_dir, "best_meta_model.pt"))
                        self.logger.info(f"Saved best model with validation loss {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(
                            f"Early stopping after {epoch+1} epochs. "
                            f"Best validation loss: {best_val_loss:.6f}"
                        )
                        break
            else:
                self.logger.info(f"Meta-epoch {epoch+1}: Train Loss: {avg_epoch_loss:.6f}")
        
        # Save final model
        if self.save_dir:
            self._save_model(os.path.join(self.save_dir, "final_meta_model.pt"))
        
        # Save training curves
        self._save_training_curves()
        
        return {
            "train_losses": self.meta_train_losses,
            "valid_losses": self.meta_valid_losses,
            "best_val_loss": best_val_loss if val_tasks is not None else None
        }
    
    def _save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict()
        }, path)
    
    def _save_training_curves(self):
        """Save training and validation loss curves."""
        if self.save_dir:
            metrics = {
                "meta_train_losses": self.meta_train_losses,
                "meta_valid_losses": self.meta_valid_losses
            }
            with open(os.path.join(self.save_dir, "meta_training_metrics.json"), "w") as f:
                json.dump(metrics, f)


class ContinualLearningTrainer:
    """
    Trainer for continual learning with adapters.
    Handles sequential task adaptation and evaluation.
    """
    def __init__(
        self,
        model,
        device,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        logger=None,
        log_interval=10,
        save_dir=None
    ):
        self.model = model
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {"lr": 0.001}
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.log_interval = log_interval
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Track metrics for all tasks seen so far
        self.task_metrics = {}
        self.training_curves = {}
    
    def train_task(
        self,
        adapter_name,
        train_dataloader,
        val_dataloader=None,
        n_epochs=10,
        early_stopping_patience=5
    ):
        """
        Train an adapter on a single task.
        
        Args:
            adapter_name: Name of the adapter to train
            train_dataloader: DataLoader for task training data
            val_dataloader: Optional DataLoader for task validation data
            n_epochs: Number of training epochs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
        
        Returns:
            Dictionary with task-specific training metrics
        """
        # Ensure the adapter exists
        if not hasattr(self.model.model, "adapter_controllers"):
            self.model.model.add_adapter(adapter_name)
        else:
            # Check if adapter exists in the first controller (as a proxy for all controllers)
            first_controller = next(iter(self.model.model.adapter_controllers.values()))
            if adapter_name not in first_controller.adapters:
                self.model.model.add_adapter(adapter_name)
        
        # Get adapter parameters
        adapter_params = self.model.model.get_adapter_parameters(adapter_name)
        
        # Create optimizer for this adapter
        optimizer = self.optimizer_class(adapter_params, **self.optimizer_kwargs)
        
        # Initialize tracking metrics
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_losses = []
            
            with tqdm(train_dataloader, desc=f"Task {adapter_name}, Epoch {epoch+1}/{n_epochs}") as pbar:
                for i, batch in enumerate(pbar):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    optimizer.zero_grad()
                    inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(inputs, adapter_name=adapter_name)
                    loss = F.cross_entropy(outputs.logits, batch['labels'])
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track loss
                    epoch_losses.append(loss.item())
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item()})
                    
                    # Log periodically
                    if (i + 1) % self.log_interval == 0:
                        self.logger.info(
                            f"Task {adapter_name}, Epoch {epoch+1}, Step {i+1}: Loss: {loss.item():.6f}"
                        )
            
            # Calculate average epoch loss
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_train_loss)
            
            # Validation if provided
            if val_dataloader is not None:
                avg_val_loss, val_accuracy = self._validate(adapter_name, val_dataloader)
                val_losses.append(avg_val_loss)
                
                self.logger.info(
                    f"Task {adapter_name}, Epoch {epoch+1}: "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best adapter weights
                    if self.save_dir:
                        self._save_adapter(adapter_name, os.path.join(self.save_dir, f"{adapter_name}_best.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(
                            f"Early stopping for task {adapter_name} after {epoch+1} epochs. "
                            f"Best validation loss: {best_val_loss:.6f}"
                        )
                        break
            else:
                self.logger.info(
                    f"Task {adapter_name}, Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}"
                )
        
        # Save final adapter weights
        if self.save_dir:
            self._save_adapter(adapter_name, os.path.join(self.save_dir, f"{adapter_name}_final.pt"))
        
        # Store metrics
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss if val_dataloader is not None else None
        }
        
        self.training_curves[adapter_name] = metrics
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
    
    def _save_adapter(self, adapter_name, path):
        """Save adapter weights to a file."""
        adapter_state_dict = {}
        for name, param in self.model.model.named_parameters():
            if adapter_name in name:
                adapter_state_dict[name] = param.data.cpu()
        
        torch.save(adapter_state_dict, path)
    
    def _load_adapter(self, adapter_name, path):
        """Load adapter weights from a file."""
        adapter_state_dict = torch.load(path)
        
        with torch.no_grad():
            for name, param in self.model.model.named_parameters():
                if name in adapter_state_dict:
                    param.copy_(adapter_state_dict[name])
    
    def evaluate_all_tasks(self, task_sequence, task_dataloaders):
        """
        Evaluate the model on all tasks seen so far to measure catastrophic forgetting.
        
        Args:
            task_sequence: List of task IDs in the order they were learned
            task_dataloaders: Dictionary mapping task IDs to their validation DataLoaders
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_tasks_metrics = {}
        
        for task_id in task_sequence:
            adapter_name = f"task_{task_id}"
            val_dataloader = task_dataloaders[task_id]
            
            # Evaluate on this task
            avg_loss, accuracy = self._validate(adapter_name, val_dataloader)
            
            all_tasks_metrics[task_id] = {
                "loss": avg_loss,
                "accuracy": accuracy
            }
            
            self.logger.info(
                f"Task {task_id} evaluation - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}"
            )
        
        return all_tasks_metrics
    
    def calculate_forgetting_metrics(self, task_sequence, initial_metrics, final_metrics):
        """
        Calculate metrics for catastrophic forgetting.
        
        Args:
            task_sequence: List of task IDs in the order they were learned
            initial_metrics: Dictionary of metrics right after learning each task
            final_metrics: Dictionary of metrics after learning all tasks
            
        Returns:
            Dictionary with forgetting metrics
        """
        n_tasks = len(task_sequence)
        
        # Average Accuracy (ACC) after learning all tasks
        acc = sum(final_metrics[task_id]["accuracy"] for task_id in task_sequence) / n_tasks
        
        # Backward Transfer (BWT)
        bwt_values = []
        for i in range(n_tasks - 1):
            task_id = task_sequence[i]
            r_n_i = final_metrics[task_id]["accuracy"]  # Accuracy on task i after learning all tasks
            r_i_i = initial_metrics[task_id]["accuracy"]  # Accuracy on task i after learning task i
            bwt_values.append(r_n_i - r_i_i)
        
        bwt = sum(bwt_values) / len(bwt_values) if bwt_values else 0
        
        # Forward Transfer (FWT) 
        # This requires a baseline performance on each task before any learning
        # For simplicity, we'll skip this metric as it's not directly available in our setup
        
        forgetting_metrics = {
            "average_accuracy": acc,
            "backward_transfer": bwt,
            # "forward_transfer": fwt  # Not calculated
        }
        
        return forgetting_metrics
    
    def train_on_task_sequence(
        self,
        task_sequence,
        batch_size=16,
        n_epochs_per_task=10,
        early_stopping_patience=5,
        track_forgetting=True
    ):
        """
        Train the model on a sequence of tasks for continual learning.
        
        Args:
            task_sequence: List of tasks in sequence, each with train and validation sets
            batch_size: Batch size for data loaders
            n_epochs_per_task: Number of epochs to train on each task
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            track_forgetting: Whether to evaluate forgetting metrics
            
        Returns:
            Dictionary with training and evaluation metrics
        """
        # Create DataLoaders for all tasks
        task_dataloaders = {}
        val_dataloaders = {}
        
        for task in task_sequence:
            task_id = task["task_id"]
            
            # Create DataLoader for training
            train_loader = DataLoader(
                task["train_set"],
                batch_size=batch_size,
                shuffle=True
            )
            
            # Create DataLoader for validation
            val_loader = DataLoader(
                task["val_set"],
                batch_size=batch_size,
                shuffle=False
            )
            
            task_dataloaders[task_id] = train_loader
            val_dataloaders[task_id] = val_loader
        
        # Track task order
        task_order = [task["task_id"] for task in task_sequence]
        
        # Track metrics for measuring catastrophic forgetting
        initial_task_metrics = {}  # Metrics right after learning each task
        
        # Train on each task in sequence
        for task in task_sequence:
            task_id = task["task_id"]
            adapter_name = f"task_{task_id}"
            
            self.logger.info(f"Training on task {task_id}")
            
            # Train on this task
            metrics = self.train_task(
                adapter_name=adapter_name,
                train_dataloader=task_dataloaders[task_id],
                val_dataloader=val_dataloaders[task_id],
                n_epochs=n_epochs_per_task,
                early_stopping_patience=early_stopping_patience
            )
            
            # Evaluate on all tasks seen so far if tracking forgetting
            if track_forgetting:
                # Measure performance on the just-learned task
                _, accuracy = self._validate(adapter_name, val_dataloaders[task_id])
                initial_task_metrics[task_id] = {
                    "accuracy": accuracy
                }
                
                # Only evaluate all tasks if we've seen more than one
                current_task_idx = task_order.index(task_id)
                if current_task_idx > 0:
                    tasks_so_far = task_order[:current_task_idx + 1]
                    task_metrics = {}
                    
                    for seen_task_id in tasks_so_far:
                        seen_adapter_name = f"task_{seen_task_id}"
                        _, seen_accuracy = self._validate(
                            seen_adapter_name,
                            val_dataloaders[seen_task_id]
                        )
                        
                        task_metrics[seen_task_id] = {
                            "accuracy": seen_accuracy
                        }
                    
                    self.task_metrics[f"after_task_{task_id}"] = task_metrics
        
        # Final evaluation on all tasks
        if track_forgetting:
            final_metrics = {}
            for task_id in task_order:
                adapter_name = f"task_{task_id}"
                _, accuracy = self._validate(adapter_name, val_dataloaders[task_id])
                final_metrics[task_id] = {
                    "accuracy": accuracy
                }
            
            self.task_metrics["final"] = final_metrics
            
            # Calculate forgetting metrics
            forgetting_metrics = self.calculate_forgetting_metrics(
                task_order,
                initial_task_metrics,
                final_metrics
            )
            
            self.task_metrics["forgetting_metrics"] = forgetting_metrics
            
            self.logger.info(
                f"Continual learning metrics - "
                f"Average Accuracy: {forgetting_metrics['average_accuracy']:.4f}, "
                f"Backward Transfer: {forgetting_metrics['backward_transfer']:.4f}"
            )
        
        # Save training curves and metrics
        if self.save_dir:
            with open(os.path.join(self.save_dir, "training_curves.json"), "w") as f:
                json.dump(self.training_curves, f)
            
            with open(os.path.join(self.save_dir, "task_metrics.json"), "w") as f:
                json.dump(self.task_metrics, f)
        
        return {
            "training_curves": self.training_curves,
            "task_metrics": self.task_metrics
        }


class MeLPATrainer(ContinualLearningTrainer):
    """
    Trainer for MeLPA-based continual learning.
    Extends ContinualLearningTrainer with meta-learning capabilities.
    """
    def __init__(
        self,
        melpa_model,
        device,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        logger=None,
        log_interval=10,
        save_dir=None,
        use_meta_init=True,
        use_meta_update=True
    ):
        super().__init__(
            model=melpa_model,
            device=device,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            logger=logger,
            log_interval=log_interval,
            save_dir=save_dir
        )
        self.use_meta_init = use_meta_init
        self.use_meta_update = use_meta_update
    
    def train_task(
        self,
        adapter_name,
        train_dataloader,
        val_dataloader=None,
        n_epochs=10,
        early_stopping_patience=5,
        task_context=None
    ):
        """
        Train an adapter on a single task using meta-learned initialization and updates.
        
        Args:
            adapter_name: Name of the adapter to train
            train_dataloader: DataLoader for task training data
            val_dataloader: Optional DataLoader for task validation data
            n_epochs: Number of training epochs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            task_context: Optional task context for meta-initialization
        
        Returns:
            Dictionary with task-specific training metrics
        """
        # Initialize adapter with meta-learned initialization
        if self.use_meta_init and self.model.init_network is not None:
            self.logger.info(f"Using meta-learned initialization for adapter {adapter_name}")
            self.model.initialize_adapter(adapter_name, task_context)
        else:
            # Fall back to standard initialization
            self.model.model.add_adapter(adapter_name)
        
        # Use standard optimizer if not using meta-learned updates
        if not self.use_meta_update or self.model.update_mechanism is None:
            return super().train_task(
                adapter_name=adapter_name,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience
            )
        
        # Meta-learned updates
        self.logger.info(f"Using meta-learned updates for adapter {adapter_name}")
        
        # Initialize tracking metrics
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_losses = []
            
            with tqdm(train_dataloader, desc=f"Task {adapter_name}, Epoch {epoch+1}/{n_epochs}") as pbar:
                for i, batch in enumerate(pbar):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Update with meta-learned update rule
                    self.model.update_adapter(adapter_name, batch)
                    
                    # Calculate loss for tracking (no gradient computation needed)
                    with torch.no_grad():
                        inputs = {k: v for k, v in batch.items() if k != 'labels'}
                        outputs = self.model(inputs, adapter_name=adapter_name)
                        loss = F.cross_entropy(outputs.logits, batch['labels'])
                        epoch_losses.append(loss.item())
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item()})
                    
                    # Log periodically
                    if (i + 1) % self.log_interval == 0:
                        self.logger.info(
                            f"Task {adapter_name}, Epoch {epoch+1}, Step {i+1}: Loss: {loss.item():.6f}"
                        )
            
            # Calculate average epoch loss
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_train_loss)
            
            # Validation if provided
            if val_dataloader is not None:
                avg_val_loss, val_accuracy = self._validate(adapter_name, val_dataloader)
                val_losses.append(avg_val_loss)
                
                self.logger.info(
                    f"Task {adapter_name}, Epoch {epoch+1}: "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best adapter weights
                    if self.save_dir:
                        self._save_adapter(adapter_name, os.path.join(self.save_dir, f"{adapter_name}_best.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(
                            f"Early stopping for task {adapter_name} after {epoch+1} epochs. "
                            f"Best validation loss: {best_val_loss:.6f}"
                        )
                        break
            else:
                self.logger.info(
                    f"Task {adapter_name}, Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}"
                )
        
        # Save final adapter weights
        if self.save_dir:
            self._save_adapter(adapter_name, os.path.join(self.save_dir, f"{adapter_name}_final.pt"))
        
        # Store metrics
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss if val_dataloader is not None else None
        }
        
        self.training_curves[adapter_name] = metrics
        return metrics