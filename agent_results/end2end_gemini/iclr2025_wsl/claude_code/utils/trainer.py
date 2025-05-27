import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import logging
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for training and evaluating models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        model_name: str = "model",
        save_dir: str = "experiments",
        property_names: List[str] = ["accuracy", "robustness", "generalization_gap"],
        tensorboard_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to use for training
            scheduler: Learning rate scheduler
            model_name: Name of the model (for saving)
            save_dir: Directory to save model checkpoints
            property_names: Names of properties being predicted
            tensorboard_dir: Directory for TensorBoard logs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.model_name = model_name
        self.save_dir = save_dir
        self.property_names = property_names
        self.early_stopping_patience = early_stopping_patience
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # TensorBoard writer
        if tensorboard_dir is not None:
            self.writer = SummaryWriter(tensorboard_dir)
        else:
            self.writer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_mae': [],
            'val_mae': [],
            'test_mae': [],
            'learning_rate': [],
            'epoch_times': [],
            'property_maes': {prop: {'train': [], 'val': [], 'test': []} for prop in property_names},
            'property_r2s': {prop: {'train': [], 'val': [], 'test': []} for prop in property_names},
        }
        
        # Best validation loss and epoch
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, average MAE, per-property MAEs, per-property R2s)
        """
        self.model.train()
        epoch_loss = 0.0
        all_targets = []
        all_predictions = []
        
        for batch_idx, (tokens, targets) in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            # Move to device
            tokens = tokens.to(self.device)
            targets = targets.to(self.device).float()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(tokens)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item() * tokens.size(0)
            
            # Collect predictions and targets for metrics
            all_targets.append(targets.cpu().detach().numpy())
            all_predictions.append(outputs.cpu().detach().numpy())
        
        # Calculate average loss
        epoch_loss /= len(self.train_loader.dataset)
        
        # Concatenate all predictions and targets
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate overall MAE
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Calculate per-property MAEs and R2s
        property_maes = {}
        property_r2s = {}
        for i, prop in enumerate(self.property_names):
            prop_mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
            prop_r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            property_maes[prop] = prop_mae
            property_r2s[prop] = prop_r2
        
        return epoch_loss, mae, property_maes, property_r2s
    
    def evaluate(self, loader: DataLoader, desc: str = "Evaluating") -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
        """
        Evaluate the model on a data loader.
        
        Args:
            loader: DataLoader to evaluate on
            desc: Description for progress bar
            
        Returns:
            Tuple of (average loss, average MAE, per-property MAEs, per-property R2s)
        """
        self.model.eval()
        epoch_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (tokens, targets) in enumerate(tqdm(loader, desc=desc, leave=False)):
                # Move to device
                tokens = tokens.to(self.device)
                targets = targets.to(self.device).float()
                
                # Forward pass
                outputs = self.model(tokens)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Accumulate loss
                epoch_loss += loss.item() * tokens.size(0)
                
                # Collect predictions and targets for metrics
                all_targets.append(targets.cpu().detach().numpy())
                all_predictions.append(outputs.cpu().detach().numpy())
        
        # Calculate average loss
        epoch_loss /= len(loader.dataset)
        
        # Concatenate all predictions and targets
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate overall MAE
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Calculate per-property MAEs and R2s
        property_maes = {}
        property_r2s = {}
        for i, prop in enumerate(self.property_names):
            prop_mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
            prop_r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            property_maes[prop] = prop_mae
            property_r2s[prop] = prop_r2
        
        return epoch_loss, mae, property_maes, property_r2s
    
    def train(self, num_epochs: int, save_best: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            save_best: Whether to save the best model
            
        Returns:
            Training history
        """
        logger.info(f"Training {self.model_name} for {num_epochs} epochs")
        
        # Initial evaluation
        val_loss, val_mae, val_property_maes, val_property_r2s = self.evaluate(self.val_loader, "Initial Validation")
        test_loss, test_mae, test_property_maes, test_property_r2s = self.evaluate(self.test_loader, "Initial Test")
        
        logger.info(f"Initial Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        logger.info(f"Initial Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")
        
        # Initialize history
        self.history['val_loss'].append(val_loss)
        self.history['test_loss'].append(test_loss)
        self.history['val_mae'].append(val_mae)
        self.history['test_mae'].append(test_mae)
        
        for prop in self.property_names:
            self.history['property_maes'][prop]['val'].append(val_property_maes[prop])
            self.history['property_maes'][prop]['test'].append(test_property_maes[prop])
            self.history['property_r2s'][prop]['val'].append(val_property_r2s[prop])
            self.history['property_r2s'][prop]['test'].append(test_property_r2s[prop])
        
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_mae, train_property_maes, train_property_r2s = self.train_epoch()
            
            # Step learning rate scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Evaluate on validation set
            val_loss, val_mae, val_property_maes, val_property_r2s = self.evaluate(self.val_loader, "Validation")
            
            # Evaluate on test set
            test_loss, test_mae, test_property_maes, test_property_r2s = self.evaluate(self.test_loader, "Test")
            
            # Record time
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['test_mae'].append(test_mae)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            for prop in self.property_names:
                self.history['property_maes'][prop]['train'].append(train_property_maes[prop])
                self.history['property_maes'][prop]['val'].append(val_property_maes[prop])
                self.history['property_maes'][prop]['test'].append(test_property_maes[prop])
                self.history['property_r2s'][prop]['train'].append(train_property_r2s[prop])
                self.history['property_r2s'][prop]['val'].append(val_property_r2s[prop])
                self.history['property_r2s'][prop]['test'].append(test_property_r2s[prop])
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, "
                       f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, "
                       f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, "
                       f"LR: {current_lr:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Loss/test', test_loss, epoch)
                self.writer.add_scalar('MAE/train', train_mae, epoch)
                self.writer.add_scalar('MAE/val', val_mae, epoch)
                self.writer.add_scalar('MAE/test', test_mae, epoch)
                self.writer.add_scalar('LR', current_lr, epoch)
                
                for prop in self.property_names:
                    self.writer.add_scalar(f'MAE/{prop}/train', train_property_maes[prop], epoch)
                    self.writer.add_scalar(f'MAE/{prop}/val', val_property_maes[prop], epoch)
                    self.writer.add_scalar(f'MAE/{prop}/test', test_property_maes[prop], epoch)
                    self.writer.add_scalar(f'R2/{prop}/train', train_property_r2s[prop], epoch)
                    self.writer.add_scalar(f'R2/{prop}/val', val_property_r2s[prop], epoch)
                    self.writer.add_scalar(f'R2/{prop}/test', test_property_r2s[prop], epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                logger.info(f"New best validation loss: {val_loss:.4f}")
                
                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
            
            # Early stopping
            if epoch - self.best_epoch >= self.early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False)
        
        # Save history
        self.save_history()
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
        else:
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_history(self) -> None:
        """Save training history."""
        history_path = os.path.join(self.save_dir, f"{self.model_name}_history.json")
        
        # Convert numpy arrays to floats for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_json[key] = [float(x) for x in value]
            elif isinstance(value, dict):
                history_json[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        history_json[key][sub_key] = {}
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            history_json[key][sub_key][sub_sub_key] = [float(x) for x in sub_sub_value]
                    else:
                        history_json[key][sub_key] = [float(x) for x in sub_value]
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=4)
        
        logger.info(f"Saved history to {history_path}")
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].plot(self.history['test_loss'], label='Test')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train')
        axes[0, 1].plot(self.history['val_mae'], label='Validation')
        axes[0, 1].plot(self.history['test_mae'], label='Test')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Overall Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot property-specific MAE
        for i, prop in enumerate(self.property_names[:4]):  # Plot up to 4 properties
            row = 1
            col = i % 2
            
            if i < 2:
                axes[row, col].plot(self.history['property_maes'][prop]['train'], label='Train')
                axes[row, col].plot(self.history['property_maes'][prop]['val'], label='Validation')
                axes[row, col].plot(self.history['property_maes'][prop]['test'], label='Test')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('MAE')
                axes[row, col].set_title(f'{prop} MAE')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.close()
    
    def detailed_evaluation(self, loader: DataLoader, desc: str = "Detailed Evaluation") -> Dict[str, Any]:
        """
        Perform detailed evaluation on a data loader.
        
        Args:
            loader: DataLoader to evaluate on
            desc: Description for progress bar
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (tokens, targets) in enumerate(tqdm(loader, desc=desc, leave=False)):
                # Move to device
                tokens = tokens.to(self.device)
                targets = targets.to(self.device).float()
                
                # Forward pass
                outputs = self.model(tokens)
                
                # Collect predictions and targets for metrics
                all_targets.append(targets.cpu().detach().numpy())
                all_predictions.append(outputs.cpu().detach().numpy())
        
        # Concatenate all predictions and targets
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate metrics
        results = {}
        
        # Overall metrics
        results['mae'] = mean_absolute_error(all_targets, all_predictions)
        results['rmse'] = np.sqrt(mean_squared_error(all_targets, all_predictions))
        results['mse'] = mean_squared_error(all_targets, all_predictions)
        
        # Property-specific metrics
        results['property_metrics'] = {}
        for i, prop in enumerate(self.property_names):
            prop_results = {}
            prop_results['mae'] = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
            prop_results['rmse'] = np.sqrt(mean_squared_error(all_targets[:, i], all_predictions[:, i]))
            prop_results['mse'] = mean_squared_error(all_targets[:, i], all_predictions[:, i])
            prop_results['r2'] = r2_score(all_targets[:, i], all_predictions[:, i])
            
            # Calculate prediction error distribution
            errors = np.abs(all_predictions[:, i] - all_targets[:, i])
            prop_results['error_mean'] = np.mean(errors)
            prop_results['error_std'] = np.std(errors)
            prop_results['error_median'] = np.median(errors)
            prop_results['error_q1'] = np.percentile(errors, 25)
            prop_results['error_q3'] = np.percentile(errors, 75)
            
            results['property_metrics'][prop] = prop_results
        
        # Add raw predictions and targets
        results['predictions'] = all_predictions
        results['targets'] = all_targets
        
        return results
    
    def create_results_table(self, detailed_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a results table from detailed evaluation results.
        
        Args:
            detailed_results: Results from detailed_evaluation
            
        Returns:
            DataFrame with results
        """
        results_data = []
        
        for prop in self.property_names:
            metrics = detailed_results['property_metrics'][prop]
            
            row = {
                'Property': prop,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MSE': metrics['mse'],
                'R2': metrics['r2'],
                'Mean Error': metrics['error_mean'],
                'Error Std': metrics['error_std'],
                'Median Error': metrics['error_median'],
                'Q1 Error': metrics['error_q1'],
                'Q3 Error': metrics['error_q3'],
            }
            
            results_data.append(row)
        
        # Add overall row
        overall_row = {
            'Property': 'Overall',
            'MAE': detailed_results['mae'],
            'RMSE': detailed_results['rmse'],
            'MSE': detailed_results['mse'],
            'R2': '-',
            'Mean Error': '-',
            'Error Std': '-',
            'Median Error': '-',
            'Q1 Error': '-',
            'Q3 Error': '-',
        }
        
        results_data.append(overall_row)
        
        return pd.DataFrame(results_data)
    
    def plot_predictions_vs_targets(
        self, 
        detailed_results: Dict[str, Any], 
        save_dir: str = "figures",
    ) -> List[str]:
        """
        Plot predictions vs targets for each property.
        
        Args:
            detailed_results: Results from detailed_evaluation
            save_dir: Directory to save plots
            
        Returns:
            List of paths to saved plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = detailed_results['predictions']
        targets = detailed_results['targets']
        
        plot_paths = []
        
        for i, prop in enumerate(self.property_names):
            plt.figure(figsize=(8, 8))
            
            # Plot predictions vs targets
            plt.scatter(targets[:, i], predictions[:, i], alpha=0.5, label="Predictions")
            
            # Plot y=x line
            min_val = min(np.min(targets[:, i]), np.min(predictions[:, i]))
            max_val = max(np.max(targets[:, i]), np.max(predictions[:, i]))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y=x")
            
            # Add metrics
            metrics = detailed_results['property_metrics'][prop]
            plt.text(
                0.05, 0.95, 
                f"MAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}\nR2: {metrics['r2']:.4f}", 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            plt.xlabel(f"True {prop}")
            plt.ylabel(f"Predicted {prop}")
            plt.title(f"{prop} Predictions vs Targets")
            plt.legend()
            plt.grid(True)
            
            # Save plot
            save_path = os.path.join(save_dir, f"{self.model_name}_{prop}_predictions.png")
            plt.savefig(save_path)
            plt.close()
            
            plot_paths.append(save_path)
        
        return plot_paths
    
    def plot_error_distributions(
        self, 
        detailed_results: Dict[str, Any], 
        save_dir: str = "figures",
    ) -> List[str]:
        """
        Plot error distributions for each property.
        
        Args:
            detailed_results: Results from detailed_evaluation
            save_dir: Directory to save plots
            
        Returns:
            List of paths to saved plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        predictions = detailed_results['predictions']
        targets = detailed_results['targets']
        
        plot_paths = []
        
        for i, prop in enumerate(self.property_names):
            plt.figure(figsize=(8, 6))
            
            # Calculate errors
            errors = predictions[:, i] - targets[:, i]
            
            # Plot error distribution
            plt.hist(errors, bins=20, alpha=0.7)
            
            # Add metrics
            metrics = detailed_results['property_metrics'][prop]
            plt.axvline(x=0, color='r', linestyle='--', label="Zero Error")
            
            plt.xlabel(f"{prop} Error (Predicted - True)")
            plt.ylabel("Frequency")
            plt.title(f"{prop} Error Distribution")
            plt.grid(True)
            
            # Save plot
            save_path = os.path.join(save_dir, f"{self.model_name}_{prop}_error_dist.png")
            plt.savefig(save_path)
            plt.close()
            
            plot_paths.append(save_path)
        
        return plot_paths