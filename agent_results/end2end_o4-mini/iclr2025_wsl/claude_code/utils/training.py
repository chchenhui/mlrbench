"""
Training utilities for permutation-equivariant weight graph embeddings.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


class TrainingManager:
    """Handles training, validation, and evaluation of models."""
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device=None,
        experiment_dir=None,
        logger=None
    ):
        """
        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to use (if None, will use CUDA if available)
            experiment_dir: Directory to save results (created if doesn't exist)
            logger: Logger object for logging output
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up experiment directory
        if experiment_dir is not None:
            self.experiment_dir = experiment_dir
            os.makedirs(experiment_dir, exist_ok=True)
        else:
            self.experiment_dir = None
        
        # Set up logger
        self.logger = logger if logger is not None else self._default_logger
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
    
    def _default_logger(self, message):
        """Default logger (prints to console)."""
        print(message)
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_data in tqdm(train_loader, desc='Training', leave=False):
            # Move data to device
            batch_data = self._move_to_device(batch_data)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(batch_data)
            
            # Calculate loss
            loss = self.criterion(embeddings, batch_data['positives_mask'])
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
        
        # Return average loss
        return total_loss / max(1, batch_count)
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average loss for the validation set
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc='Validation', leave=False):
                # Move data to device
                batch_data = self._move_to_device(batch_data)
                
                # Forward pass
                embeddings = self.model(batch_data)
                
                # Calculate loss
                loss = self.criterion(embeddings, batch_data['positives_mask'])
                
                # Update statistics
                total_loss += loss.item()
                batch_count += 1
        
        # Return average loss
        return total_loss / max(1, batch_count)
    
    def train(self, train_loader, val_loader, num_epochs, save_every=None, early_stopping=None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            save_every: Save model checkpoint every N epochs (if None, only save final model)
            early_stopping: Stop training if validation loss doesn't improve for N epochs
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Track epoch time
            start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Record time
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_times'].append(epoch_time)
            
            # Log progress
            self.logger(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={epoch_time:.2f}s")
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if self.experiment_dir is not None:
                    self.save_checkpoint(os.path.join(self.experiment_dir, 'best_model.pt'))
            else:
                patience_counter += 1
            
            # Save checkpoint
            if save_every is not None and epoch % save_every == 0 and self.experiment_dir is not None:
                self.save_checkpoint(os.path.join(self.experiment_dir, f'model_epoch_{epoch}.pt'))
            
            # Early stopping
            if early_stopping is not None and patience_counter >= early_stopping:
                self.logger(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final model
        if self.experiment_dir is not None:
            self.save_checkpoint(os.path.join(self.experiment_dir, 'final_model.pt'))
        
        # Save training history
        if self.experiment_dir is not None:
            self.save_history()
        
        return self.history
    
    def save_checkpoint(self, path):
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, strict=True):
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the checkpoint
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
    
    def save_history(self, path=None):
        """
        Save training history.
        
        Args:
            path: Path to save the history (if None, uses default path in experiment_dir)
        """
        if path is None and self.experiment_dir is not None:
            path = os.path.join(self.experiment_dir, 'training_history.json')
        
        if path is not None:
            # Convert history values to Python lists
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    history_serializable[key] = [v.item() for v in value]
                else:
                    history_serializable[key] = value
            
            with open(path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
    
    def _move_to_device(self, batch_data):
        """
        Move batch data to the device.
        
        Args:
            batch_data: Batch data
            
        Returns:
            Batch data on device
        """
        if isinstance(batch_data, torch.Tensor):
            return batch_data.to(self.device)
        elif isinstance(batch_data, dict):
            return {k: self._move_to_device(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, list):
            return [self._move_to_device(x) for x in batch_data]
        else:
            return batch_data


class Evaluator:
    """Evaluates model performance on downstream tasks."""
    def __init__(self, model, device=None, experiment_dir=None, logger=None):
        """
        Args:
            model: Trained model
            device: Device to use (if None, will use CUDA if available)
            experiment_dir: Directory to save results
            logger: Logger object for logging output
        """
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up experiment directory
        self.experiment_dir = experiment_dir
        
        # Set up logger
        self.logger = logger if logger is not None else print
    
    def compute_embeddings(self, data_loader):
        """
        Compute embeddings for all models in the dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Dictionary with embeddings, model IDs, architecture indices, etc.
        """
        self.model.eval()
        
        embeddings = []
        model_ids = []
        arch_indices = []
        task_indices = []
        accuracies = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc='Computing embeddings'):
                # Move data to device
                batch_data = self._move_to_device(batch_data)
                
                # Forward pass
                batch_embeddings = self.model(batch_data)
                
                # Collect results
                embeddings.append(batch_embeddings.cpu())
                
                for metadata in batch_data['metadata']:
                    model_ids.append(metadata['model_id'])
                    arch_indices.append(metadata['arch_idx'])
                    accuracies.append(metadata['accuracy'])
                    
                    if 'task_idx' in metadata:
                        task_indices.append(metadata['task_idx'])
                    else:
                        task_indices.append(-1)  # No task ID
        
        # Concatenate embeddings
        all_embeddings = torch.cat(embeddings, dim=0)
        
        return {
            'embeddings': all_embeddings,
            'model_ids': model_ids,
            'arch_indices': torch.tensor(arch_indices),
            'task_indices': torch.tensor(task_indices),
            'accuracies': torch.tensor(accuracies)
        }
    
    def evaluate_retrieval(self, data_loader, top_ks=[1, 5, 10], by_architecture=True, by_task=True):
        """
        Evaluate model retrieval performance.
        
        Args:
            data_loader: DataLoader for the test dataset
            top_ks: List of k values to compute Recall@k
            by_architecture: Whether to evaluate retrieval by architecture
            by_task: Whether to evaluate retrieval by task
            
        Returns:
            Dictionary with retrieval metrics
        """
        # Compute embeddings
        data = self.compute_embeddings(data_loader)
        
        # Normalize embeddings for cosine similarity
        embeddings = data['embeddings']
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        # Compute all pairwise cosine similarities
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Mask out self-similarity
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # Get top-k indices
        _, indices = sim_matrix.topk(max(top_ks), dim=1)
        
        results = {}
        
        # Evaluate retrieval by architecture
        if by_architecture:
            arch_indices = data['arch_indices']
            
            for k in top_ks:
                correct = 0
                for i, idx in enumerate(arch_indices):
                    # Get retrieved architecture indices
                    retrieved = indices[i, :k]
                    retrieved_arch = arch_indices[retrieved]
                    
                    # Check if the correct architecture is retrieved
                    if idx in retrieved_arch:
                        correct += 1
                
                recall = correct / len(arch_indices)
                results[f'recall@{k}_architecture'] = recall
                self.logger(f"Recall@{k} (architecture): {recall:.4f}")
        
        # Evaluate retrieval by task
        if by_task:
            task_indices = data['task_indices']
            
            # Only consider models with task IDs
            mask = task_indices >= 0
            if mask.sum() > 0:
                # Filter data
                filtered_sim_matrix = sim_matrix[mask][:, mask]
                filtered_task_indices = task_indices[mask]
                
                # Get top-k indices
                _, filtered_indices = filtered_sim_matrix.topk(max(top_ks), dim=1)
                
                for k in top_ks:
                    correct = 0
                    for i, idx in enumerate(filtered_task_indices):
                        # Get retrieved task indices
                        retrieved = filtered_indices[i, :k]
                        retrieved_tasks = filtered_task_indices[retrieved]
                        
                        # Check if the correct task is retrieved
                        if idx in retrieved_tasks:
                            correct += 1
                    
                    recall = correct / len(filtered_task_indices)
                    results[f'recall@{k}_task'] = recall
                    self.logger(f"Recall@{k} (task): {recall:.4f}")
        
        # Compute mean reciprocal rank (MRR) for architecture retrieval
        mrr = 0.0
        for i, idx in enumerate(data['arch_indices']):
            # Get retrieved architecture indices
            retrieved_arch = data['arch_indices'][indices[i]]
            
            # Find the first occurrence of the correct architecture
            for rank, arch in enumerate(retrieved_arch):
                if arch == idx:
                    mrr += 1.0 / (rank + 1)
                    break
        
        mrr /= len(data['arch_indices'])
        results['mrr_architecture'] = mrr
        self.logger(f"MRR (architecture): {mrr:.4f}")
        
        # Save results
        if self.experiment_dir is not None:
            with open(os.path.join(self.experiment_dir, 'retrieval_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def train_accuracy_regressor(self, train_loader, val_loader=None, 
                                hidden_dims=[128, 64], lr=0.001, num_epochs=100):
        """
        Train a model to predict accuracy from embeddings.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            hidden_dims: Hidden layer dimensions for the regressor
            lr: Learning rate
            num_epochs: Number of epochs to train for
            
        Returns:
            Trained regressor model and training history
        """
        # Compute embeddings for training data
        train_data = self.compute_embeddings(train_loader)
        
        # Create regressor model
        embedding_dim = train_data['embeddings'].size(1)
        from ..models.models import AccuracyRegressor
        regressor = AccuracyRegressor(embedding_dim, hidden_dims).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(regressor.parameters(), lr=lr)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        
        # Compute validation embeddings if provided
        val_data = None
        if val_loader is not None:
            val_data = self.compute_embeddings(val_loader)
        
        # Training loop
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(1, num_epochs + 1):
            # Training
            regressor.train()
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = train_data['embeddings'].to(self.device)
            targets = train_data['accuracies'].float().to(self.device).view(-1, 1)
            outputs = regressor(embeddings)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_data is not None:
                regressor.eval()
                with torch.no_grad():
                    val_embeddings = val_data['embeddings'].to(self.device)
                    val_targets = val_data['accuracies'].float().to(self.device).view(-1, 1)
                    val_outputs = regressor(val_embeddings)
                    
                    val_loss = criterion(val_outputs, val_targets).item()
                    val_r2 = r2_score(val_targets.cpu().numpy(), val_outputs.cpu().numpy())
                    
                    history['val_loss'].append(val_loss)
                    history['val_r2'].append(val_r2)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = regressor.state_dict().copy()
                    
                    self.logger(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_r2={val_r2:.4f}")
            else:
                self.logger(f"Epoch {epoch}: train_loss={train_loss:.6f}")
        
        # Load best model if validation was used
        if val_data is not None and best_state is not None:
            regressor.load_state_dict(best_state)
        
        # Save model and history
        if self.experiment_dir is not None:
            torch.save(regressor.state_dict(), os.path.join(self.experiment_dir, 'accuracy_regressor.pt'))
            
            with open(os.path.join(self.experiment_dir, 'accuracy_regressor_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        
        return regressor, history
    
    def evaluate_accuracy_prediction(self, regressor, test_loader):
        """
        Evaluate accuracy prediction performance.
        
        Args:
            regressor: Trained regressor model
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Compute embeddings
        test_data = self.compute_embeddings(test_loader)
        
        # Predict accuracies
        regressor.eval()
        with torch.no_grad():
            embeddings = test_data['embeddings'].to(self.device)
            targets = test_data['accuracies'].float().to(self.device).view(-1, 1)
            predictions = regressor(embeddings)
        
        # Convert to numpy
        targets_np = targets.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        
        # Compute metrics
        mse = ((targets_np - predictions_np) ** 2).mean()
        r2 = r2_score(targets_np, predictions_np)
        spearman_corr, _ = spearmanr(targets_np.flatten(), predictions_np.flatten())
        
        results = {
            'mse': float(mse),
            'r2': float(r2),
            'spearman_correlation': float(spearman_corr)
        }
        
        self.logger(f"MSE: {mse:.6f}")
        self.logger(f"R²: {r2:.4f}")
        self.logger(f"Spearman correlation: {spearman_corr:.4f}")
        
        # Save results
        if self.experiment_dir is not None:
            with open(os.path.join(self.experiment_dir, 'accuracy_prediction_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Plot actual vs. predicted accuracies
            plt.figure(figsize=(8, 8))
            plt.scatter(targets_np, predictions_np, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Actual Accuracy')
            plt.ylabel('Predicted Accuracy')
            plt.title(f'Accuracy Prediction (R² = {r2:.4f})')
            plt.axis('equal')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True)
            plt.savefig(os.path.join(self.experiment_dir, 'accuracy_prediction.png'))
            plt.close()
        
        return results
    
    def train_embedding_decoder(self, train_loader, val_loader=None, 
                              lr=0.001, num_epochs=100):
        """
        Train a decoder to convert embeddings back to weights for model merging.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            lr: Learning rate
            num_epochs: Number of epochs to train for
            
        Returns:
            Trained decoder model and training history
        """
        # Compute embeddings for training data
        train_data = self.compute_embeddings(train_loader)
        
        # Create layer size information from first batch
        batch = next(iter(train_loader))
        batch = self._move_to_device(batch)
        
        # Assume all models in batch have same architecture (for simplicity)
        # In a real implementation, we would need to handle variable architectures
        model_idx = 0
        layer_sizes = []
        
        # Extract the first model's layer sizes
        num_layers = batch['num_layers'][model_idx].item()
        layer_node_features = batch['layer_node_features']
        
        for layer_idx in range(num_layers):
            # Find start and end indices for this model's nodes in this layer
            layer_batch = batch['layer_batch_indices'][layer_idx]
            model_mask = layer_batch == model_idx
            num_nodes = model_mask.sum().item()
            
            # Get input and output dimensions from node features
            node_features = batch['layer_node_features'][layer_idx][model_mask]
            in_dim = int(node_features[0, 2].item())  # Input dimension is 3rd feature
            out_dim = num_nodes  # Output dimension is number of nodes
            
            layer_sizes.append((in_dim, out_dim))
        
        # Create decoder model
        from ..models.models import EmbeddingDecoder
        embedding_dim = train_data['embeddings'].size(1)
        decoder = EmbeddingDecoder(embedding_dim, layer_sizes).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(decoder.parameters(), lr=lr)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Compute validation embeddings if provided
        val_data = None
        if val_loader is not None:
            val_data = self.compute_embeddings(val_loader)
        
        # Training loop
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(1, num_epochs + 1):
            # Training
            decoder.train()
            
            total_loss = 0.0
            batch_count = 0
            
            for batch_data in tqdm(train_loader, desc=f'Epoch {epoch} (training)', leave=False):
                batch_data = self._move_to_device(batch_data)
                
                # Forward pass through the embedding model
                embeddings = self.model(batch_data)
                
                # Decode embeddings back to weights and biases
                decoded_layers = decoder(embeddings)
                
                # Compute loss
                loss = 0.0
                model_count = 0
                
                for model_idx in range(batch_data['num_layers'].size(0)):
                    num_layers = batch_data['num_layers'][model_idx].item()
                    
                    for layer_idx in range(num_layers):
                        # Extract the real weights and biases for this model and layer
                        # (This is a simplified approach; in practice would need more careful extraction)
                        # Here we assume each model has its own graph representation in the batch
                        # You'd need to modify this to match your actual batch structure
                        
                        # For demonstration, we just compute MSE between decoded parameters and zeros
                        # as a placeholder for the real weights and biases
                        decoded_weights, decoded_biases = decoded_layers[layer_idx]
                        
                        # Placeholder loss - replace with actual weights and biases
                        weights_loss = criterion(decoded_weights[model_idx], torch.zeros_like(decoded_weights[model_idx]))
                        biases_loss = criterion(decoded_biases[model_idx], torch.zeros_like(decoded_biases[model_idx]))
                        
                        loss += weights_loss + biases_loss
                        model_count += 1
                
                if model_count > 0:
                    loss /= model_count
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            train_loss = total_loss / max(1, batch_count)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                decoder.eval()
                
                total_val_loss = 0.0
                val_batch_count = 0
                
                with torch.no_grad():
                    for batch_data in tqdm(val_loader, desc=f'Epoch {epoch} (validation)', leave=False):
                        batch_data = self._move_to_device(batch_data)
                        
                        # Forward pass
                        embeddings = self.model(batch_data)
                        decoded_layers = decoder(embeddings)
                        
                        # Compute loss (same placeholder approach as training)
                        loss = 0.0
                        model_count = 0
                        
                        for model_idx in range(batch_data['num_layers'].size(0)):
                            num_layers = batch_data['num_layers'][model_idx].item()
                            
                            for layer_idx in range(num_layers):
                                decoded_weights, decoded_biases = decoded_layers[layer_idx]
                                
                                weights_loss = criterion(decoded_weights[model_idx], torch.zeros_like(decoded_weights[model_idx]))
                                biases_loss = criterion(decoded_biases[model_idx], torch.zeros_like(decoded_biases[model_idx]))
                                
                                loss += weights_loss + biases_loss
                                model_count += 1
                        
                        if model_count > 0:
                            loss /= model_count
                        
                        total_val_loss += loss.item()
                        val_batch_count += 1
                
                val_loss = total_val_loss / max(1, val_batch_count)
                history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = decoder.state_dict().copy()
                
                self.logger(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                self.logger(f"Epoch {epoch}: train_loss={train_loss:.6f}")
        
        # Load best model if validation was used
        if val_loader is not None and best_state is not None:
            decoder.load_state_dict(best_state)
        
        # Save model and history
        if self.experiment_dir is not None:
            torch.save(decoder.state_dict(), os.path.join(self.experiment_dir, 'embedding_decoder.pt'))
            
            with open(os.path.join(self.experiment_dir, 'embedding_decoder_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        
        return decoder, history
    
    def interpolate_embeddings(self, model_A_id, model_B_id, alphas, data_loader, decoder):
        """
        Interpolate between two models in the embedding space.
        
        Args:
            model_A_id: ID of the first model
            model_B_id: ID of the second model
            alphas: List of interpolation coefficients (0.0 to 1.0)
            data_loader: DataLoader containing both models
            decoder: Trained decoder model to convert embeddings to weights
            
        Returns:
            Dictionary with interpolated models
        """
        # Compute embeddings for all models
        data = self.compute_embeddings(data_loader)
        
        # Find model indices
        model_A_idx = data['model_ids'].index(model_A_id)
        model_B_idx = data['model_ids'].index(model_B_id)
        
        # Get embeddings
        embedding_A = data['embeddings'][model_A_idx].unsqueeze(0).to(self.device)
        embedding_B = data['embeddings'][model_B_idx].unsqueeze(0).to(self.device)
        
        # Interpolate embeddings
        interpolated_embeddings = {}
        for alpha in alphas:
            interpolated = (1 - alpha) * embedding_A + alpha * embedding_B
            interpolated_embeddings[alpha] = interpolated
        
        # Decode interpolated embeddings
        decoder.eval()
        with torch.no_grad():
            interpolated_weights = {}
            for alpha, embedding in interpolated_embeddings.items():
                decoded_layers = decoder(embedding)
                interpolated_weights[alpha] = decoded_layers
        
        return interpolated_weights
    
    def _move_to_device(self, batch_data):
        """
        Move batch data to the device.
        
        Args:
            batch_data: Batch data
            
        Returns:
            Batch data on device
        """
        if isinstance(batch_data, torch.Tensor):
            return batch_data.to(self.device)
        elif isinstance(batch_data, dict):
            return {k: self._move_to_device(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, list):
            return [self._move_to_device(x) for x in batch_data]
        else:
            return batch_data