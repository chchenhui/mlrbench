"""
Probe network implementation for the Gradient-Informed Fingerprinting (GIF) method.

This module defines the MLP probe network used to generate gradient-based signatures
for training data samples, which are then combined with static embeddings to form
fingerprints for efficient nearest-neighbor search.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProbeNetwork(nn.Module):
    """Multi-layer perceptron probe network."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Define activation function
        if activation.lower() == "relu":
            self.activation = F.relu
        elif activation.lower() == "gelu":
            self.activation = F.gelu
        elif activation.lower() == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the probe network."""
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer (no activation or dropout for output)
        x = self.output_layer(x)
        
        return x


class GradientExtractor:
    """Extract gradient signatures from trained probe network."""
    
    def __init__(
        self,
        probe: ProbeNetwork,
        projection_dim: int = 128,
        random_seed: int = 42,
        device: str = None
    ):
        self.probe = probe
        self.projection_dim = projection_dim
        self.random_seed = random_seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Put probe on device
        self.probe.to(self.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Create random projection matrix
        param_size = self._count_parameters()
        logger.info(f"Parameter count: {param_size}")
        
        self.projection_matrix = self._create_projection_matrix(param_size, projection_dim)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _count_parameters(self) -> int:
        """Count the total number of parameters in the probe network."""
        return sum(p.numel() for p in self.probe.parameters())
    
    def _create_projection_matrix(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """Create a random projection matrix for dimensionality reduction."""
        # Create a random matrix
        projection = torch.randn(output_dim, input_dim, device=self.device)
        
        # Normalize rows to maintain distances
        row_norms = torch.norm(projection, dim=1, keepdim=True)
        projection = projection / row_norms
        
        logger.info(f"Created projection matrix with shape: {projection.shape}")
        
        return projection
    
    def _flatten_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """Flatten gradients into a single vector."""
        return torch.cat([g.detach().view(-1) for g in gradients])
    
    def compute_gradient_signature(
        self, 
        embedding: torch.Tensor, 
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient signature for a single embedding."""
        # Ensure model is in eval mode for consistent gradients
        self.probe.eval()
        
        # Move data to device
        embedding = embedding.to(self.device)
        label = label.to(self.device)
        
        # Forward pass
        self.probe.zero_grad()
        output = self.probe(embedding)
        loss = self.criterion(output.unsqueeze(0), label.unsqueeze(0))
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        gradients = [p.grad.clone() for p in self.probe.parameters() if p.grad is not None]
        flattened_gradients = self._flatten_gradients(gradients)
        
        # Apply projection
        projected_gradients = torch.matmul(self.projection_matrix, flattened_gradients)
        
        return flattened_gradients, projected_gradients
    
    def compute_batch_gradient_signatures(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient signatures for a batch of embeddings."""
        # Initialize tensors to store results
        batch_size = embeddings.shape[0]
        param_size = self._count_parameters()
        
        flattened_grads = torch.zeros((batch_size, param_size), device=self.device)
        projected_grads = torch.zeros((batch_size, self.projection_dim), device=self.device)
        
        # Compute gradient for each sample
        for i in range(batch_size):
            flattened_grads[i], projected_grads[i] = self.compute_gradient_signature(
                embeddings[i], labels[i]
            )
        
        return flattened_grads, projected_grads
    
    def save_projection_matrix(self, path: str) -> None:
        """Save the random projection matrix to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.projection_matrix, path)
        logger.info(f"Saved projection matrix to {path}")
    
    def load_projection_matrix(self, path: str) -> None:
        """Load the random projection matrix from disk."""
        self.projection_matrix = torch.load(path, map_location=self.device)
        logger.info(f"Loaded projection matrix from {path}")


class ProbeTrainer:
    """Train the probe network for fingerprint generation."""
    
    def __init__(
        self,
        probe: ProbeNetwork,
        device: str = None,
        optimizer_cls: Any = optim.AdamW,
        optimizer_kwargs: Dict[str, Any] = None,
        scheduler_cls: Optional[Any] = None,
        scheduler_kwargs: Dict[str, Any] = None
    ):
        self.probe = probe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.probe.to(self.device)
        
        # Set up optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-3, 'weight_decay': 1e-5}
        
        self.optimizer = optimizer_cls(self.probe.parameters(), **optimizer_kwargs)
        
        # Set up learning rate scheduler
        self.scheduler = None
        if scheduler_cls is not None:
            if scheduler_kwargs is None:
                scheduler_kwargs = {}
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train the probe network for one epoch."""
        self.probe.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Train on batches
        for batch in tqdm(dataloader, desc="Training"):
            # Get data
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.probe(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the probe network."""
        self.probe.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Get data
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                # Forward pass
                outputs = self.probe(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item() * inputs.size(0)
                
                # Compute accuracy
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = total_loss / len(dataloader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        model_save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the probe network."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.validate(val_dataloader)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for early stopping
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if model_save_path is not None:
                        self.save_probe(model_save_path)
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Save final model if no early stopping or no best model was saved
        if model_save_path is not None and (early_stopping_patience is None or patience_counter < early_stopping_patience):
            self.save_probe(model_save_path)
        
        return self.history
    
    def save_probe(self, path: str) -> None:
        """Save the trained probe network to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.probe.state_dict(), path)
        logger.info(f"Saved probe model to {path}")
    
    def load_probe(self, path: str) -> None:
        """Load a trained probe network from disk."""
        self.probe.load_state_dict(torch.load(path, map_location=self.device))
        self.probe.to(self.device)
        logger.info(f"Loaded probe model from {path}")


class FingerprintGenerator:
    """Generate fingerprints for data samples by combining static embeddings with gradient signatures."""
    
    def __init__(
        self,
        probe: ProbeNetwork,
        projection_dim: int = 128,
        device: str = None,
        fingerprint_type: str = "combined"  # "static", "gradient", or "combined"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fingerprint_type = fingerprint_type
        
        # Create gradient extractor
        self.gradient_extractor = GradientExtractor(
            probe=probe,
            projection_dim=projection_dim,
            device=self.device
        )
    
    def create_fingerprints(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """Create fingerprints for a set of embeddings and their cluster labels."""
        num_samples = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        projection_dim = self.gradient_extractor.projection_dim
        
        # Initialize tensor to store fingerprints
        if self.fingerprint_type == "combined":
            fingerprints = np.zeros((num_samples, embedding_dim + projection_dim))
        elif self.fingerprint_type == "static":
            fingerprints = np.zeros((num_samples, embedding_dim))
        else:  # gradient
            fingerprints = np.zeros((num_samples, projection_dim))
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, num_samples)
            batch_embeddings = torch.tensor(embeddings[i:end_idx], dtype=torch.float32)
            batch_labels = torch.tensor(labels[i:end_idx], dtype=torch.long)
            
            # Compute gradient signatures
            _, batch_projected_grads = self.gradient_extractor.compute_batch_gradient_signatures(
                batch_embeddings, batch_labels
            )
            
            # Combine with static embeddings based on fingerprint type
            if self.fingerprint_type == "combined":
                # Make sure both tensors are on the same device
                batch_embeddings_cpu = batch_embeddings.cpu()
                batch_projected_grads_cpu = batch_projected_grads.cpu()
                batch_fingerprints = torch.cat(
                    [batch_embeddings_cpu, batch_projected_grads_cpu], dim=1
                ).numpy()
            elif self.fingerprint_type == "static":
                batch_fingerprints = batch_embeddings.cpu().numpy()
            else:  # gradient
                batch_fingerprints = batch_projected_grads.cpu().numpy()
            
            # Store fingerprints
            fingerprints[i:end_idx] = batch_fingerprints
        
        return fingerprints
    
    def save_projection_matrix(self, path: str) -> None:
        """Save the projection matrix used for gradient dimensionality reduction."""
        self.gradient_extractor.save_projection_matrix(path)
    
    def load_projection_matrix(self, path: str) -> None:
        """Load a projection matrix for gradient dimensionality reduction."""
        self.gradient_extractor.load_projection_matrix(path)


# Command-line interface for testing
if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib.pyplot as plt
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description="Test probe network training and fingerprint generation")
    parser.add_argument("--embedding_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters (output dimension)")
    parser.add_argument("--projection_dim", type=int, default=128, help="Gradient projection dimension")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    
    args = parser.parse_args()
    
    # Create synthetic data for testing
    num_samples = 1000
    embedding_dim = args.embedding_dim
    n_clusters = args.n_clusters
    
    # Create random embeddings and labels
    embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    labels = np.random.randint(0, n_clusters, size=num_samples).astype(np.int64)
    
    # Split into train and validation sets
    train_size = int(0.8 * num_samples)
    train_embeddings = embeddings[:train_size]
    train_labels = labels[:train_size]
    val_embeddings = embeddings[train_size:]
    val_labels = labels[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_embeddings, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create probe network
    probe = ProbeNetwork(
        input_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=n_clusters,
        num_layers=2,
        dropout=0.1
    )
    
    # Create trainer
    trainer = ProbeTrainer(probe)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train probe network
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        model_save_path=os.path.join(args.output_dir, "probe_model.pt")
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "probe_training.png"))
    
    # Test fingerprint generation
    fingerprint_generator = FingerprintGenerator(
        probe=probe,
        projection_dim=args.projection_dim,
        fingerprint_type="combined"
    )
    
    # Save projection matrix
    fingerprint_generator.save_projection_matrix(
        os.path.join(args.output_dir, "projection_matrix.pt")
    )
    
    # Generate fingerprints for validation data
    fingerprints = fingerprint_generator.create_fingerprints(
        embeddings=val_embeddings,
        labels=val_labels,
        batch_size=args.batch_size
    )
    
    # Print fingerprint statistics
    print(f"Generated {fingerprints.shape[0]} fingerprints with dimension {fingerprints.shape[1]}")
    print(f"Fingerprint statistics:")
    print(f"Mean: {np.mean(fingerprints)}")
    print(f"Std: {np.std(fingerprints)}")
    print(f"Min: {np.min(fingerprints)}")
    print(f"Max: {np.max(fingerprints)}")
    
    # Create visualizations of fingerprints
    plt.figure(figsize=(10, 6))
    
    # Plot heatmap of a few fingerprints
    num_to_plot = min(10, fingerprints.shape[0])
    plt.imshow(fingerprints[:num_to_plot], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Dimension')
    plt.ylabel('Sample')
    plt.title(f'Fingerprint Heatmap (First {num_to_plot} Samples)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fingerprint_visualization.png"))