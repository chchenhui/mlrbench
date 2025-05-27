"""
Contrastive Learning Framework for Weight Space Embeddings.
This module implements the contrastive learning objective with symmetry-preserving augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from tqdm import tqdm

# Local imports
from config import TRAIN_CONFIG, LOG_CONFIG
from weight_to_graph import WeightToGraph

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("contrastive_learning")

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Implementation of the NT-Xent loss from SimCLR paper, adapted for weight space.
    """
    
    def __init__(self, temperature=TRAIN_CONFIG["temperature"]):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
    
    def forward(self, z_i, z_j, negatives=None):
        """
        Compute NT-Xent loss between positive pairs and negative samples.
        
        Args:
            z_i: First embeddings [batch_size, dim]
            z_j: Second embeddings (positive pairs for z_i) [batch_size, dim]
            negatives: Optional tensor of explicit negative samples [num_negatives, dim]
            
        Returns:
            NT-Xent loss
        """
        device = z_i.device
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate positive pairs
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
        
        # Compute similarity matrix
        similarity_matrix = self.similarity_f(
            representations.unsqueeze(1),
            representations.unsqueeze(0)
        ) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Add explicit negatives if provided
        if negatives is not None:
            negatives = F.normalize(negatives, dim=1)
            neg_sim = self.similarity_f(
                representations.unsqueeze(1),
                negatives.unsqueeze(0)
            ) / self.temperature  # [2*batch_size, num_negatives]
            
            # Expand similarity matrix to include negatives
            similarity_matrix = torch.cat([similarity_matrix, neg_sim], dim=1)
        
        # Create labels: positives are the corresponding pairs
        sim_ij = torch.diag(similarity_matrix, batch_size)  # z_i to z_j
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # z_j to z_i
        
        # Create positives and negatives mask
        mask = torch.ones((2 * batch_size, 2 * batch_size + (0 if negatives is None else negatives.size(0)))).to(device)
        
        # Set diagonal to zero (self-similarity)
        mask = mask.fill_diagonal_(0)
        
        # Set positive pairs to zero in the mask
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        
        # Compute similarity excluding self-similarity and positive pairs (using the mask)
        negatives = similarity_matrix * mask
        
        # Create labels: for each sample, the positive is at a known position
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(device)
        
        # Final similarity matrix to compute CE loss
        logits = torch.cat([
            torch.cat([negatives[:batch_size], sim_ij.unsqueeze(1)], dim=1),
            torch.cat([negatives[batch_size:], sim_ji.unsqueeze(1)], dim=1)
        ], dim=0)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Normalize by number of positive pairs
        loss = loss / (2 * batch_size)
        
        return loss


class SymmetryAugmenter:
    """
    Augmentation class for applying symmetry-preserving transformations to model weights.
    """
    
    def __init__(self, 
                 permutation_prob=TRAIN_CONFIG["augmentation"]["permutation_prob"],
                 scaling_range=TRAIN_CONFIG["augmentation"]["scaling_range"],
                 dropout_prob=TRAIN_CONFIG["augmentation"]["dropout_prob"]):
        self.permutation_prob = permutation_prob
        self.scaling_range = scaling_range
        self.dropout_prob = dropout_prob
        self.converter = WeightToGraph()
        logger.info(f"Initialized SymmetryAugmenter with permutation_prob={permutation_prob}, "
                   f"scaling_range={scaling_range}, dropout_prob={dropout_prob}")
    
    def augment_model_weights(self, model_weights):
        """
        Apply symmetry-preserving augmentations to model weights.
        
        Args:
            model_weights: Dictionary of model weights.
            
        Returns:
            Augmented weights.
        """
        aug_weights = {}
        
        # Apply transformations to each weight matrix
        for key, weight in model_weights.items():
            # Skip non-tensor weights
            if not isinstance(weight, torch.Tensor):
                aug_weights[key] = weight
                continue
                
            # Skip weights that aren't 2D matrices (e.g., biases)
            if len(weight.shape) != 2:
                aug_weights[key] = weight
                continue
            
            # Apply transforms
            aug_weight = weight.clone()
            
            # 1. Permutation (with probability)
            if random.random() < self.permutation_prob:
                # Permute output dimensions
                perm = torch.randperm(weight.shape[0])
                aug_weight = aug_weight[perm]
            
            # 2. Scaling
            min_scale, max_scale = self.scaling_range
            scales = torch.rand(weight.shape[0]) * (max_scale - min_scale) + min_scale
            aug_weight = aug_weight * scales.unsqueeze(1)
            
            # 3. DropConnect (with probability)
            if random.random() < self.dropout_prob:
                mask = torch.rand_like(aug_weight) > 0.05  # Drop 5% of connections
                aug_weight = aug_weight * mask
            
            aug_weights[key] = aug_weight
        
        return aug_weights
    
    def augment_model_graphs(self, model_graphs):
        """
        Apply symmetry-preserving augmentations to model graphs.
        
        Args:
            model_graphs: List of torch_geometric.data.Data objects representing layer graphs.
            
        Returns:
            Augmented graphs.
        """
        aug_graphs = []
        
        for graph in model_graphs:
            # Apply transforms sequentially
            aug_graph = graph.clone()
            
            # 1. Permutation (with probability)
            if random.random() < self.permutation_prob:
                aug_graph = self.converter.apply_permutation_transform(aug_graph)
            
            # 2. Scaling
            aug_graph = self.converter.apply_scaling_transform(aug_graph, scale_range=self.scaling_range)
            
            # 3. DropConnect (with probability)
            if random.random() < self.dropout_prob:
                aug_graph = self.converter.apply_dropout_transform(aug_graph, dropout_prob=0.05)
            
            aug_graphs.append(aug_graph)
        
        return aug_graphs


class MetricPredictionLoss(nn.Module):
    """
    Loss for predicting performance metrics from embeddings.
    """
    
    def __init__(self):
        super(MetricPredictionLoss, self).__init__()
        
        # Simple MLP for predicting performance
        self.predictor = nn.Sequential(
            nn.Linear(MODEL_CONFIG["gnn_encoder"]["output_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # MSE loss
        self.criterion = nn.MSELoss()
    
    def forward(self, embeddings, performance):
        """
        Compute performance prediction loss.
        
        Args:
            embeddings: Model embeddings [batch_size, dim]
            performance: Ground truth performance metrics [batch_size]
            
        Returns:
            MSE loss
        """
        # Predict performance
        predicted = self.predictor(embeddings).squeeze()
        
        # Compute loss
        loss = self.criterion(predicted, performance)
        
        return loss


class ContrastiveLearningFramework:
    """
    Full contrastive learning framework for training the model embedder.
    """
    
    def __init__(self, model_embedder, 
                 lambda_contrastive=TRAIN_CONFIG["lambda_contrastive"],
                 temperature=TRAIN_CONFIG["temperature"],
                 num_negatives=TRAIN_CONFIG["num_negatives"],
                 device=None):
        self.model_embedder = model_embedder
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        self.num_negatives = num_negatives
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model_embedder.to(self.device)
        
        # Initialize loss functions
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.metric_loss = MetricPredictionLoss()
        self.metric_loss.to(self.device)
        
        # Initialize augmenter
        self.augmenter = SymmetryAugmenter()
        
        logger.info(f"Initialized ContrastiveLearningFramework with lambda_contrastive={lambda_contrastive}, "
                   f"temperature={temperature}, num_negatives={num_negatives}, device={self.device}")
    
    def train_step(self, model_graphs_batch, performance_batch, optimizer):
        """
        Perform a single training step.
        
        Args:
            model_graphs_batch: List of lists of layer graphs, one list per model.
            performance_batch: Performance metrics for each model [batch_size].
            optimizer: Optimizer to use.
            
        Returns:
            Dict with loss information.
        """
        # Set model to train mode
        self.model_embedder.train()
        self.metric_loss.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Create augmented versions of the models
        aug_model_graphs_batch = []
        for model_graphs in model_graphs_batch:
            aug_model_graphs = self.augmenter.augment_model_graphs(model_graphs)
            aug_model_graphs_batch.append(aug_model_graphs)
        
        # Forward pass
        embeddings = self.model_embedder.encode_batch(model_graphs_batch)
        aug_embeddings = self.model_embedder.encode_batch(aug_model_graphs_batch)
        
        # Sample negatives from the batch (optional)
        # We'll use a subset of other models in the batch as negatives
        batch_size = len(model_graphs_batch)
        if batch_size > 1:
            neg_indices = []
            for i in range(batch_size):
                # Sample num_negatives indices excluding i
                candidates = list(range(batch_size))
                candidates.remove(i)
                if len(candidates) > self.num_negatives:
                    neg_indices.append(random.sample(candidates, self.num_negatives))
                else:
                    # If not enough candidates, sample with replacement
                    neg_indices.append(random.choices(candidates, k=self.num_negatives))
            
            # Get negative embeddings for each model
            negative_embeddings = []
            for i, indices in enumerate(neg_indices):
                neg_embeds = embeddings[indices]
                negative_embeddings.append(neg_embeds)
            
            # Stack into tensor [batch_size, num_negatives, dim]
            negative_embeddings = torch.stack(negative_embeddings, dim=0)
        else:
            # Not enough samples for negatives
            negative_embeddings = None
        
        # Compute contrastive loss
        if negative_embeddings is not None:
            contrastive_loss = 0
            for i in range(batch_size):
                z_i = embeddings[i:i+1]  # [1, dim]
                z_j = aug_embeddings[i:i+1]  # [1, dim]
                neg = negative_embeddings[i]  # [num_negatives, dim]
                
                contrastive_loss += self.contrastive_loss(z_i, z_j, neg)
            
            contrastive_loss /= batch_size
        else:
            # No negatives - just use in-batch contrastive
            contrastive_loss = self.contrastive_loss(embeddings, aug_embeddings)
        
        # Compute metric prediction loss
        metric_loss = self.metric_loss(embeddings, performance_batch)
        
        # Compute combined loss
        combined_loss = self.lambda_contrastive * contrastive_loss + (1 - self.lambda_contrastive) * metric_loss
        
        # Backward pass
        combined_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        return {
            "loss": combined_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "metric_loss": metric_loss.item()
        }
    
    def eval_step(self, model_graphs_batch, performance_batch):
        """
        Perform a single evaluation step.
        
        Args:
            model_graphs_batch: List of lists of layer graphs, one list per model.
            performance_batch: Performance metrics for each model [batch_size].
            
        Returns:
            Dict with loss information.
        """
        # Set model to eval mode
        self.model_embedder.eval()
        self.metric_loss.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            # Create augmented versions of the models
            aug_model_graphs_batch = []
            for model_graphs in model_graphs_batch:
                aug_model_graphs = self.augmenter.augment_model_graphs(model_graphs)
                aug_model_graphs_batch.append(aug_model_graphs)
            
            # Forward pass
            embeddings = self.model_embedder.encode_batch(model_graphs_batch)
            aug_embeddings = self.model_embedder.encode_batch(aug_model_graphs_batch)
            
            # Sample negatives from the batch (optional)
            batch_size = len(model_graphs_batch)
            if batch_size > 1:
                neg_indices = []
                for i in range(batch_size):
                    # Sample num_negatives indices excluding i
                    candidates = list(range(batch_size))
                    candidates.remove(i)
                    if len(candidates) > self.num_negatives:
                        neg_indices.append(random.sample(candidates, self.num_negatives))
                    else:
                        # If not enough candidates, sample with replacement
                        neg_indices.append(random.choices(candidates, k=self.num_negatives))
                
                # Get negative embeddings for each model
                negative_embeddings = []
                for i, indices in enumerate(neg_indices):
                    neg_embeds = embeddings[indices]
                    negative_embeddings.append(neg_embeds)
                
                # Stack into tensor [batch_size, num_negatives, dim]
                negative_embeddings = torch.stack(negative_embeddings, dim=0)
            else:
                # Not enough samples for negatives
                negative_embeddings = None
            
            # Compute contrastive loss
            if negative_embeddings is not None:
                contrastive_loss = 0
                for i in range(batch_size):
                    z_i = embeddings[i:i+1]  # [1, dim]
                    z_j = aug_embeddings[i:i+1]  # [1, dim]
                    neg = negative_embeddings[i]  # [num_negatives, dim]
                    
                    contrastive_loss += self.contrastive_loss(z_i, z_j, neg)
                
                contrastive_loss /= batch_size
            else:
                # No negatives - just use in-batch contrastive
                contrastive_loss = self.contrastive_loss(embeddings, aug_embeddings)
            
            # Compute metric prediction loss
            metric_loss = self.metric_loss(embeddings, performance_batch)
            
            # Compute combined loss
            combined_loss = self.lambda_contrastive * contrastive_loss + (1 - self.lambda_contrastive) * metric_loss
            
            return {
                "loss": combined_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
                "metric_loss": metric_loss.item(),
                "embeddings": embeddings.cpu().numpy()
            }
    
    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer to use.
            
        Returns:
            Dict with average loss information.
        """
        epoch_losses = {
            "loss": 0.0,
            "contrastive_loss": 0.0,
            "metric_loss": 0.0
        }
        
        # Iterate over batches
        for batch_idx, (model_graphs_batch, performance_batch) in enumerate(tqdm(train_loader, desc="Training")):
            # Move performance to device
            performance_batch = performance_batch.to(self.device)
            
            # Train step
            step_losses = self.train_step(model_graphs_batch, performance_batch, optimizer)
            
            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += step_losses[key]
        
        # Compute averages
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def evaluate(self, val_loader):
        """
        Evaluate on validation set.
        
        Args:
            val_loader: DataLoader for validation data.
            
        Returns:
            Dict with average loss information and embeddings.
        """
        epoch_losses = {
            "loss": 0.0,
            "contrastive_loss": 0.0,
            "metric_loss": 0.0
        }
        
        all_embeddings = []
        all_performances = []
        all_model_ids = []
        
        # Iterate over batches
        for batch_idx, (model_graphs_batch, performance_batch, model_ids) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Move performance to device
            performance_batch = performance_batch.to(self.device)
            
            # Eval step
            step_results = self.eval_step(model_graphs_batch, performance_batch)
            
            # Update epoch losses
            for key in epoch_losses:
                epoch_losses[key] += step_results[key]
            
            # Store embeddings and info
            all_embeddings.append(step_results["embeddings"])
            all_performances.append(performance_batch.cpu().numpy())
            all_model_ids.extend(model_ids)
        
        # Compute averages
        num_batches = len(val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Concatenate embeddings and performances
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_performances = np.concatenate(all_performances, axis=0)
        
        return {
            **epoch_losses,
            "embeddings": all_embeddings,
            "performances": all_performances,
            "model_ids": all_model_ids
        }
    
    def train(self, train_loader, val_loader, num_epochs, optimizer, scheduler=None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of epochs to train for.
            optimizer: Optimizer to use.
            scheduler: Optional learning rate scheduler.
            
        Returns:
            Dict with training history.
        """
        history = {
            "train_loss": [],
            "train_contrastive_loss": [],
            "train_metric_loss": [],
            "val_loss": [],
            "val_contrastive_loss": [],
            "val_metric_loss": []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader, optimizer)
            
            # Update history
            history["train_loss"].append(train_losses["loss"])
            history["train_contrastive_loss"].append(train_losses["contrastive_loss"])
            history["train_metric_loss"].append(train_losses["metric_loss"])
            
            # Evaluate
            val_results = self.evaluate(val_loader)
            
            # Update history
            history["val_loss"].append(val_results["loss"])
            history["val_contrastive_loss"].append(val_results["contrastive_loss"])
            history["val_metric_loss"].append(val_results["metric_loss"])
            
            # Update scheduler if provided
            if scheduler is not None:
                scheduler.step(val_results["loss"])
            
            # Log epoch results
            logger.info(f"Train Loss: {train_losses['loss']:.4f}, "
                       f"Contrastive: {train_losses['contrastive_loss']:.4f}, "
                       f"Metric: {train_losses['metric_loss']:.4f}")
            logger.info(f"Val Loss: {val_results['loss']:.4f}, "
                       f"Contrastive: {val_results['contrastive_loss']:.4f}, "
                       f"Metric: {val_results['metric_loss']:.4f}")
            
            # Check for improvement
            if val_results["loss"] < best_val_loss:
                best_val_loss = val_results["loss"]
                best_model_state = {
                    "model_embedder": self.model_embedder.state_dict(),
                    "metric_loss": self.metric_loss.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_results["loss"]
                }
                best_epoch = epoch
                patience_counter = 0
                logger.info(f"New best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model_embedder.load_state_dict(best_model_state["model_embedder"])
            self.metric_loss.load_state_dict(best_model_state["metric_loss"])
            logger.info(f"Restored best model from epoch {best_epoch+1}")
        
        return history, best_model_state
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model to.
        """
        torch.save({
            "model_embedder": self.model_embedder.state_dict(),
            "metric_loss": self.metric_loss.state_dict(),
            "lambda_contrastive": self.lambda_contrastive,
            "temperature": self.temperature,
            "num_negatives": self.num_negatives
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_embedder.load_state_dict(checkpoint["model_embedder"])
        self.metric_loss.load_state_dict(checkpoint["metric_loss"])
        self.lambda_contrastive = checkpoint.get("lambda_contrastive", self.lambda_contrastive)
        self.temperature = checkpoint.get("temperature", self.temperature)
        self.num_negatives = checkpoint.get("num_negatives", self.num_negatives)
        
        logger.info(f"Model loaded from {filepath}")


# Test code
if __name__ == "__main__":
    # Import additional modules for testing
    from gnn_encoder import ModelEmbedder
    import torch
    from torch_geometric.data import Data
    
    # Create a simple model embedder for testing
    model_embedder = ModelEmbedder()
    
    # Create contrastive learning framework
    framework = ContrastiveLearningFramework(model_embedder)
    
    # Create some synthetic graph data for testing
    def create_test_graph(num_nodes=10, dim=MODEL_CONFIG["gnn_encoder"]["node_dim"], 
                         edge_dim=MODEL_CONFIG["gnn_encoder"]["edge_dim"]):
        # Create random node features
        x = torch.randn(num_nodes, dim)
        
        # Create random edges (ensuring all nodes are connected)
        edge_index = []
        for i in range(num_nodes):
            # Each node connects to at least one other node
            targets = random.sample(range(num_nodes), max(1, num_nodes // 3))
            for t in targets:
                if t != i:  # Avoid self-loops
                    edge_index.append([i, t])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create random edge features
        edge_attr = torch.randn(edge_index.size(1), edge_dim)
        
        # Create graph
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.n_in = num_nodes // 2
        graph.n_out = num_nodes - graph.n_in
        graph.layer_name = "test_layer"
        graph.total_nodes = num_nodes
        
        return graph
    
    # Create test batch
    num_models = 4
    num_layers_per_model = 3
    batch_size = 2
    
    test_models = []
    for _ in range(num_models):
        model_graphs = []
        for _ in range(num_layers_per_model):
            graph = create_test_graph()
            model_graphs.append(graph)
        test_models.append(model_graphs)
    
    # Create mock performance values
    performances = torch.rand(num_models)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model_embedder.parameters()) + list(framework.metric_loss.parameters()),
        lr=0.001
    )
    
    # Create mock train loader
    class MockLoader:
        def __init__(self, models, performances, batch_size):
            self.models = models
            self.performances = performances
            self.batch_size = batch_size
            self.num_samples = len(models)
        
        def __iter__(self):
            indices = list(range(self.num_samples))
            random.shuffle(indices)
            
            for i in range(0, self.num_samples, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_models = [self.models[j] for j in batch_indices]
                batch_perfs = self.performances[batch_indices]
                batch_ids = [f"model_{j}" for j in batch_indices]
                
                yield batch_models, batch_perfs, batch_ids
        
        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    # Create mock loaders
    train_loader = MockLoader(test_models[:3], performances[:3], batch_size)
    val_loader = MockLoader(test_models[3:], performances[3:], batch_size)
    
    # Test single train step
    print("Testing single train step...")
    batch_models, batch_perfs = next(iter(train_loader))[:2]
    step_losses = framework.train_step(batch_models, batch_perfs, optimizer)
    print(f"Step losses: {step_losses}")
    
    # Test single eval step
    print("\nTesting single eval step...")
    batch_models, batch_perfs = next(iter(val_loader))[:2]
    step_results = framework.eval_step(batch_models, batch_perfs)
    print(f"Step results: loss={step_results['loss']}, embeddings shape={step_results['embeddings'].shape}")
    
    # Test full training loop
    print("\nTesting full training loop (1 epoch)...")
    history, best_model = framework.train(train_loader, val_loader, num_epochs=1, optimizer=optimizer)
    print(f"Training history: {history}")
    print(f"Best model: epoch={best_model['epoch']}, val_loss={best_model['val_loss']}")
    
    # Test save and load
    framework.save_model("test_model.pt")
    framework.load_model("test_model.pt")
    print("\nModel saved and loaded successfully.")