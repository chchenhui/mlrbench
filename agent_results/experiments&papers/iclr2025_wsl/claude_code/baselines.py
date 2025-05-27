"""
Baseline Methods for Model Zoo Retrieval.
This module implements baseline methods for comparison with the permutation-equivariant approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import logging

# Local imports
from config import MODEL_CONFIG, LOG_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("baselines")

class FlatVectorizer:
    """
    Flattens model weights into vectors for use with PCA or other flat methods.
    """
    
    def __init__(self, max_params=1000000):
        self.max_params = max_params
        logger.info(f"Initialized FlatVectorizer with max_params={max_params}")
    
    def vectorize_weights(self, model_weights):
        """
        Convert model weights to a flat vector.
        
        Args:
            model_weights: Dictionary of model weights.
            
        Returns:
            Flattened weight vector.
        """
        # Extract all weight tensors
        weight_tensors = []
        
        for key, weight in model_weights.items():
            # Skip non-tensor weights
            if not isinstance(weight, torch.Tensor):
                continue
            
            # Flatten the tensor
            flat_weight = weight.view(-1).cpu().numpy()
            weight_tensors.append(flat_weight)
        
        # Concatenate all flattened tensors
        if not weight_tensors:
            logger.warning("No weight tensors found!")
            return np.zeros(1)
        
        flat_vector = np.concatenate(weight_tensors)
        
        # Cap the size if necessary
        if len(flat_vector) > self.max_params:
            logger.info(f"Truncating vector from {len(flat_vector)} to {self.max_params} parameters")
            flat_vector = flat_vector[:self.max_params]
        
        # Pad if too small (for consistency)
        if len(flat_vector) < self.max_params:
            logger.info(f"Padding vector from {len(flat_vector)} to {self.max_params} parameters")
            padding = np.zeros(self.max_params - len(flat_vector))
            flat_vector = np.concatenate([flat_vector, padding])
        
        return flat_vector
    
    def vectorize_batch(self, model_weights_batch):
        """
        Vectorize a batch of model weights.
        
        Args:
            model_weights_batch: List of model weight dictionaries.
            
        Returns:
            Array of flattened weight vectors.
        """
        vectors = []
        
        for model_weights in model_weights_batch:
            vector = self.vectorize_weights(model_weights)
            vectors.append(vector)
        
        return np.stack(vectors)


class PCAEncoder:
    """
    PCA-based encoder for model weights.
    """
    
    def __init__(self, n_components=MODEL_CONFIG["pca_encoder"]["n_components"]):
        self.n_components = n_components
        self.vectorizer = FlatVectorizer()
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        logger.info(f"Initialized PCAEncoder with n_components={n_components}")
    
    def fit(self, model_weights_batch):
        """
        Fit the PCA model on a batch of model weights.
        
        Args:
            model_weights_batch: List of model weight dictionaries.
        """
        # Vectorize the weights
        vectors = self.vectorizer.vectorize_batch(model_weights_batch)
        
        # Fit PCA
        self.pca.fit(vectors)
        self.fitted = True
        
        # Log explained variance
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA fitted with {self.n_components} components, "
                   f"explaining {explained_variance:.4f} of variance")
    
    def encode(self, model_weights):
        """
        Encode a model's weights using PCA.
        
        Args:
            model_weights: Dictionary of model weights.
            
        Returns:
            PCA embedding vector.
        """
        if not self.fitted:
            logger.warning("PCA not fitted yet! Call fit() first.")
            return np.zeros(self.n_components)
        
        # Vectorize the weights
        vector = self.vectorizer.vectorize_weights(model_weights)
        vector = vector.reshape(1, -1)
        
        # Apply PCA
        embedding = self.pca.transform(vector)[0]
        
        return embedding
    
    def encode_batch(self, model_weights_batch):
        """
        Encode a batch of models.
        
        Args:
            model_weights_batch: List of model weight dictionaries.
            
        Returns:
            Batch of PCA embeddings.
        """
        if not self.fitted:
            logger.warning("PCA not fitted yet! Call fit() first.")
            return np.zeros((len(model_weights_batch), self.n_components))
        
        # Vectorize the weights
        vectors = self.vectorizer.vectorize_batch(model_weights_batch)
        
        # Apply PCA
        embeddings = self.pca.transform(vectors)
        
        return embeddings


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for model weights (non-equivariant baseline).
    """
    
    def __init__(self, 
                 hidden_dim=MODEL_CONFIG["transformer_encoder"]["hidden_dim"],
                 num_layers=MODEL_CONFIG["transformer_encoder"]["num_layers"],
                 num_heads=MODEL_CONFIG["transformer_encoder"]["num_heads"],
                 dropout=MODEL_CONFIG["transformer_encoder"]["dropout"],
                 output_dim=MODEL_CONFIG["transformer_encoder"]["output_dim"]):
        super(TransformerEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection (to project flattened weight matrices)
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.info(f"Initialized TransformerEncoder with hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, num_heads={num_heads}")
    
    def forward(self, model_weights):
        """
        Encode a model's weights using the transformer.
        
        Args:
            model_weights: Dictionary of model weights.
            
        Returns:
            Transformer embedding vector.
        """
        # Extract and process weight matrices
        weight_matrices = []
        
        for key, weight in model_weights.items():
            # Skip non-tensor weights
            if not isinstance(weight, torch.Tensor):
                continue
            
            # Flatten the tensor
            flat_weight = weight.view(-1).unsqueeze(1)  # [num_params, 1]
            weight_matrices.append(flat_weight)
        
        # Concatenate all matrices (if any)
        if not weight_matrices:
            logger.warning("No weight tensors found!")
            return torch.zeros(self.output_dim, device=next(self.parameters()).device)
        
        # Concatenate along the first dimension
        # This creates a "sequence" of weights
        weights_seq = torch.cat(weight_matrices, dim=0)  # [total_params, 1]
        
        # Truncate or pad sequence if needed (for efficiency)
        max_seq_len = 50000  # Limit for very large models
        if weights_seq.size(0) > max_seq_len:
            weights_seq = weights_seq[:max_seq_len]
        
        # Project to hidden dimension
        weights_seq = self.input_proj(weights_seq)  # [seq_len, hidden_dim]
        
        # Add batch dimension
        weights_seq = weights_seq.unsqueeze(0)  # [1, seq_len, hidden_dim]
        
        # Apply transformer
        transformer_out = self.transformer(weights_seq)  # [1, seq_len, hidden_dim]
        
        # Pool sequence (mean pooling)
        pooled = transformer_out.mean(dim=1)  # [1, hidden_dim]
        
        # Project to output dimension
        output = self.output_proj(pooled)  # [1, output_dim]
        
        # Apply layer norm
        output = self.layer_norm(output)
        
        return output.squeeze(0)  # [output_dim]
    
    def encode_batch(self, model_weights_batch):
        """
        Encode a batch of models.
        
        Args:
            model_weights_batch: List of model weight dictionaries.
            
        Returns:
            Batch of transformer embeddings.
        """
        embeddings = []
        
        for model_weights in model_weights_batch:
            embedding = self.forward(model_weights)
            embeddings.append(embedding)
        
        return torch.stack(embeddings, dim=0)


class HypernetworkEncoder(nn.Module):
    """
    Hypernetwork-based encoder for model weights (supervised baseline).
    """
    
    def __init__(self, 
                 hidden_dim=MODEL_CONFIG["hypernetwork"]["hidden_dim"],
                 output_dim=MODEL_CONFIG["hypernetwork"]["output_dim"],
                 num_layers=MODEL_CONFIG["hypernetwork"]["num_layers"],
                 dropout=MODEL_CONFIG["hypernetwork"]["dropout"]):
        super(HypernetworkEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Vectorizer for flat representation
        self.vectorizer = FlatVectorizer()
        
        # MLP for processing vectorized weights
        layers = []
        input_dim = self.vectorizer.max_params
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Performance prediction head
        self.performance_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized HypernetworkEncoder with hidden_dim={hidden_dim}, "
                   f"output_dim={output_dim}, num_layers={num_layers}")
    
    def forward(self, model_weights):
        """
        Encode a model's weights using the hypernetwork.
        
        Args:
            model_weights: Dictionary of model weights.
            
        Returns:
            Embedding vector and predicted performance.
        """
        # Vectorize the weights
        vector = self.vectorizer.vectorize_weights(model_weights)
        vector = torch.tensor(vector, dtype=torch.float, device=next(self.parameters()).device)
        
        # Apply MLP
        embedding = self.mlp(vector)
        
        # Predict performance
        performance = self.performance_head(embedding)
        
        return embedding, performance
    
    def encode_batch(self, model_weights_batch):
        """
        Encode a batch of models.
        
        Args:
            model_weights_batch: List of model weight dictionaries.
            
        Returns:
            Batch of embeddings and predicted performances.
        """
        embeddings = []
        performances = []
        
        for model_weights in model_weights_batch:
            embedding, performance = self.forward(model_weights)
            embeddings.append(embedding)
            performances.append(performance)
        
        return torch.stack(embeddings, dim=0), torch.cat(performances, dim=0)
    
    def train_step(self, model_weights_batch, true_performances, optimizer):
        """
        Perform a single training step.
        
        Args:
            model_weights_batch: List of model weight dictionaries.
            true_performances: Ground truth performance values.
            optimizer: Optimizer to use.
            
        Returns:
            Dictionary with loss information.
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        embeddings, pred_performances = self.encode_batch(model_weights_batch)
        
        # Compute loss (MSE for performance prediction)
        loss = F.mse_loss(pred_performances, true_performances)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        return {"loss": loss.item()}


# Test code
if __name__ == "__main__":
    # Test flat vectorizer
    print("Testing FlatVectorizer...")
    vectorizer = FlatVectorizer(max_params=1000)
    
    # Create a dummy model weights dict
    dummy_weights = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(3, 10),
        "layer2.bias": torch.randn(3)
    }
    
    flat_vector = vectorizer.vectorize_weights(dummy_weights)
    print(f"Flat vector shape: {flat_vector.shape}")
    
    # Test PCA encoder
    print("\nTesting PCAEncoder...")
    pca_encoder = PCAEncoder(n_components=10)
    
    # Create a batch of dummy models
    dummy_batch = [dummy_weights for _ in range(5)]
    
    # Fit and encode
    pca_encoder.fit(dummy_batch)
    pca_embedding = pca_encoder.encode(dummy_weights)
    print(f"PCA embedding shape: {pca_embedding.shape}")
    
    # Test transformer encoder
    print("\nTesting TransformerEncoder...")
    transformer_encoder = TransformerEncoder(
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        output_dim=10
    )
    
    transformer_embedding = transformer_encoder.forward(dummy_weights)
    print(f"Transformer embedding shape: {transformer_embedding.shape}")
    
    # Test hypernetwork encoder
    print("\nTesting HypernetworkEncoder...")
    hypernetwork_encoder = HypernetworkEncoder(
        hidden_dim=32,
        output_dim=10,
        num_layers=2
    )
    
    embedding, performance = hypernetwork_encoder.forward(dummy_weights)
    print(f"Hypernetwork embedding shape: {embedding.shape}")
    print(f"Predicted performance: {performance.item():.4f}")
    
    # Test batch encoding
    print("\nTesting batch encoding...")
    batch_embeddings, batch_performances = hypernetwork_encoder.encode_batch(dummy_batch)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    print(f"Batch performances shape: {batch_performances.shape}")
    
    # Test training step
    true_performances = torch.rand(len(dummy_batch))
    optimizer = torch.optim.Adam(hypernetwork_encoder.parameters(), lr=0.001)
    loss_info = hypernetwork_encoder.train_step(dummy_batch, true_performances, optimizer)
    print(f"Training loss: {loss_info['loss']:.4f}")