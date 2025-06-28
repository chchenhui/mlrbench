import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, Subset
import logging
from utils import Timer, plot_influence_distribution
import torch.nn.functional as F
import random
from collections import defaultdict

logger = logging.getLogger("influence_space")

class MultiModalModel(torch.nn.Module):
    """
    Simple multi-modal model for image-caption retrieval.
    """
    def __init__(
        self, 
        image_dim: int = 512, 
        text_dim: int = 512, 
        embed_dim: int = 256
    ):
        """
        Initialize multi-modal model.
        
        Args:
            image_dim: Dimension of image embeddings
            text_dim: Dimension of text embeddings
            embed_dim: Dimension of joint embedding space
        """
        super().__init__()
        
        # Image projection
        self.image_proj = torch.nn.Sequential(
            torch.nn.Linear(image_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim)
        )
        
        # Text projection
        self.text_proj = torch.nn.Sequential(
            torch.nn.Linear(text_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim)
        )
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            image_features: Image features [B, image_dim]
            text_features: Text features [B, text_dim]
            
        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        # Project image and text features to joint embedding space
        image_embeddings = self.image_proj(image_features)
        text_embeddings = self.text_proj(text_features)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        return image_embeddings, text_embeddings

def contrastive_loss(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Compute contrastive loss for image-text pairs.
    
    Args:
        image_embeddings: Image embeddings [B, D]
        text_embeddings: Text embeddings [B, D]
        temperature: Temperature parameter for scaling logits
        
    Returns:
        Contrastive loss
    """
    # Compute similarity matrix
    logits = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    
    # Labels are the diagonal indices (paired samples)
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Compute cross-entropy loss for image-to-text and text-to-image
    i2t_loss = F.cross_entropy(logits, labels)
    t2i_loss = F.cross_entropy(logits.t(), labels)
    
    # Average both directions
    loss = (i2t_loss + t2i_loss) / 2
    
    return loss

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        device: Device to run training on
        epoch: Current epoch number
        log_interval: Interval for logging training progress
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Extract data
        images = batch["image_embeddings"].to(device)
        texts = batch["text_embeddings"].to(device)
        
        # Forward pass
        image_embeddings, text_embeddings = model(images, texts)
        
        # Compute loss
        loss = contrastive_loss(image_embeddings, text_embeddings)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            logger.info(f"Epoch: {epoch}, Batch: [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")
    
    # Compute average loss
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")
    
    return avg_loss

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation data.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    total_loss = 0
    
    for batch in dataloader:
        # Extract data
        images = batch["image_embeddings"].to(device)
        texts = batch["text_embeddings"].to(device)
        
        # Forward pass
        image_embeddings, text_embeddings = model(images, texts)
        
        # Compute loss
        loss = contrastive_loss(image_embeddings, text_embeddings)
        total_loss += loss.item()
        
        # Store embeddings for recall computation
        all_image_embeddings.append(image_embeddings.cpu())
        all_text_embeddings.append(text_embeddings.cpu())
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_embeddings, all_text_embeddings.t()).numpy()
    
    # Compute recall metrics
    from utils import compute_recalls
    recall_metrics = compute_recalls(similarity_matrix, ks=[1, 5, 10])
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    recall_metrics["val_loss"] = avg_loss
    
    return recall_metrics

def compute_sample_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    batch_size: int = 16,
    max_samples: int = 500
) -> Dict[int, torch.Tensor]:
    """
    Compute gradients for individual samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing samples
        device: Device to run computation on
        batch_size: Batch size for gradient computation
        max_samples: Maximum number of samples to compute gradients for (to limit memory usage)
        
    Returns:
        Dictionary mapping sample indices to gradient vectors
    """
    model.eval()
    gradients = {}
    
    logger.info(f"Computing gradients for up to {max_samples} samples...")
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        # Break if we've processed enough samples
        if len(gradients) >= max_samples:
            break
        
        # Extract data
        images = batch["image_embeddings"].to(device)
        texts = batch["text_embeddings"].to(device)
        indices = batch["idx"].tolist()
        
        # Process samples in smaller batches to avoid OOM
        for i in range(0, len(indices), batch_size):
            # Get batch data
            batch_indices = indices[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            # Process each sample individually to get its gradient
            for j, idx in enumerate(batch_indices):
                if len(gradients) >= max_samples:
                    break
                
                # Zero gradients
                model.zero_grad()
                
                # Forward pass for a single sample
                img_emb, txt_emb = model(batch_images[j:j+1], batch_texts[j:j+1])
                
                # Compute loss
                loss = contrastive_loss(img_emb, txt_emb)
                
                # Backward pass
                loss.backward()
                
                # Extract gradients and flatten
                grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad.append(param.grad.view(-1))
                grad = torch.cat(grad)
                
                # Store gradient
                gradients[idx] = grad.detach().cpu()
    
    logger.info(f"Computed gradients for {len(gradients)} samples")
    return gradients

def compute_cluster_gradients(
    model: torch.nn.Module,
    dataloader: DataLoader,
    clusters: List[List[int]],
    device: torch.device,
    samples_per_cluster: int = 5
) -> Dict[int, torch.Tensor]:
    """
    Compute average gradients for each cluster.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing samples
        clusters: List of indices for each cluster
        device: Device to run computation on
        samples_per_cluster: Number of samples to use per cluster for gradient estimation
        
    Returns:
        Dictionary mapping cluster indices to gradient vectors
    """
    model.eval()
    cluster_gradients = {}
    
    logger.info(f"Computing gradients for {len(clusters)} clusters...")
    
    # Create a dictionary to map sample indices to their data
    sample_data = {}
    for batch in dataloader:
        indices = batch["idx"].tolist()
        images = batch["image_embeddings"]
        texts = batch["text_embeddings"]
        
        for i, idx in enumerate(indices):
            sample_data[idx] = {
                "image": images[i],
                "text": texts[i]
            }
    
    # Process each cluster
    for cluster_idx, cluster in enumerate(tqdm(clusters, desc="Computing cluster gradients")):
        # Skip empty clusters
        if len(cluster) == 0:
            continue
        
        # Sample a subset of indices for this cluster
        if len(cluster) <= samples_per_cluster:
            sampled_indices = cluster
        else:
            sampled_indices = random.sample(cluster, samples_per_cluster)
        
        # Collect batch data
        batch_images = []
        batch_texts = []
        
        for idx in sampled_indices:
            if idx in sample_data:
                batch_images.append(sample_data[idx]["image"])
                batch_texts.append(sample_data[idx]["text"])
        
        # Skip if no valid samples
        if len(batch_images) == 0:
            continue
        
        # Convert to tensors
        batch_images = torch.stack(batch_images).to(device)
        batch_texts = torch.stack(batch_texts).to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        img_emb, txt_emb = model(batch_images, batch_texts)
        
        # Compute loss
        loss = contrastive_loss(img_emb, txt_emb)
        
        # Backward pass
        loss.backward()
        
        # Extract and average gradients
        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.view(-1))
        grad = torch.cat(grad)
        
        # Store gradient for this cluster
        cluster_gradients[cluster_idx] = grad.detach().cpu() / len(batch_images)
    
    logger.info(f"Computed gradients for {len(cluster_gradients)} clusters")
    return cluster_gradients

def compute_validation_gradient(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient on validation data.
    
    Args:
        model: Trained model
        val_dataloader: DataLoader for validation data
        device: Device to run computation on
        
    Returns:
        Validation gradient vector
    """
    model.eval()
    
    # Zero gradients
    model.zero_grad()
    
    # Compute validation loss and backpropagate
    total_samples = 0
    for batch in val_dataloader:
        # Extract data
        images = batch["image_embeddings"].to(device)
        texts = batch["text_embeddings"].to(device)
        
        # Forward pass
        image_embeddings, text_embeddings = model(images, texts)
        
        # Compute loss
        loss = contrastive_loss(image_embeddings, text_embeddings)
        
        # Scale loss by batch size and accumulate gradients
        scaled_loss = loss * len(images) / len(val_dataloader.dataset)
        scaled_loss.backward()
        
        total_samples += len(images)
    
    # Extract and combine gradients
    grad = []
    for param in model.parameters():
        if param.grad is not None:
            grad.append(param.grad.view(-1))
    val_gradient = torch.cat(grad)
    
    # Normalize by number of samples
    val_gradient = val_gradient.detach().cpu()
    
    logger.info(f"Computed validation gradient from {total_samples} samples")
    return val_gradient

def stochastic_lanczos_quadrature(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 10,
    num_samples: int = 100,
    num_steps: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top eigenvectors and eigenvalues of the Hessian using stochastic Lanczos quadrature.
    
    Args:
        model: Trained model
        dataloader: DataLoader for training data
        device: Device to run computation on
        rank: Number of top eigenpairs to compute
        num_samples: Number of samples to use for Hessian-vector products
        num_steps: Number of Lanczos steps
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    logger.info(f"Computing top {rank} eigenpairs of the Hessian...")
    
    # Function to compute Hessian-vector product
    def hessian_vector_product(v):
        """Compute Hessian-vector product using samples."""
        model.zero_grad()
        
        # Get random batch
        batch = next(iter(dataloader))
        images = batch["image_embeddings"].to(device)
        texts = batch["text_embeddings"].to(device)
        
        # Forward pass
        image_embeddings, text_embeddings = model(images, texts)
        
        # Compute loss
        loss = contrastive_loss(image_embeddings, text_embeddings)
        
        # Compute gradient
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Compute product with vector v
        with torch.enable_grad():
            grad_vector = torch.cat([g.view(-1) for g in grads])
            hvp = torch.autograd.grad(
                grad_vector, model.parameters(), grad_outputs=v.to(device), retain_graph=True
            )
        
        return torch.cat([g.view(-1) for g in hvp]).detach().cpu()
    
    # Get parameter count
    param_count = sum(p.numel() for p in model.parameters())
    
    # Initialize random vector
    q = torch.randn(param_count)
    q = q / torch.norm(q)
    
    # Initialize storage for T matrix and Q matrix
    T = torch.zeros(num_steps, num_steps)
    Q = torch.zeros(param_count, num_steps)
    Q[:, 0] = q
    
    # Lanczos iteration
    for i in range(num_steps - 1):
        # Compute Hessian-vector product
        if i == 0:
            q_new = hessian_vector_product(q)
        else:
            q_new = hessian_vector_product(q) - beta * Q[:, i-1]
        
        # Update T matrix
        alpha = torch.dot(q_new, q)
        T[i, i] = alpha
        
        # Orthogonalize
        q_new = q_new - alpha * q
        for j in range(i):
            q_new = q_new - torch.dot(q_new, Q[:, j]) * Q[:, j]
        
        # Reorthogonalize
        for j in range(i + 1):
            q_new = q_new - torch.dot(q_new, Q[:, j]) * Q[:, j]
        
        # Compute norm
        beta = torch.norm(q_new)
        
        # Update T matrix
        if i < num_steps - 1:
            T[i, i+1] = beta
            T[i+1, i] = beta
        
        # Break if beta is too small
        if beta < 1e-10:
            logger.warning(f"Lanczos iteration terminated early at step {i+1} due to small beta")
            break
        
        # Update q and Q
        q = q_new / beta
        if i < num_steps - 1:
            Q[:, i+1] = q
    
    # Compute eigenvalues and eigenvectors of T
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    
    # Sort in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute eigenvectors in original space
    U = torch.matmul(Q, eigenvectors)
    
    # Normalize eigenvectors
    for i in range(U.shape[1]):
        U[:, i] = U[:, i] / torch.norm(U[:, i])
    
    # Take top rank eigenpairs
    eigenvalues = eigenvalues[:rank]
    U = U[:, :rank]
    
    logger.info(f"Computed top {rank} eigenvalues: {eigenvalues.tolist()}")
    return eigenvalues, U

def compute_influence_scores(
    cluster_gradients: Dict[int, torch.Tensor],
    val_gradient: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    lambda_reg: float = 0.01
) -> Dict[int, float]:
    """
    Compute influence scores for clusters using low-rank Hessian approximation.
    
    Args:
        cluster_gradients: Dictionary mapping cluster indices to gradient vectors
        val_gradient: Validation gradient vector
        eigenvalues: Top eigenvalues of the Hessian
        eigenvectors: Top eigenvectors of the Hessian
        lambda_reg: Regularization parameter for eigenvalues
        
    Returns:
        Dictionary mapping cluster indices to influence scores
    """
    logger.info("Computing influence scores...")
    
    # Add small regularization to eigenvalues
    eigenvalues_reg = eigenvalues + lambda_reg
    
    # Compute influence scores
    influence_scores = {}
    
    for cluster_idx, cluster_grad in cluster_gradients.items():
        # Project gradients onto eigenvectors
        val_proj = torch.matmul(val_gradient, eigenvectors)
        cluster_proj = torch.matmul(cluster_grad, eigenvectors)
        
        # Compute first term: contribution of top eigenpairs
        first_term = torch.sum(val_proj * cluster_proj / eigenvalues_reg)
        
        # Compute second term: contribution of remaining eigenpairs (approximated)
        # Assuming eigenvalues_reg[-1] is the smallest eigenvalue we computed
        val_proj_residual = val_gradient - torch.matmul(val_proj, eigenvectors.t())
        cluster_proj_residual = cluster_grad - torch.matmul(cluster_proj, eigenvectors.t())
        second_term = torch.dot(val_proj_residual, cluster_proj_residual) / eigenvalues_reg[-1]
        
        # Compute total influence score (negative because of definition)
        score = -(first_term + second_term).item()
        influence_scores[cluster_idx] = score
    
    logger.info(f"Computed influence scores for {len(influence_scores)} clusters")
    return influence_scores

def run_influence_estimation(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    clusters: List[List[int]],
    device: torch.device,
    rank: int = 10,
    samples_per_cluster: int = 5,
    output_dir: str = "./",
    visualize: bool = True
) -> Dict[int, float]:
    """
    Run the full influence estimation pipeline.
    
    Args:
        model: Trained model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        clusters: List of indices for each cluster
        device: Device to run computation on
        rank: Number of top eigenpairs to compute
        samples_per_cluster: Number of samples to use per cluster for gradient estimation
        output_dir: Directory to save results
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary mapping cluster indices to influence scores
    """
    # Compute cluster gradients
    cluster_gradients = compute_cluster_gradients(
        model, train_dataloader, clusters, device, samples_per_cluster
    )
    
    # Compute validation gradient
    val_gradient = compute_validation_gradient(model, val_dataloader, device)
    
    # Compute top eigenpairs of the Hessian
    eigenvalues, eigenvectors = stochastic_lanczos_quadrature(
        model, train_dataloader, device, rank=rank
    )
    
    # Compute influence scores
    influence_scores = compute_influence_scores(
        cluster_gradients, val_gradient, eigenvalues, eigenvectors
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "eigenvalues.npy"), eigenvalues.numpy())
    np.save(os.path.join(output_dir, "influence_scores.npy"), np.array(list(influence_scores.values())))
    
    # Save influence scores to file
    import json
    with open(os.path.join(output_dir, "influence_scores.json"), "w") as f:
        json.dump({str(k): v for k, v in influence_scores.items()}, f)
    
    # Visualize influence score distribution
    if visualize:
        scores = list(influence_scores.values())
        plot_influence_distribution(
            scores, 
            title="Influence Score Distribution",
            save_path=os.path.join(output_dir, "influence_scores.png")
        )
    
    return influence_scores

class EmbeddingDataset(Dataset):
    """
    Dataset wrapper for pre-computed embeddings.
    """
    def __init__(
        self, 
        image_embeddings: np.ndarray, 
        text_embeddings: np.ndarray, 
        indices: List[int]
    ):
        """
        Initialize dataset with pre-computed embeddings.
        
        Args:
            image_embeddings: Pre-computed image embeddings
            text_embeddings: Pre-computed text embeddings
            indices: Original indices of the samples
        """
        assert len(image_embeddings) == len(text_embeddings) == len(indices), "Incompatible embedding dimensions"
        
        self.image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.
        
        Args:
            idx: Index in dataset
            
        Returns:
            Dictionary with embeddings and metadata
        """
        return {
            "image_embeddings": self.image_embeddings[idx],
            "text_embeddings": self.text_embeddings[idx],
            "idx": self.indices[idx]
        }

def create_embedding_dataloaders(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    indices: List[int],
    val_split: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, Dataset]:
    """
    Create train and validation dataloaders from pre-computed embeddings.
    
    Args:
        image_embeddings: Pre-computed image embeddings
        text_embeddings: Pre-computed text embeddings
        indices: Original indices of the samples
        val_split: Fraction of data to use for validation
        batch_size: Batch size for data loading
        random_state: Random seed for data splitting
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, full_dataset)
    """
    # Create dataset
    dataset = EmbeddingDataset(image_embeddings, text_embeddings, indices)
    
    # Split into train and validation
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate random indices for train/val split
    perm = np.random.permutation(len(dataset))
    train_indices = perm[:train_size].tolist()
    val_indices = perm[train_size:].tolist()
    
    # Create train and validation datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, dataset