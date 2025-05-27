"""
Baseline attribution methods for comparison with the Gradient-Informed Fingerprinting (GIF) method.

This module implements simplified versions of baseline attribution methods,
including TRACE, TRAK, and vanilla influence functions.
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TRACEMethod:
    """
    Implementation of TRACE: TRansformer-based Attribution using Contrastive Embeddings.
    
    Based on "TRACE: TRansformer-based Attribution using Contrastive Embeddings in LLMs"
    (Wang et al., 2024). This is a simplified version for comparison purposes.
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        temperature: float = 0.1,
        index_type: str = "hnsw",
        contrastive_margin: float = 0.5
    ):
        self.encoder_name = encoder_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.index_type = index_type
        self.contrastive_margin = contrastive_margin
        
        # We'll use SentenceTransformer for encoding texts
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(encoder_name)
            self.encoder.to(self.device)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install -U sentence-transformers")
        
        # Dimension of the embeddings
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Linear projection layer for contrastive learning
        self.projector = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.projector.to(self.device)
        
        # Index for similarity search
        self.index = None
        self.id_to_index = {}
        self.index_to_id = []
    
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index for similarity search."""
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.embedding_dim, 16)  # 16 connections per layer
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def _encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts and apply the contrastive projection."""
        # Use the encoder to get embeddings
        base_embeddings = self.encoder.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # Apply the projection (if model is trained)
        with torch.no_grad():
            if self.projector.training:
                self.projector.eval()
            
            # Project in batches to avoid OOM
            projected_embeddings = []
            for i in range(0, len(base_embeddings), batch_size):
                batch = base_embeddings[i:i + batch_size].to(self.device)
                projection = self.projector(batch)
                projected_embeddings.append(projection.cpu().numpy())
            
            # Concatenate batches
            return np.vstack(projected_embeddings)
    
    def _contrastive_loss(
        self, 
        anchors: torch.Tensor, 
        positives: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet contrastive loss with hard negative mining.
        
        Args:
            anchors: Anchor embeddings
            positives: Positive embeddings (same class/source as anchors)
            negatives: Negative embeddings (different class/source from anchors)
        
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)
        
        # Compute similarity scores
        pos_sim = torch.sum(anchors * positives, dim=1)
        
        # Reshape for broadcasting
        anchors_reshaped = anchors.unsqueeze(1)  # [B, 1, D]
        negatives_reshaped = negatives.unsqueeze(0)  # [1, N, D]
        
        # Compute similarities with all negatives
        neg_sim = torch.sum(anchors_reshaped * negatives_reshaped, dim=2)  # [B, N]
        
        # Find hardest negative for each anchor
        hardest_neg_sim, _ = torch.max(neg_sim, dim=1)
        
        # Compute triplet loss with margin
        loss = torch.clamp(hardest_neg_sim - pos_sim + self.contrastive_margin, min=0.0)
        
        return loss.mean()
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        batch_size: int = 32,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_negatives: int = 5,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the contrastive projection model.
        
        Args:
            train_texts: List of training text samples
            train_labels: List of training labels (cluster IDs or class labels)
            val_texts: Optional list of validation text samples
            val_labels: Optional list of validation labels
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_negatives: Number of negative samples per anchor
            output_dir: Optional directory to save model checkpoints
        
        Returns:
            Dictionary of training history (losses)
        """
        # Convert labels to numpy for easier manipulation
        train_labels = np.array(train_labels)
        
        # Set up training
        self.projector.train()
        optimizer = optim.AdamW(
            self.projector.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create output directory if specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [] if val_texts is not None else None
        }
        
        # Encode all training texts (initial encoding without projection)
        logger.info("Encoding training texts...")
        base_train_embeddings = self.encoder.encode(
            train_texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # Encode validation texts if available
        if val_texts is not None and val_labels is not None:
            logger.info("Encoding validation texts...")
            val_labels = np.array(val_labels)
            base_val_embeddings = self.encoder.encode(
                val_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_tensor=True
            )
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.projector.train()
            epoch_losses = []
            
            # Create batches of similar and dissimilar examples
            num_samples = len(train_texts)
            indices = np.random.permutation(num_samples)
            
            for start_idx in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get anchor embeddings
                anchor_embeddings = base_train_embeddings[batch_indices].to(self.device)
                anchor_labels = train_labels[batch_indices]
                
                # Get positive embeddings (same label as anchor)
                positive_embeddings = []
                for i, label in enumerate(anchor_labels):
                    # Find all samples with the same label (excluding the anchor itself)
                    positive_indices = np.where(train_labels == label)[0]
                    positive_indices = positive_indices[positive_indices != batch_indices[i]]
                    
                    if len(positive_indices) > 0:
                        # Randomly select one positive
                        pos_idx = np.random.choice(positive_indices)
                        positive_embeddings.append(base_train_embeddings[pos_idx])
                    else:
                        # If no positives available, use the anchor itself
                        positive_embeddings.append(base_train_embeddings[batch_indices[i]])
                
                positive_embeddings = torch.stack(positive_embeddings).to(self.device)
                
                # Get negative embeddings (different label from anchor)
                negative_embeddings = []
                for i, label in enumerate(anchor_labels):
                    # Find all samples with different labels
                    negative_indices = np.where(train_labels != label)[0]
                    
                    if len(negative_indices) > 0:
                        # Randomly select negatives
                        neg_indices = np.random.choice(
                            negative_indices, 
                            size=min(num_negatives, len(negative_indices)),
                            replace=False
                        )
                        batch_negatives = [base_train_embeddings[idx] for idx in neg_indices]
                        negative_embeddings.extend(batch_negatives)
                    else:
                        # If no negatives available, use random samples
                        random_indices = np.random.choice(
                            num_samples, 
                            size=num_negatives,
                            replace=False
                        )
                        batch_negatives = [base_train_embeddings[idx] for idx in random_indices]
                        negative_embeddings.extend(batch_negatives)
                
                negative_embeddings = torch.stack(negative_embeddings).to(self.device)
                
                # Project embeddings
                projected_anchors = self.projector(anchor_embeddings)
                projected_positives = self.projector(positive_embeddings)
                projected_negatives = self.projector(negative_embeddings)
                
                # Compute contrastive loss
                loss = self._contrastive_loss(
                    projected_anchors, 
                    projected_positives, 
                    projected_negatives
                )
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Compute average loss for the epoch
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validate if validation data is available
            if val_texts is not None and val_labels is not None:
                self.projector.eval()
                val_losses = []
                
                with torch.no_grad():
                    # Process validation data in batches
                    val_batch_size = min(batch_size, len(val_texts))
                    for start_idx in range(0, len(val_texts), val_batch_size):
                        # Get batch
                        end_idx = min(start_idx + val_batch_size, len(val_texts))
                        batch_val_embeddings = base_val_embeddings[start_idx:end_idx].to(self.device)
                        batch_val_labels = val_labels[start_idx:end_idx]
                        
                        # Project embeddings
                        projected_val = self.projector(batch_val_embeddings)
                        
                        # Select positives and negatives
                        # (Simplified for validation - we'll use the first example in batch as anchor)
                        anchor = projected_val[0:1]
                        anchor_label = batch_val_labels[0]
                        
                        # Find positives
                        positive_mask = batch_val_labels == anchor_label
                        positive_mask[0] = False  # Exclude the anchor itself
                        
                        if np.any(positive_mask):
                            positives = projected_val[positive_mask]
                            
                            # Find negatives
                            negative_mask = batch_val_labels != anchor_label
                            
                            if np.any(negative_mask):
                                negatives = projected_val[negative_mask]
                                
                                # Compute loss
                                loss = self._contrastive_loss(anchor, positives, negatives)
                                val_losses.append(loss.item())
                
                if val_losses:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    history["val_loss"].append(avg_val_loss)
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Save checkpoint if output directory is specified
            if output_dir is not None:
                checkpoint_path = os.path.join(output_dir, f"trace_projector_epoch{epoch+1}.pt")
                torch.save(self.projector.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return history
    
    def build_index(self, texts: List[str], ids: List[str], batch_size: int = 32) -> None:
        """
        Build a search index from text samples.
        
        Args:
            texts: List of text samples
            ids: List of sample IDs
            batch_size: Batch size for encoding
        """
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self._encode_batch(texts, batch_size)
        
        # Create index
        logger.info(f"Building {self.index_type} index...")
        self.index = self._create_index()
        
        # Map IDs to indices
        self.id_to_index = {id_: i for i, id_ in enumerate(ids)}
        self.index_to_id = ids.copy()
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Index built with {len(texts)} samples")
    
    def search(self, query_texts: List[str], k: int = 10, batch_size: int = 32) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for the most similar texts to the queries.
        
        Args:
            query_texts: List of query texts
            k: Number of results to return for each query
            batch_size: Batch size for encoding
        
        Returns:
            Tuple of (list of lists of IDs, list of lists of distances)
        """
        if self.index is None:
            raise ValueError("Index has not been built yet. Call build_index first.")
        
        # Generate query embeddings
        logger.info(f"Generating embeddings for {len(query_texts)} queries...")
        query_embeddings = self._encode_batch(query_texts, batch_size)
        
        # Search index
        logger.info(f"Searching for top {k} matches...")
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        # Convert indices to IDs
        results = []
        for query_indices in indices:
            # Filter out invalid indices
            valid_indices = [idx for idx in query_indices if idx >= 0 and idx < len(self.index_to_id)]
            
            # Map to IDs
            query_results = [self.index_to_id[idx] for idx in valid_indices]
            
            # Pad with None if necessary
            query_results.extend([None] * (k - len(query_results)))
            
            results.append(query_results)
        
        return results, distances.tolist()
    
    def save(self, output_dir: str) -> None:
        """
        Save the TRACE model and index.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save projector
        projector_path = os.path.join(output_dir, "trace_projector.pt")
        torch.save(self.projector.state_dict(), projector_path)
        
        # Save index
        if self.index is not None:
            index_path = os.path.join(output_dir, "trace_index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save ID mappings
            mappings = {
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id
            }
            mappings_path = os.path.join(output_dir, "trace_id_mappings.pkl")
            with open(mappings_path, "wb") as f:
                pickle.dump(mappings, f)
        
        logger.info(f"Saved TRACE model and index to {output_dir}")
    
    def load(self, input_dir: str) -> None:
        """
        Load the TRACE model and index.
        
        Args:
            input_dir: Input directory
        """
        # Load projector
        projector_path = os.path.join(input_dir, "trace_projector.pt")
        self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))
        
        # Load index if it exists
        index_path = os.path.join(input_dir, "trace_index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
            # Load ID mappings
            mappings_path = os.path.join(input_dir, "trace_id_mappings.pkl")
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
            
            self.id_to_index = mappings["id_to_index"]
            self.index_to_id = mappings["index_to_id"]
        
        logger.info(f"Loaded TRACE model and index from {input_dir}")


class TRAKMethod:
    """
    Implementation of TRAK: Attributing Model Behavior at Scale.
    
    Based on "TRAK: Attributing Model Behavior at Scale" (Park et al., 2023).
    This is a simplified version for comparison purposes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        projection_dim: int = 128,
        num_examples: int = 1000
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_dim = projection_dim
        self.num_examples = num_examples
        
        # Move model to device
        self.model.to(self.device)
        
        # Random projection matrices for gradients
        self.projection_matrices = None
        
        # Trackers for training examples and their gradients
        self.train_gradients = None
        self.train_ids = None
    
    def _create_projection_matrices(self, model_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Create random projection matrices for each parameter tensor.
        
        Args:
            model_params: List of model parameter tensors
        
        Returns:
            List of projection matrices
        """
        projection_matrices = []
        
        for param in model_params:
            # Create a random projection matrix for this parameter
            param_size = param.numel()
            projection = torch.randn(
                self.projection_dim, 
                param_size, 
                device=self.device
            )
            
            # Normalize rows for stable projection
            row_norms = torch.norm(projection, dim=1, keepdim=True)
            projection = projection / row_norms
            
            projection_matrices.append(projection)
        
        return projection_matrices
    
    def _compute_gradient(self, inputs: torch.Tensor, targets: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute gradients of the model with respect to inputs.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
        
        Returns:
            List of gradient tensors
        """
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone().detach())
        
        return gradients
    
    def _project_gradients(
        self, 
        gradients: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Project gradients using random projection matrices.
        
        Args:
            gradients: List of gradient tensors
        
        Returns:
            Projected gradients tensor
        """
        projected_gradients = []
        
        for grad, proj in zip(gradients, self.projection_matrices):
            # Flatten gradient
            flat_grad = grad.view(-1)
            
            # Project gradient
            projected = torch.matmul(proj, flat_grad)
            projected_gradients.append(projected)
        
        # Sum projections across parameters
        return torch.stack(projected_gradients).sum(dim=0)
    
    def fit(self, train_dataloader: DataLoader, ids: List[str]) -> None:
        """
        Fit the TRAK method on training data.
        
        Args:
            train_dataloader: DataLoader for training data
            ids: List of training example IDs
        """
        if len(ids) != len(train_dataloader.dataset):
            raise ValueError("Number of IDs must match number of training examples")
        
        # Create projection matrices
        params = list(self.model.parameters())
        self.projection_matrices = self._create_projection_matrices(params)
        
        # Compute and project gradients for training examples
        logger.info(f"Computing gradients for {len(ids)} training examples...")
        
        # Select a subset of examples if specified
        if self.num_examples < len(ids):
            indices = np.random.choice(len(ids), self.num_examples, replace=False)
            selected_ids = [ids[i] for i in indices]
        else:
            indices = list(range(len(ids)))
            selected_ids = ids
        
        # Process examples in batches
        batch_size = train_dataloader.batch_size
        all_gradients = []
        processed_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                # Get inputs and targets
                if isinstance(batch, tuple) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch["input"]
                    targets = batch["target"]
                
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Process each example in the batch
                for i in range(inputs.size(0)):
                    # Check if this example should be included
                    example_idx = batch_idx * batch_size + i
                    if example_idx in indices:
                        # Compute gradient for this example
                        gradients = self._compute_gradient(
                            inputs[i:i+1], 
                            targets[i:i+1]
                        )
                        
                        # Project gradients
                        projected = self._project_gradients(gradients)
                        
                        all_gradients.append(projected)
                        processed_ids.append(ids[example_idx])
        
        # Stack all projected gradients
        self.train_gradients = torch.stack(all_gradients)
        self.train_ids = processed_ids
        
        logger.info(f"Computed gradients for {len(self.train_ids)} training examples")
    
    def attribute(
        self, 
        test_dataloader: DataLoader, 
        k: int = 10
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Attribute test examples to training examples.
        
        Args:
            test_dataloader: DataLoader for test data
            k: Number of top attributions to return
        
        Returns:
            List of (attribution IDs, attribution scores) tuples
        """
        if self.train_gradients is None:
            raise ValueError("TRAK has not been fit yet. Call fit first.")
        
        # Process test examples in batches
        logger.info(f"Computing attributions for test examples...")
        
        attributions = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # Get inputs and targets
                if isinstance(batch, tuple) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch["input"]
                    targets = batch["target"]
                
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Process each example in the batch
                for i in range(inputs.size(0)):
                    # Compute gradient for this example
                    gradients = self._compute_gradient(
                        inputs[i:i+1], 
                        targets[i:i+1]
                    )
                    
                    # Project gradients
                    projected = self._project_gradients(gradients)
                    
                    # Compute similarity with training gradients
                    similarities = torch.matmul(self.train_gradients, projected)
                    
                    # Get top k attributions
                    top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(self.train_ids)))
                    
                    # Get corresponding IDs and scores
                    top_ids = [self.train_ids[idx] for idx in top_k_indices.cpu().numpy()]
                    top_scores = top_k_values.cpu().numpy().tolist()
                    
                    attributions.append((top_ids, top_scores))
        
        return attributions
    
    def save(self, output_dir: str) -> None:
        """
        Save the TRAK model.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save projection matrices and train gradients
        state = {
            "projection_matrices": [p.cpu() for p in self.projection_matrices],
            "train_gradients": self.train_gradients.cpu(),
            "train_ids": self.train_ids,
            "projection_dim": self.projection_dim,
            "num_examples": self.num_examples
        }
        
        state_path = os.path.join(output_dir, "trak_state.pt")
        torch.save(state, state_path)
        
        logger.info(f"Saved TRAK state to {output_dir}")
    
    def load(self, input_dir: str) -> None:
        """
        Load the TRAK model.
        
        Args:
            input_dir: Input directory
        """
        state_path = os.path.join(input_dir, "trak_state.pt")
        state = torch.load(state_path, map_location=self.device)
        
        self.projection_matrices = [p.to(self.device) for p in state["projection_matrices"]]
        self.train_gradients = state["train_gradients"].to(self.device)
        self.train_ids = state["train_ids"]
        self.projection_dim = state["projection_dim"]
        self.num_examples = state["num_examples"]
        
        logger.info(f"Loaded TRAK state from {input_dir}")


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Test baseline attribution methods")
    parser.add_argument("--method", type=str, default="trace", choices=["trace", "trak"],
                        help="Baseline method to test")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for testing")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic data for testing
    if args.method == "trace":
        # Test TRACE
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate synthetic texts
        train_texts = []
        train_labels = []
        
        for i in range(args.num_samples):
            # Create texts with different topics based on label
            label = i % 5  # 5 different classes
            if label == 0:
                text = f"Sample {i}: This is about technology and computers."
            elif label == 1:
                text = f"Sample {i}: This example discusses nature and animals."
            elif label == 2:
                text = f"Sample {i}: Here we talk about politics and government."
            elif label == 3:
                text = f"Sample {i}: This is related to sports and athletes."
            else:
                text = f"Sample {i}: The topic here is food and cooking."
            
            train_texts.append(text)
            train_labels.append(label)
        
        # Create validation data
        val_texts = []
        val_labels = []
        
        for i in range(args.num_samples // 5):
            # Create texts with different topics based on label
            label = i % 5  # 5 different classes
            if label == 0:
                text = f"Validation {i}: A discussion about modern technology."
            elif label == 1:
                text = f"Validation {i}: Nature conservation and wildlife."
            elif label == 2:
                text = f"Validation {i}: Politics in the 21st century."
            elif label == 3:
                text = f"Validation {i}: Professional sports competitions."
            else:
                text = f"Validation {i}: Recipes and cooking techniques."
            
            val_texts.append(text)
            val_labels.append(label)
        
        # Create TRACE method
        trace = TRACEMethod(encoder_name="sentence-transformers/all-mpnet-base-v2")
        
        # Train for a few epochs
        history = trace.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            batch_size=8,
            num_epochs=2,
            output_dir=args.output_dir
        )
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train")
        if history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("TRACE Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "trace_training.png"))
        
        # Build index
        train_ids = [f"train_{i}" for i in range(len(train_texts))]
        trace.build_index(train_texts, train_ids)
        
        # Test search
        query_texts = [
            "A new computer technology was released today.",
            "The animals in the national park are thriving.",
            "The government announced a new policy yesterday."
        ]
        
        results, distances = trace.search(query_texts, k=5)
        
        # Print results
        for i, (query, result_ids, result_dists) in enumerate(zip(query_texts, results, distances)):
            print(f"\nQuery {i+1}: {query}")
            print("Results:")
            for j, (id_, dist) in enumerate(zip(result_ids, result_dists)):
                if id_ is not None:
                    label = train_labels[int(id_.split("_")[1])]
                    print(f"  {j+1}. ID: {id_}, Label: {label}, Distance: {dist:.4f}")
        
        # Save model
        trace.save(args.output_dir)
    
    elif args.method == "trak":
        # Test TRAK
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(args.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 classes
        )
        
        # Generate synthetic data
        train_embeddings = torch.randn(args.num_samples, args.embedding_dim)
        train_labels = torch.randint(0, 5, (args.num_samples,))
        train_ids = [f"train_{i}" for i in range(args.num_samples)]
        
        test_embeddings = torch.randn(args.num_samples // 5, args.embedding_dim)
        test_labels = torch.randint(0, 5, (args.num_samples // 5,))
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_embeddings, train_labels)
        test_dataset = TensorDataset(test_embeddings, test_labels)
        
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Create TRAK method
        trak = TRAKMethod(
            model=model,
            projection_dim=64,
            num_examples=args.num_samples // 2  # Use half the training examples
        )
        
        # Fit TRAK
        trak.fit(train_dataloader, train_ids)
        
        # Compute attributions
        attributions = trak.attribute(test_dataloader, k=5)
        
        # Print results for a few examples
        for i in range(min(5, len(attributions))):
            top_ids, top_scores = attributions[i]
            print(f"\nTest Example {i+1}:")
            print("Top Attributions:")
            for j, (id_, score) in enumerate(zip(top_ids, top_scores)):
                print(f"  {j+1}. ID: {id_}, Score: {score:.4f}")
        
        # Save model
        trak.save(args.output_dir)