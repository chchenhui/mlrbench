"""
Embedding and clustering module for the Gradient-Informed Fingerprinting (GIF) method.

This module handles generating static embeddings for data samples and creating
clusters that serve as pseudo-labels for the probe network.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StaticEmbedder:
    """Generate static embeddings for text data using pretrained models."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-mpnet-base-v2", 
        device: str = None,
        pooling_strategy: str = "mean",
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        if "sentence-transformers" in model_name:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            self.model.to(self.device)
            self.tokenizer = None  # SentenceTransformer handles tokenization internally
        else:
            logger.info(f"Loading Transformer model: {model_name}")
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.embedding_dim = self._get_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            # For SentenceTransformer
            return self.model.get_sentence_embedding_dimension()
        else:
            # For Hugging Face Transformers
            if hasattr(self.model.config, "hidden_size"):
                return self.model.config.hidden_size
            elif hasattr(self.model.config, "dim"):
                return self.model.config.dim
            else:
                # Default fallback
                return 768
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts into embeddings."""
        if "sentence-transformers" in self.model_name:
            # SentenceTransformer already handles batching internally
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Apply pooling strategy
                if self.pooling_strategy == "cls":
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooling_strategy == "mean":
                    # Mean pooling across token dimension with attention mask
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    batch_embeddings = torch.sum(
                        outputs.last_hidden_state * attention_mask, dim=1
                    ) / torch.sum(attention_mask, dim=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_dataset(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """Encode all texts in a dataset using the embedder."""
        all_embeddings = []
        all_ids = []
        
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            texts = batch["text"]
            ids = batch["id"]
            
            batch_embeddings = self.encode_batch(texts)
            all_embeddings.append(batch_embeddings)
            all_ids.extend(ids)
        
        embeddings = np.vstack(all_embeddings)
        
        return embeddings, all_ids


class Clusterer:
    """Cluster embeddings to generate pseudo-labels for the probe network."""
    
    def __init__(
        self, 
        n_clusters: int = 100, 
        random_state: int = 42,
        algorithm: str = "kmeans"
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.algorithm = algorithm
        self.model = None
    
    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the clustering model on the embeddings."""
        logger.info(f"Fitting {self.algorithm} with {self.n_clusters} clusters on {embeddings.shape[0]} samples...")
        
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init="auto"
            )
            self.model.fit(embeddings)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
        
        logger.info("Clustering complete")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings."""
        if self.model is None:
            raise ValueError("Clusterer has not been fit yet")
        
        return self.model.predict(embeddings)
    
    def save(self, path: str) -> None:
        """Save the trained clusterer to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved clusterer to {path}")
    
    def load(self, path: str) -> None:
        """Load a trained clusterer from a file."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        if hasattr(self.model, "n_clusters"):
            self.n_clusters = self.model.n_clusters
        logger.info(f"Loaded clusterer from {path}")


class EmbeddingManager:
    """Manage the embedding and clustering process."""
    
    def __init__(
        self,
        embedder: StaticEmbedder,
        clusterer: Clusterer,
        output_dir: str = "data"
    ):
        self.embedder = embedder
        self.clusterer = clusterer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.embeddings = {}
        self.ids = {}
        self.clusters = {}
    
    def process_dataset(
        self, 
        dataloaders: Dict[str, DataLoader], 
        subsample_size: Optional[int] = None,
        fit_on: str = "train"
    ) -> Dict[str, Dict[str, Any]]:
        """Process datasets by generating embeddings and cluster assignments."""
        results = {}
        
        # Generate embeddings for all splits
        for split, dataloader in dataloaders.items():
            logger.info(f"Processing {split} split...")
            
            # Generate embeddings
            embeddings, ids = self.embedder.encode_dataset(dataloader)
            self.embeddings[split] = embeddings
            self.ids[split] = ids
            
            # Save embeddings
            self._save_embeddings(embeddings, ids, split)
            
            results[split] = {
                "embeddings_shape": embeddings.shape,
                "n_samples": len(ids)
            }
        
        # Fit clusterer on specified split (usually train)
        fit_embeddings = self.embeddings[fit_on]
        
        # Subsample for clustering if needed
        if subsample_size and subsample_size < fit_embeddings.shape[0]:
            logger.info(f"Subsampling {subsample_size} embeddings for clustering...")
            indices = np.random.choice(
                fit_embeddings.shape[0], 
                subsample_size, 
                replace=False
            )
            fit_embeddings = fit_embeddings[indices]
        
        # Fit and save clusterer
        self.clusterer.fit(fit_embeddings)
        self.clusterer.save(os.path.join(self.output_dir, "clusterer.pkl"))
        
        # Generate cluster assignments for all splits
        for split in dataloaders.keys():
            clusters = self.clusterer.predict(self.embeddings[split])
            self.clusters[split] = clusters
            
            # Save cluster assignments
            self._save_clusters(clusters, self.ids[split], split)
            
            results[split]["n_clusters"] = self.clusterer.n_clusters
            
        return results
    
    def _save_embeddings(self, embeddings: np.ndarray, ids: List[str], split: str) -> None:
        """Save embeddings and their IDs to disk."""
        emb_path = os.path.join(self.output_dir, f"{split}_embeddings.npy")
        ids_path = os.path.join(self.output_dir, f"{split}_embedding_ids.json")
        
        np.save(emb_path, embeddings)
        with open(ids_path, "w") as f:
            json.dump(ids, f)
        
        logger.info(f"Saved {split} embeddings to {emb_path} and IDs to {ids_path}")
    
    def _save_clusters(self, clusters: np.ndarray, ids: List[str], split: str) -> None:
        """Save cluster assignments and their IDs to disk."""
        clusters_path = os.path.join(self.output_dir, f"{split}_clusters.npy")
        mapping_path = os.path.join(self.output_dir, f"{split}_id_to_cluster.json")
        
        np.save(clusters_path, clusters)
        
        # Create mapping from ID to cluster
        id_to_cluster = {id_: int(cluster) for id_, cluster in zip(ids, clusters)}
        with open(mapping_path, "w") as f:
            json.dump(id_to_cluster, f)
        
        logger.info(f"Saved {split} clusters to {clusters_path} and mapping to {mapping_path}")
    
    def load_embeddings(self, split: str) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings and their IDs from disk."""
        emb_path = os.path.join(self.output_dir, f"{split}_embeddings.npy")
        ids_path = os.path.join(self.output_dir, f"{split}_embedding_ids.json")
        
        embeddings = np.load(emb_path)
        with open(ids_path, "r") as f:
            ids = json.load(f)
        
        self.embeddings[split] = embeddings
        self.ids[split] = ids
        
        logger.info(f"Loaded {split} embeddings from {emb_path}")
        return embeddings, ids
    
    def load_clusters(self, split: str) -> Tuple[np.ndarray, Dict[str, int]]:
        """Load cluster assignments and their ID-to-cluster mapping from disk."""
        clusters_path = os.path.join(self.output_dir, f"{split}_clusters.npy")
        mapping_path = os.path.join(self.output_dir, f"{split}_id_to_cluster.json")
        
        clusters = np.load(clusters_path)
        with open(mapping_path, "r") as f:
            id_to_cluster = json.load(f)
        
        self.clusters[split] = clusters
        
        logger.info(f"Loaded {split} clusters from {clusters_path}")
        return clusters, id_to_cluster


# Command-line interface for testing
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_loader import DataConfig, DataManager
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test embedding and clustering")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples")
    parser.add_argument("--embedder", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedder model")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    # Create data manager
    config = DataConfig(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        data_dir=args.output_dir
    )
    data_manager = DataManager(config)
    dataloaders = data_manager.get_dataloaders(batch_size=16)
    
    # Create embedder and clusterer
    embedder = StaticEmbedder(model_name=args.embedder)
    clusterer = Clusterer(n_clusters=args.n_clusters)
    
    # Create embedding manager
    emb_manager = EmbeddingManager(
        embedder=embedder,
        clusterer=clusterer,
        output_dir=args.output_dir
    )
    
    # Process dataset
    results = emb_manager.process_dataset(dataloaders)
    
    # Print results
    print(f"Processing results: {json.dumps(results, indent=2)}")
    
    # Test loading
    train_embeddings, train_ids = emb_manager.load_embeddings("train")
    train_clusters, id_to_cluster = emb_manager.load_clusters("train")
    
    print(f"Loaded train embeddings with shape {train_embeddings.shape}")
    print(f"First 5 cluster assignments: {train_clusters[:5]}")