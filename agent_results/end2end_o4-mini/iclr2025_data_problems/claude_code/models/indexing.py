"""
ANN indexing module for the Gradient-Informed Fingerprinting (GIF) method.

This module implements approximate nearest neighbor (ANN) indexing for efficient
similarity search in high-dimensional fingerprint space using FAISS.
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import faiss
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ANNIndex:
    """Approximate Nearest Neighbor (ANN) index for efficient similarity search."""
    
    def __init__(
        self,
        index_type: str = "hnsw",
        dimension: Optional[int] = None,
        metric: str = "l2",
        use_gpu: bool = False,
        index_params: Optional[Dict[str, Any]] = None
    ):
        self.index_type = index_type
        self.dimension = dimension
        self.metric = metric
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.index_params = index_params or {}
        
        # Will be set when index is built or loaded
        self.index = None
        self.id_to_index = {}  # Maps external IDs to internal indices
        self.index_to_id = []  # Maps internal indices to external IDs
        
        # Stats for tracking build and search performance
        self.stats = {
            "build_time": None,
            "index_size": None,
            "num_vectors": 0,
            "search_times": []
        }
    
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index based on the specified parameters."""
        if self.dimension is None:
            raise ValueError("Dimension must be specified before creating index")
        
        if self.metric == "l2":
            metric_type = faiss.METRIC_L2
        elif self.metric == "inner_product":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "cosine":
            # For cosine similarity, we'll normalize vectors and use inner product
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        if self.index_type == "flat":
            # Exact search, no approximation
            index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World graph
            m = self.index_params.get("M", 16)  # Connections per layer
            ef_construction = self.index_params.get("efConstruction", 200)  # Search width during construction
            
            index = faiss.IndexHNSWFlat(self.dimension, m, metric_type)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = self.index_params.get("efSearch", 128)  # Search width during search
        
        elif self.index_type == "ivf":
            # Inverted File index
            nlist = self.index_params.get("nlist", 100)  # Number of Voronoi cells
            
            # First build a flat index to serve as quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric_type)
            
            # IVF indices must be trained before use
            index.train_called = False
        
        elif self.index_type == "pq":
            # Product Quantization for memory-efficient storage
            nlist = self.index_params.get("nlist", 100)  # Number of Voronoi cells
            m = self.index_params.get("m", 8)  # Number of subquantizers
            nbits = self.index_params.get("nbits", 8)  # Number of bits per quantizer
            
            # First build a flat index to serve as quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, metric_type)
            
            # IVF indices must be trained before use
            index.train_called = False
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # For cosine similarity, we'll use a preprocessing step to normalize vectors
        if self.metric == "cosine":
            index = faiss.IndexIDMap(index)
        
        # Move to GPU if requested and available
        if self.use_gpu:
            device = self.index_params.get("gpu_device", 0)
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, device, index)
                logger.info(f"Index moved to GPU device {device}")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
                self.use_gpu = False
        
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity search."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return vectors / norms
    
    def build(
        self, 
        vectors: np.ndarray, 
        ids: Optional[List[str]] = None, 
        batch_size: int = 10000
    ) -> None:
        """Build the ANN index with the provided vectors."""
        # Start timing
        start_time = time.time()
        
        # Set the dimension based on input if not already set
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        elif self.dimension != vectors.shape[1]:
            raise ValueError(f"Expected dimension {self.dimension}, got {vectors.shape[1]}")
        
        num_vectors = vectors.shape[0]
        
        # Create the index
        self.index = self._create_index()
        
        # Generate sequential IDs if not provided
        if ids is None:
            ids = [str(i) for i in range(num_vectors)]
        elif len(ids) != num_vectors:
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({num_vectors})")
        
        # Create ID mappings
        self.id_to_index = {id_: i for i, id_ in enumerate(ids)}
        self.index_to_id = ids.copy()
        self.stats["num_vectors"] = num_vectors
        
        # For IVF-based indices, we need to train before adding vectors
        if hasattr(self.index, 'train_called') and not self.index.train_called:
            logger.info(f"Training index on {num_vectors} vectors...")
            
            # If training data is too large, subsample
            train_size = min(num_vectors, self.index_params.get("train_size", 100000))
            if train_size < num_vectors:
                indices = np.random.choice(num_vectors, train_size, replace=False)
                train_vectors = vectors[indices]
            else:
                train_vectors = vectors
            
            # Normalize for cosine similarity if needed
            if self.metric == "cosine":
                train_vectors = self._normalize_vectors(train_vectors)
            
            self.index.train(train_vectors)
            self.index.train_called = True
        
        # Add vectors to the index in batches
        logger.info(f"Adding {num_vectors} vectors to the index in batches of {batch_size}...")
        
        for i in tqdm(range(0, num_vectors, batch_size)):
            batch_end = min(i + batch_size, num_vectors)
            batch_vectors = vectors[i:batch_end].astype(np.float32)
            
            # Normalize for cosine similarity if needed
            if self.metric == "cosine":
                batch_vectors = self._normalize_vectors(batch_vectors)
            
            if isinstance(self.index, faiss.IndexIDMap):
                # For indices that support custom IDs
                batch_ids = np.arange(i, batch_end, dtype=np.int64)
                self.index.add_with_ids(batch_vectors, batch_ids)
            else:
                # For regular indices (sequential IDs)
                self.index.add(batch_vectors)
        
        # Record stats
        self.stats["build_time"] = time.time() - start_time
        
        # Estimate index size in memory (bytes)
        if hasattr(faiss, "get_index_memory_usage"):
            try:
                self.stats["index_size"] = faiss.get_index_memory_usage(self.index)
            except:
                # Rough approximation if function fails
                self.stats["index_size"] = vectors.nbytes * self.index_params.get("size_ratio", 1.5)
        else:
            # Rough approximation
            self.stats["index_size"] = vectors.nbytes * self.index_params.get("size_ratio", 1.5)
        
        logger.info(f"Index built in {self.stats['build_time']:.2f} seconds")
        logger.info(f"Estimated index size: {self.stats['index_size'] / (1024**2):.2f} MB")
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int = 10, 
        return_distances: bool = True
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[List[float]]]]:
        """Search the index for the nearest neighbors of the query vectors."""
        if self.index is None:
            raise ValueError("Index has not been built or loaded yet")
        
        # Start timing
        start_time = time.time()
        
        # Prepare query vectors
        query_vectors = query_vectors.astype(np.float32)
        
        # Normalize for cosine similarity if needed
        if self.metric == "cosine":
            query_vectors = self._normalize_vectors(query_vectors)
        
        # Perform the search
        if isinstance(self.index, faiss.IndexIDMap):
            # For indices that support custom IDs
            distances, indices = self.index.search(query_vectors, k)
        else:
            # For regular indices (sequential IDs)
            distances, indices = self.index.search(query_vectors, k)
        
        # Map internal indices to external IDs
        results = []
        for query_indices in indices:
            # Filter out invalid indices (-1 is FAISS sentinel for "not found")
            valid_indices = [idx for idx in query_indices if idx >= 0 and idx < len(self.index_to_id)]
            
            # Map to external IDs
            query_results = [self.index_to_id[idx] for idx in valid_indices]
            
            # Pad with None if we didn't find enough results
            query_results.extend([None] * (k - len(query_results)))
            
            results.append(query_results)
        
        # Record search time
        search_time = time.time() - start_time
        self.stats["search_times"].append(search_time)
        
        # Return results
        if return_distances:
            return results, distances.tolist()
        else:
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index and search performance."""
        stats = self.stats.copy()
        
        # Add additional info
        if self.stats["search_times"]:
            stats["avg_search_time"] = sum(self.stats["search_times"]) / len(self.stats["search_times"])
            stats["min_search_time"] = min(self.stats["search_times"])
            stats["max_search_time"] = max(self.stats["search_times"])
        
        # Add index info
        stats["index_type"] = self.index_type
        stats["dimension"] = self.dimension
        stats["metric"] = self.metric
        stats["index_params"] = self.index_params
        
        return stats
    
    def save(self, directory: str, prefix: str = "ann_index") -> None:
        """Save the ANN index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(directory, f"{prefix}.index")
        
        # For GPU index, we need to move it back to CPU first
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "metric": self.metric,
            "index_params": self.index_params,
            "stats": self.stats,
            "num_vectors": self.stats["num_vectors"]
        }
        
        metadata_path = os.path.join(directory, f"{prefix}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save ID mappings
        id_mappings = {
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id
        }
        
        mappings_path = os.path.join(directory, f"{prefix}_id_mappings.pkl")
        with open(mappings_path, "wb") as f:
            pickle.dump(id_mappings, f)
        
        logger.info(f"Saved ANN index to {directory}/{prefix}.*")
    
    def load(self, directory: str, prefix: str = "ann_index") -> None:
        """Load the ANN index and metadata from disk."""
        # Load metadata
        metadata_path = os.path.join(directory, f"{prefix}_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Set properties from metadata
        self.index_type = metadata["index_type"]
        self.dimension = metadata["dimension"]
        self.metric = metadata["metric"]
        self.index_params = metadata["index_params"]
        self.stats = metadata["stats"]
        
        # Load index
        index_path = os.path.join(directory, f"{prefix}.index")
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if self.use_gpu:
            device = self.index_params.get("gpu_device", 0)
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, device, self.index)
                logger.info(f"Index moved to GPU device {device}")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
                self.use_gpu = False
        
        # Load ID mappings
        mappings_path = os.path.join(directory, f"{prefix}_id_mappings.pkl")
        with open(mappings_path, "rb") as f:
            id_mappings = pickle.load(f)
        
        self.id_to_index = id_mappings["id_to_index"]
        self.index_to_id = id_mappings["index_to_id"]
        
        logger.info(f"Loaded ANN index from {directory}/{prefix}.* with {self.stats['num_vectors']} vectors")


# Command-line interface for testing
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Test ANN indexing")
    parser.add_argument("--index_type", type=str, default="hnsw", choices=["flat", "hnsw", "ivf", "pq"],
                        help="Index type")
    parser.add_argument("--dimension", type=int, default=256, help="Vector dimension")
    parser.add_argument("--num_vectors", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "inner_product", "cosine"],
                        help="Distance metric")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create random vectors for testing
    np.random.seed(42)
    vectors = np.random.randn(args.num_vectors, args.dimension).astype(np.float32)
    ids = [f"sample_{i}" for i in range(args.num_vectors)]
    
    # Create ANN index
    index_params = {
        "M": 16,                # HNSW: connections per layer
        "efConstruction": 200,  # HNSW: search width during construction
        "efSearch": 128,        # HNSW: search width during search
        "nlist": 100,           # IVF: number of Voronoi cells
        "m": 8,                 # PQ: number of subquantizers
        "nbits": 8              # PQ: bits per quantizer
    }
    
    ann_index = ANNIndex(
        index_type=args.index_type,
        dimension=args.dimension,
        metric=args.metric,
        use_gpu=args.use_gpu,
        index_params=index_params
    )
    
    # Build the index
    logger.info(f"Building {args.index_type} index with {args.num_vectors} vectors of dimension {args.dimension}...")
    ann_index.build(vectors, ids)
    
    # Save the index
    ann_index.save(args.output_dir)
    
    # Load the index back
    ann_index_loaded = ANNIndex(use_gpu=args.use_gpu)
    ann_index_loaded.load(args.output_dir)
    
    # Test search
    num_queries = 100
    k_values = [1, 10, 100]
    
    query_vectors = np.random.randn(num_queries, args.dimension).astype(np.float32)
    
    search_times = []
    
    logger.info(f"Testing search with {num_queries} queries...")
    
    for k in k_values:
        # Search with loaded index
        t0 = time.time()
        results, distances = ann_index_loaded.search(query_vectors, k=k)
        search_time = time.time() - t0
        
        search_times.append(search_time)
        
        logger.info(f"Search with k={k} took {search_time:.4f} seconds")
        logger.info(f"First result: {results[0][:3]}... with distances {distances[0][:3]}...")
    
    # Get stats
    stats = ann_index_loaded.get_stats()
    
    # Plot search time vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, search_times, marker='o')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Search time (seconds)')
    plt.title(f'Search Time vs. k for {args.index_type} index')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "search_time_vs_k.png"))
    
    # Save stats to file
    with open(os.path.join(args.output_dir, "ann_index_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")