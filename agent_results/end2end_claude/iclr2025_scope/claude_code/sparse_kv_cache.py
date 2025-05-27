"""
Adaptive Sparse KV-Cache implementation for efficient long context understanding.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from relevance_predictor import TokenRelevancePredictor, AttentionStatisticsExtractor, HandcraftedFeatureExtractor, RelevanceThresholdController

logger = logging.getLogger(__name__)

class AdaptiveSparseKVCache:
    """
    Adaptive Token-Relevance Sparse KV-Cache (ATSKV) implementation.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int,
        max_seq_len: int,
        feature_dim: int = 64,
        lambda_momentum: float = 0.8,
        initial_sparsity: float = 0.7,
        device: torch.device = None
    ):
        """
        Initialize the Adaptive Sparse KV-Cache.
        
        Args:
            hidden_size: Hidden size of the transformer model
            num_attention_heads: Number of attention heads in the transformer model
            num_hidden_layers: Number of hidden layers in the transformer model
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            feature_dim: Dimension of handcrafted features
            lambda_momentum: Momentum factor for mask updates
            initial_sparsity: Initial sparsity level
            device: Device to run the model on
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.lambda_momentum = lambda_momentum
        self.initial_sparsity = initial_sparsity
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize relevance predictors for each layer
        self.relevance_predictors = nn.ModuleDict({
            f"layer_{i}": TokenRelevancePredictor(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                feature_dim=feature_dim,
                hidden_dim=32
            ).to(self.device) for i in range(num_hidden_layers)
        })
        
        # Initialize feature extractors
        self.attention_extractors = {
            i: AttentionStatisticsExtractor(
                num_heads=num_attention_heads,
                seq_len=max_seq_len
            ) for i in range(num_hidden_layers)
        }
        
        # Temporary placeholder for vocab size
        vocab_size = 50257  # Default for GPT-2, adjust based on model
        
        self.feature_extractors = {
            i: HandcraftedFeatureExtractor(
                feature_dim=feature_dim,
                vocab_size=vocab_size
            ).to(self.device) for i in range(num_hidden_layers)
        }
        
        # Initialize threshold controller
        self.threshold_controller = RelevanceThresholdController(
            num_layers=num_hidden_layers,
            initial_quantile=1.0 - initial_sparsity,
            min_quantile=0.1,
            max_quantile=0.5,
            lambda_momentum=lambda_momentum
        )
        
        # Storage for KV cache
        self.kv_cache = {
            # Each layer will have:
            # 'k': tensor of shape [batch_size, num_heads, seq_len, head_dim]
            # 'v': tensor of shape [batch_size, num_heads, seq_len, head_dim]
            # 'mask': tensor of shape [batch_size, seq_len]
        }
        
        # Metrics tracking
        self.metrics = {
            "memory_usage": [],
            "sparsity_level": [],
            "relevance_scores": {},
            "thresholds": {},
            "attention_patterns": {},
        }
    
    def update_relevance_predictor(self, layer_idx: int, optimizer, loss: torch.Tensor):
        """
        Update the relevance predictor for a specific layer using gradient descent.
        
        Args:
            layer_idx: Index of the layer
            optimizer: The optimizer for the relevance predictor
            loss: The loss tensor
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def reset_cache(self):
        """Reset the KV cache."""
        self.kv_cache = {}
    
    def predict_token_relevance(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_scores: torch.Tensor,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict the relevance of each token's KV representation.
        
        Args:
            layer_idx: Index of the current layer
            hidden_states: Hidden states from the transformer layer [batch_size, seq_len, hidden_size]
            attention_scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Optional token type IDs [batch_size, seq_len]
            
        Returns:
            Relevance scores [batch_size, seq_len]
        """
        # Extract attention statistics
        attention_features = self.attention_extractors[layer_idx].extract_statistics(
            attention_scores=attention_scores,
            layer_idx=layer_idx,
            token_types=token_type_ids
        )
        
        # Compute handcrafted features
        hidden_state_norms = torch.norm(hidden_states, dim=-1)
        
        handcrafted_features = self.feature_extractors[layer_idx].extract_features(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            hidden_state_norms=hidden_state_norms,
            layer_idx=layer_idx
        )
        
        # Predict relevance scores
        relevance_scores = self.relevance_predictors[f"layer_{layer_idx}"](
            hidden_states=hidden_states,
            attention_patterns=attention_features,
            handcrafted_features=handcrafted_features
        )
        
        # Store metrics
        if layer_idx not in self.metrics["relevance_scores"]:
            self.metrics["relevance_scores"][layer_idx] = []
        
        self.metrics["relevance_scores"][layer_idx].append(relevance_scores.detach().clone())
        
        return relevance_scores
    
    def compute_retention_mask(
        self,
        layer_idx: int,
        relevance_scores: torch.Tensor,
        current_memory: float,
        target_memory: float
    ) -> torch.Tensor:
        """
        Compute the binary mask for KV cache retention.
        
        Args:
            layer_idx: Index of the current layer
            relevance_scores: Relevance scores [batch_size, seq_len]
            current_memory: Current memory usage
            target_memory: Target memory usage
            
        Returns:
            Binary mask [batch_size, seq_len]
        """
        # Compute threshold and mask
        threshold, mask = self.threshold_controller.compute_threshold(
            relevance_scores=relevance_scores,
            layer_idx=layer_idx,
            current_memory=current_memory,
            target_memory=target_memory
        )
        
        # Store metrics
        if layer_idx not in self.metrics["thresholds"]:
            self.metrics["thresholds"][layer_idx] = []
        
        self.metrics["thresholds"][layer_idx].append(threshold)
        
        return mask
    
    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        retention_mask: torch.Tensor
    ):
        """
        Update the KV cache based on retention mask.
        
        Args:
            layer_idx: Index of the current layer
            key_states: Key states [batch_size, num_heads, seq_len, head_dim]
            value_states: Value states [batch_size, num_heads, seq_len, head_dim]
            retention_mask: Binary mask [batch_size, seq_len]
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key not in self.kv_cache:
            self.kv_cache[layer_key] = {}
        
        # Expand mask for broadcasting
        mask_expanded = retention_mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
        
        # Apply mask to key and value states
        masked_keys = key_states * mask_expanded
        masked_values = value_states * mask_expanded
        
        # Store in cache
        self.kv_cache[layer_key]['k'] = masked_keys
        self.kv_cache[layer_key]['v'] = masked_values
        self.kv_cache[layer_key]['mask'] = retention_mask
        
        # Compute and store sparsity level
        sparsity = 1.0 - (retention_mask.float().mean().item())
        
        if len(self.metrics["sparsity_level"]) <= layer_idx:
            self.metrics["sparsity_level"].append([])
        
        self.metrics["sparsity_level"][layer_idx].append(sparsity)
    
    def get_cached_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the cached key-value tensors for a layer.
        
        Args:
            layer_idx: Index of the current layer
            
        Returns:
            Tuple of (key_states, value_states, mask)
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key in self.kv_cache:
            return (
                self.kv_cache[layer_key]['k'],
                self.kv_cache[layer_key]['v'],
                self.kv_cache[layer_key]['mask']
            )
        
        return None, None, None
    
    def compute_memory_usage(self) -> Dict[str, float]:
        """
        Compute the current memory usage of the KV cache.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        total_elements = 0
        total_active_elements = 0
        
        for layer_key, layer_cache in self.kv_cache.items():
            if 'k' in layer_cache and 'v' in layer_cache and 'mask' in layer_cache:
                # Count elements in key and value tensors
                k_elements = layer_cache['k'].numel()
                v_elements = layer_cache['v'].numel()
                
                # Count active elements (non-zero)
                k_active = torch.count_nonzero(layer_cache['k']).item()
                v_active = torch.count_nonzero(layer_cache['v']).item()
                
                total_elements += k_elements + v_elements
                total_active_elements += k_active + v_active
        
        # Compute memory in bytes (assuming float32)
        bytes_per_element = 4  # float32
        total_memory = total_elements * bytes_per_element
        active_memory = total_active_elements * bytes_per_element
        
        # Convert to MB
        total_memory_mb = total_memory / (1024 * 1024)
        active_memory_mb = active_memory / (1024 * 1024)
        
        # Compute sparsity
        sparsity = 1.0 - (active_memory / total_memory) if total_memory > 0 else 0.0
        
        # Store metrics
        self.metrics["memory_usage"].append(active_memory_mb)
        
        return {
            "total_memory_mb": total_memory_mb,
            "active_memory_mb": active_memory_mb,
            "sparsity": sparsity
        }
    
    def save_metrics(self, save_dir: str):
        """
        Save metrics to disk.
        
        Args:
            save_dir: Directory to save metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert tensors to lists
        serializable_metrics = {}
        
        # Process relevance scores
        relevance_dict = {}
        for layer_idx, scores_list in self.metrics["relevance_scores"].items():
            relevance_dict[f"layer_{layer_idx}"] = [s.cpu().numpy().tolist() for s in scores_list]
        
        serializable_metrics["relevance_scores"] = relevance_dict
        
        # Process thresholds
        serializable_metrics["thresholds"] = self.metrics["thresholds"]
        
        # Process sparsity levels
        serializable_metrics["sparsity_level"] = self.metrics["sparsity_level"]
        
        # Process memory usage
        serializable_metrics["memory_usage"] = self.metrics["memory_usage"]
        
        # Save to disk
        import json
        with open(os.path.join(save_dir, "atskv_metrics.json"), 'w') as f:
            json.dump(serializable_metrics, f)
    
    def train_mode(self):
        """Set the relevance predictors to training mode."""
        for predictor in self.relevance_predictors.values():
            predictor.train()
    
    def eval_mode(self):
        """Set the relevance predictors to evaluation mode."""
        for predictor in self.relevance_predictors.values():
            predictor.eval()
    
    def save_model(self, save_dir: str):
        """
        Save the relevance predictors to disk.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each layer's relevance predictor
        for layer_name, predictor in self.relevance_predictors.items():
            torch.save(
                predictor.state_dict(),
                os.path.join(save_dir, f"{layer_name}_relevance_predictor.pt")
            )
        
        # Save threshold controller parameters
        import json
        with open(os.path.join(save_dir, "threshold_controller_params.json"), 'w') as f:
            json.dump({
                "layer_quantiles": self.threshold_controller.layer_quantiles,
                "layer_betas": self.threshold_controller.layer_betas
            }, f)
    
    def load_model(self, load_dir: str):
        """
        Load the relevance predictors from disk.
        
        Args:
            load_dir: Directory to load models from
        """
        # Load each layer's relevance predictor
        for layer_name, predictor in self.relevance_predictors.items():
            predictor.load_state_dict(
                torch.load(os.path.join(load_dir, f"{layer_name}_relevance_predictor.pt"))
            )
        
        # Load threshold controller parameters
        import json
        with open(os.path.join(load_dir, "threshold_controller_params.json"), 'r') as f:
            params = json.load(f)
            self.threshold_controller.layer_quantiles = params["layer_quantiles"]
            self.threshold_controller.layer_betas = params["layer_betas"]


class ExternalMemoryManager:
    """
    Manager for external memory to offload less relevant KV pairs.
    """
    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        max_seq_len: int,
        hash_dim: int = 16
    ):
        """
        Initialize the external memory manager.
        
        Args:
            num_layers: Number of layers in the model
            head_dim: Dimension of each attention head
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            hash_dim: Dimension for locality-sensitive hashing
        """
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.hash_dim = hash_dim
        
        # Initialize projection matrices for LSH
        self.projection_matrices = {
            i: np.random.randn(hash_dim, head_dim) for i in range(num_layers)
        }
        
        # Initialize external storage
        self.external_storage = {
            i: {
                'k': {},  # Will store key vectors by bucket
                'v': {}   # Will store value vectors by bucket
            } for i in range(num_layers)
        }
        
        # Position mapping to track original positions
        self.position_mapping = {i: {} for i in range(num_layers)}
        
        # Initialize promotion and demotion thresholds
        self.promotion_thresholds = {i: 0.8 for i in range(num_layers)}
        self.demotion_thresholds = {i: 0.2 for i in range(num_layers)}
    
    def hash_vector(self, layer_idx: int, key_vector: np.ndarray) -> str:
        """
        Hash a key vector using locality-sensitive hashing.
        
        Args:
            layer_idx: Index of the layer
            key_vector: Key vector to hash
            
        Returns:
            Hash string representing the bucket
        """
        # Project the vector
        projection = np.sign(np.dot(self.projection_matrices[layer_idx], key_vector))
        
        # Convert to binary string
        binary = ''.join(['1' if bit > 0 else '0' for bit in projection])
        
        return binary
    
    def store_kv_pair(
        self,
        layer_idx: int,
        position: int,
        key_vector: np.ndarray,
        value_vector: np.ndarray
    ):
        """
        Store a KV pair in external memory.
        
        Args:
            layer_idx: Index of the layer
            position: Original position in the sequence
            key_vector: Key vector
            value_vector: Value vector
        """
        # Hash the key vector to find the bucket
        bucket = self.hash_vector(layer_idx, key_vector)
        
        # Initialize bucket if needed
        if bucket not in self.external_storage[layer_idx]['k']:
            self.external_storage[layer_idx]['k'][bucket] = []
            self.external_storage[layer_idx]['v'][bucket] = []
            self.position_mapping[layer_idx][bucket] = []
        
        # Store vectors and position
        self.external_storage[layer_idx]['k'][bucket].append(key_vector)
        self.external_storage[layer_idx]['v'][bucket].append(value_vector)
        self.position_mapping[layer_idx][bucket].append(position)
    
    def retrieve_similar_kv_pairs(
        self,
        layer_idx: int,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Retrieve KV pairs similar to a query vector.
        
        Args:
            layer_idx: Index of the layer
            query_vector: Query vector
            top_k: Number of similar vectors to retrieve
            
        Returns:
            Tuple of (key_vectors, value_vectors, positions)
        """
        # Hash the query vector to find the bucket
        bucket = self.hash_vector(layer_idx, query_vector)
        
        # Get vectors from the bucket
        if bucket in self.external_storage[layer_idx]['k']:
            key_vectors = self.external_storage[layer_idx]['k'][bucket]
            value_vectors = self.external_storage[layer_idx]['v'][bucket]
            positions = self.position_mapping[layer_idx][bucket]
            
            if not key_vectors:
                return [], [], []
            
            # Calculate similarities
            similarities = [np.dot(query_vector, k) / (np.linalg.norm(query_vector) * np.linalg.norm(k)) 
                          for k in key_vectors]
            
            # Get top-k indices
            if len(similarities) <= top_k:
                top_indices = list(range(len(similarities)))
            else:
                top_indices = np.argsort(similarities)[-top_k:]
            
            # Return top-k vectors and positions
            return (
                [key_vectors[i] for i in top_indices],
                [value_vectors[i] for i in top_indices],
                [positions[i] for i in top_indices]
            )
        
        return [], [], []
    
    def migrate_kv_pairs(
        self,
        layer_idx: int,
        relevance_scores: np.ndarray,
        active_cache_k: np.ndarray,
        active_cache_v: np.ndarray,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Migrate KV pairs between active cache and external storage.
        
        Args:
            layer_idx: Index of the layer
            relevance_scores: Relevance scores for each position
            active_cache_k: Key vectors in active cache
            active_cache_v: Value vectors in active cache
            positions: Positions in the sequence
            
        Returns:
            Tuple of (updated_k, updated_v, positions_to_promote, positions_to_demote)
        """
        promote_threshold = self.promotion_thresholds[layer_idx]
        demote_threshold = self.demotion_thresholds[layer_idx]
        
        positions_to_promote = []
        positions_to_demote = []
        
        # Identify positions to demote (move from active to external)
        for i, (pos, score) in enumerate(zip(positions, relevance_scores)):
            if score < demote_threshold:
                positions_to_demote.append(i)
                
                # Store in external memory
                self.store_kv_pair(
                    layer_idx=layer_idx,
                    position=pos,
                    key_vector=active_cache_k[i],
                    value_vector=active_cache_v[i]
                )
        
        # Remove demoted pairs from active cache
        mask = np.ones(len(positions), dtype=bool)
        mask[positions_to_demote] = False
        
        active_cache_k = active_cache_k[mask]
        active_cache_v = active_cache_v[mask]
        active_positions = positions[mask]
        
        # TODO: Implement promotion from external storage
        # This would require tracking relevance scores for external storage
        
        return active_cache_k, active_cache_v, active_positions, positions_to_demote