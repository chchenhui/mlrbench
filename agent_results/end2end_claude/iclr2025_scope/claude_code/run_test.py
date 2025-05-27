"""
Test script for ATSKV (Adaptive Token-Relevance Sparse KV-Cache) implementation.
This script performs basic functionality tests to ensure the implementation is working correctly.
"""
import os
import sys
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import local modules
from relevance_predictor import TokenRelevancePredictor, AttentionStatisticsExtractor, HandcraftedFeatureExtractor, RelevanceThresholdController
from sparse_kv_cache import AdaptiveSparseKVCache
from baselines import KVCacheFactory, FullKVCache, SlidingWindowKVCache, DynamicKVCache, RocketKVCache
from utils import set_seed, get_device, memory_usage_stats, calculate_kv_cache_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_relevance_predictor():
    """Test the token relevance predictor."""
    logger.info("Testing token relevance predictor...")
    
    # Create test inputs
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 12
    feature_dim = 64
    
    # Create predictor
    predictor = TokenRelevancePredictor(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        feature_dim=feature_dim
    )
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_patterns = torch.randn(batch_size, seq_len, num_heads)
    handcrafted_features = torch.randn(batch_size, seq_len, feature_dim)
    
    # Forward pass
    relevance_scores = predictor(
        hidden_states=hidden_states,
        attention_patterns=attention_patterns,
        handcrafted_features=handcrafted_features
    )
    
    # Check output shape
    assert relevance_scores.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {relevance_scores.shape}"
    
    # Check values are in [0, 1] (sigmoid output)
    assert torch.all(relevance_scores >= 0) and torch.all(relevance_scores <= 1), "Relevance scores should be in [0, 1]"
    
    logger.info("Token relevance predictor test passed!")
    return True

def test_attention_statistics_extractor():
    """Test the attention statistics extractor."""
    logger.info("Testing attention statistics extractor...")
    
    # Create test inputs
    batch_size = 2
    seq_len = 10
    num_heads = 12
    
    # Create extractor
    extractor = AttentionStatisticsExtractor(
        num_heads=num_heads,
        seq_len=seq_len
    )
    
    # Create test inputs
    attention_scores = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # Extract statistics
    attention_features = extractor.extract_statistics(
        attention_scores=attention_scores,
        layer_idx=0
    )
    
    # Check output shape
    assert attention_features.shape == (batch_size, seq_len, num_heads), f"Expected shape {(batch_size, seq_len, num_heads)}, got {attention_features.shape}"
    
    logger.info("Attention statistics extractor test passed!")
    return True

def test_handcrafted_feature_extractor():
    """Test the handcrafted feature extractor."""
    logger.info("Testing handcrafted feature extractor...")
    
    # Create test inputs
    batch_size = 2
    seq_len = 10
    vocab_size = 50000
    feature_dim = 64
    
    # Create extractor
    extractor = HandcraftedFeatureExtractor(
        feature_dim=feature_dim,
        vocab_size=vocab_size
    )
    
    # Create test inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden_state_norms = torch.rand(batch_size, seq_len)
    
    # Extract features
    features = extractor.extract_features(
        input_ids=input_ids,
        hidden_state_norms=hidden_state_norms,
        layer_idx=0
    )
    
    # Check output shape
    assert features.shape == (batch_size, seq_len, feature_dim), f"Expected shape {(batch_size, seq_len, feature_dim)}, got {features.shape}"
    
    logger.info("Handcrafted feature extractor test passed!")
    return True

def test_threshold_controller():
    """Test the relevance threshold controller."""
    logger.info("Testing relevance threshold controller...")
    
    # Create test inputs
    batch_size = 2
    seq_len = 10
    num_layers = 12
    
    # Create controller
    controller = RelevanceThresholdController(
        num_layers=num_layers,
        initial_quantile=0.7,
        min_quantile=0.5,
        max_quantile=0.9
    )
    
    # Create test inputs
    relevance_scores = torch.rand(batch_size, seq_len)
    
    # Compute threshold and mask
    threshold, mask = controller.compute_threshold(
        relevance_scores=relevance_scores,
        layer_idx=0,
        current_memory=1000,
        target_memory=800
    )
    
    # Check threshold is a number
    assert isinstance(threshold, float), f"Expected threshold to be a float, got {type(threshold)}"
    
    # Check mask shape and values
    assert mask.shape == (batch_size, seq_len), f"Expected mask shape {(batch_size, seq_len)}, got {mask.shape}"
    assert torch.all((mask == 0) | (mask == 1)), "Mask should contain only 0s and 1s"
    
    logger.info("Threshold controller test passed!")
    return True

def test_kv_cache_factory():
    """Test the KV cache factory."""
    logger.info("Testing KV cache factory...")
    
    # Set parameters
    hidden_size = 768
    num_attention_heads = 12
    num_hidden_layers = 12
    head_dim = hidden_size // num_attention_heads
    max_seq_len = 1024
    
    # Test creating each type of KV cache
    methods = ["full", "sliding_window", "dynamic_kv", "rocket_kv", "atskv"]
    
    for method in methods:
        logger.info(f"Creating {method} KV cache...")
        
        try:
            kv_cache = KVCacheFactory.create(
                method=method,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len
            )
            
            # Check the type
            if method == "full":
                assert isinstance(kv_cache, FullKVCache), f"Expected FullKVCache, got {type(kv_cache)}"
            elif method == "sliding_window":
                assert isinstance(kv_cache, SlidingWindowKVCache), f"Expected SlidingWindowKVCache, got {type(kv_cache)}"
            elif method == "dynamic_kv":
                assert isinstance(kv_cache, DynamicKVCache), f"Expected DynamicKVCache, got {type(kv_cache)}"
            elif method == "rocket_kv":
                assert isinstance(kv_cache, RocketKVCache), f"Expected RocketKVCache, got {type(kv_cache)}"
            elif method == "atskv":
                assert isinstance(kv_cache, AdaptiveSparseKVCache), f"Expected AdaptiveSparseKVCache, got {type(kv_cache)}"
            
            logger.info(f"{method} KV cache created successfully")
            
        except Exception as e:
            logger.error(f"Error creating {method} KV cache: {e}")
            return False
    
    logger.info("KV cache factory test passed!")
    return True

def test_sparse_kv_cache():
    """Test the adaptive sparse KV cache implementation."""
    logger.info("Testing adaptive sparse KV cache...")
    
    # Set parameters
    hidden_size = 768
    num_attention_heads = 12
    num_hidden_layers = 12
    head_dim = hidden_size // num_attention_heads
    max_seq_len = 1024
    batch_size = 2
    seq_len = 10
    
    # Create the sparse KV cache
    sparse_kv = AdaptiveSparseKVCache(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Create test inputs
    layer_idx = 0
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_scores = torch.rand(batch_size, num_attention_heads, seq_len, seq_len)
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    # Test token relevance prediction
    relevance_scores = sparse_kv.predict_token_relevance(
        layer_idx=layer_idx,
        hidden_states=hidden_states,
        attention_scores=attention_scores,
        input_ids=input_ids
    )
    
    # Check relevance scores shape and values
    assert relevance_scores.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {relevance_scores.shape}"
    assert torch.all(relevance_scores >= 0) and torch.all(relevance_scores <= 1), "Relevance scores should be in [0, 1]"
    
    # Test computing retention mask
    mask = sparse_kv.compute_retention_mask(
        layer_idx=layer_idx,
        relevance_scores=relevance_scores,
        current_memory=1000,
        target_memory=800
    )
    
    # Check mask shape and values
    assert mask.shape == (batch_size, seq_len), f"Expected mask shape {(batch_size, seq_len)}, got {mask.shape}"
    assert torch.all((mask == 0) | (mask == 1)), "Mask should contain only 0s and 1s"
    
    # Test updating KV cache
    key_states = torch.randn(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_attention_heads, seq_len, head_dim)
    
    sparse_kv.update_kv_cache(
        layer_idx=layer_idx,
        key_states=key_states,
        value_states=value_states,
        retention_mask=mask
    )
    
    # Test getting cached KV
    cached_k, cached_v, cached_mask = sparse_kv.get_cached_kv(layer_idx)
    
    # Check output shapes
    assert cached_k.shape == key_states.shape, f"Expected cached_k shape {key_states.shape}, got {cached_k.shape}"
    assert cached_v.shape == value_states.shape, f"Expected cached_v shape {value_states.shape}, got {cached_v.shape}"
    assert cached_mask.shape == mask.shape, f"Expected cached_mask shape {mask.shape}, got {cached_mask.shape}"
    
    # Test computing memory usage
    memory_usage = sparse_kv.compute_memory_usage()
    
    # Check memory usage is a dictionary with expected keys
    assert isinstance(memory_usage, dict), f"Expected memory_usage to be a dict, got {type(memory_usage)}"
    assert "total_memory_mb" in memory_usage, "Memory usage should include 'total_memory_mb'"
    assert "active_memory_mb" in memory_usage, "Memory usage should include 'active_memory_mb'"
    assert "sparsity" in memory_usage, "Memory usage should include 'sparsity'"
    
    logger.info("Adaptive sparse KV cache test passed!")
    return True

def test_model_integration(model_name="sshleifer/tiny-gpt2"):
    """
    Test integration with a small pre-trained model.
    
    Args:
        model_name: Name of a small pre-trained model for testing
    """
    logger.info(f"Testing integration with model: {model_name}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get model configuration
        config = model.config
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        num_attention_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        num_hidden_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
        head_dim = hidden_size // num_attention_heads
        max_seq_len = 128  # Short sequence for testing
        
        # Create all KV cache implementations
        kv_cache_impls = {
            "full": KVCacheFactory.create("full", hidden_size, num_attention_heads, num_hidden_layers, head_dim, max_seq_len),
            "sliding_window": KVCacheFactory.create("sliding_window", hidden_size, num_attention_heads, num_hidden_layers, head_dim, max_seq_len),
            "dynamic_kv": KVCacheFactory.create("dynamic_kv", hidden_size, num_attention_heads, num_hidden_layers, head_dim, max_seq_len),
            "rocket_kv": KVCacheFactory.create("rocket_kv", hidden_size, num_attention_heads, num_hidden_layers, head_dim, max_seq_len),
            "atskv": KVCacheFactory.create("atskv", hidden_size, num_attention_heads, num_hidden_layers, head_dim, max_seq_len)
        }
        
        # Create sample input
        sample_text = "This is a test sentence for evaluating KV cache implementations."
        inputs = tokenizer(sample_text, return_tensors="pt")
        
        # Run forward pass with each KV cache implementation
        for method, kv_cache in kv_cache_impls.items():
            logger.info(f"Testing {method} KV cache with model...")
            
            # Reset cache
            kv_cache.reset_cache()
            
            # Forward pass with attention output
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get hidden states and attention scores
                hidden_states = outputs.hidden_states
                attention_scores = outputs.attentions
                
                # Update KV cache for each layer
                for layer_idx in range(len(hidden_states) - 1):
                    # Get hidden state and attention for this layer
                    layer_hidden_state = hidden_states[layer_idx + 1]
                    layer_attention = attention_scores[layer_idx]
                    
                    # Extract key and value states (simplified)
                    batch_size, seq_len, h_size = layer_hidden_state.shape
                    h_dim = h_size // num_attention_heads
                    
                    # Simplistic key-value approximation
                    key_states = layer_hidden_state.view(
                        batch_size, seq_len, num_attention_heads, h_dim
                    ).transpose(1, 2)
                    value_states = layer_hidden_state.view(
                        batch_size, seq_len, num_attention_heads, h_dim
                    ).transpose(1, 2)
                    
                    # Update KV cache based on method
                    if method == "atskv":
                        # For ATSKV
                        relevance_scores = kv_cache.predict_token_relevance(
                            layer_idx=layer_idx,
                            hidden_states=layer_hidden_state,
                            attention_scores=layer_attention,
                            input_ids=inputs["input_ids"]
                        )
                        
                        memory_usage = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                        target_memory = memory_usage * 0.5
                        
                        retention_mask = kv_cache.compute_retention_mask(
                            layer_idx=layer_idx,
                            relevance_scores=relevance_scores,
                            current_memory=memory_usage,
                            target_memory=target_memory
                        )
                        
                        kv_cache.update_kv_cache(
                            layer_idx=layer_idx,
                            key_states=key_states,
                            value_states=value_states,
                            retention_mask=retention_mask
                        )
                    elif method in ["dynamic_kv", "rocket_kv"]:
                        # For DynamicKV and RocketKV
                        kv_cache.update_kv_cache(
                            layer_idx=layer_idx,
                            key_states=key_states,
                            value_states=value_states,
                            attention_scores=layer_attention
                        )
                    else:
                        # For Full and Sliding Window
                        kv_cache.update_kv_cache(
                            layer_idx=layer_idx,
                            key_states=key_states,
                            value_states=value_states
                        )
                
                # Compute memory usage
                memory_usage = kv_cache.compute_memory_usage()
                logger.info(f"{method} KV cache memory usage: {memory_usage}")
        
        logger.info("Model integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in model integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    logger.info("Running all tests...")
    
    tests = [
        test_relevance_predictor,
        test_attention_statistics_extractor,
        test_handcrafted_feature_extractor,
        test_threshold_controller,
        test_kv_cache_factory,
        test_sparse_kv_cache,
        test_model_integration
    ]
    
    results = {}
    all_passed = True
    
    for test_func in tests:
        test_name = test_func.__name__
        try:
            passed = test_func()
            results[test_name] = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"Exception in {test_name}: {e}")
            results[test_name] = "ERROR"
            all_passed = False
    
    # Print summary
    logger.info("Test results summary:")
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    if all_passed:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed or had errors!")
        return 1

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Run all tests
    sys.exit(run_all_tests())