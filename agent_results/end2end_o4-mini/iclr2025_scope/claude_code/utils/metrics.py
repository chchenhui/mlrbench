import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import psutil
import logging
import math
from transformers import PreTrainedTokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import nltk
import os
import warnings

# Download NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Class to track and compute performance metrics."""
    
    def __init__(self):
        """Initialize performance metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_tokens = 0
        self.total_time = 0
        self.peak_memory = 0
        self.total_loss = 0
        self.num_batches = 0
        self.batch_sizes = []
        self.tokens_per_batch = []
        self.time_per_batch = []
        self.memory_per_batch = []
    
    def update(self, 
               batch_size: int, 
               seq_length: int, 
               elapsed_time: float, 
               loss: Optional[float] = None):
        """
        Update metrics with batch information.
        
        Args:
            batch_size: Number of examples in the batch
            seq_length: Sequence length
            elapsed_time: Time taken for the batch
            loss: Optional loss value
        """
        tokens_in_batch = batch_size * seq_length
        
        self.total_tokens += tokens_in_batch
        self.total_time += elapsed_time
        
        # Track peak memory usage (in GB)
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Track loss if provided
        if loss is not None:
            self.total_loss += loss * batch_size
            self.num_batches += 1
        
        # Store batch-level metrics
        self.batch_sizes.append(batch_size)
        self.tokens_per_batch.append(tokens_in_batch)
        self.time_per_batch.append(elapsed_time)
        self.memory_per_batch.append(current_memory)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get computed metrics.
        
        Returns:
            metrics: Dictionary of metrics
        """
        # Avoid division by zero
        if self.total_time == 0 or self.total_tokens == 0 or self.num_batches == 0:
            return {
                "tokens_per_second": 0,
                "ms_per_token": 0,
                "peak_memory_gb": self.peak_memory,
                "average_loss": 0 if self.num_batches == 0 else self.total_loss / sum(self.batch_sizes),
                "perplexity": float('inf')
            }
        
        # Compute aggregate metrics
        tokens_per_second = self.total_tokens / self.total_time
        ms_per_token = (self.total_time * 1000) / self.total_tokens
        avg_loss = self.total_loss / sum(self.batch_sizes)
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            "tokens_per_second": tokens_per_second,
            "ms_per_token": ms_per_token,
            "peak_memory_gb": self.peak_memory,
            "average_loss": avg_loss,
            "perplexity": perplexity
        }
    
    def get_batch_metrics(self) -> Dict[str, List[float]]:
        """
        Get batch-level metrics for analysis.
        
        Returns:
            metrics: Dictionary of batch-level metrics lists
        """
        # Compute per-batch throughput
        throughput_per_batch = [tokens / time for tokens, time in zip(self.tokens_per_batch, self.time_per_batch)]
        
        return {
            "batch_sizes": self.batch_sizes,
            "tokens_per_batch": self.tokens_per_batch,
            "time_per_batch": self.time_per_batch,
            "memory_per_batch": self.memory_per_batch,
            "throughput_per_batch": throughput_per_batch
        }


class TextGenerationMetrics:
    """Class to compute text generation quality metrics."""
    
    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        """
        Initialize text generation metrics calculator.
        
        Args:
            tokenizer: Tokenizer for processing text
        """
        self.tokenizer = tokenizer
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
    
    def compute_rouge(self, 
                     predictions: List[str], 
                     references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of generated text
            references: List of reference text
            
        Returns:
            scores: Dictionary of ROUGE scores
        """
        if not predictions or not references:
            return {
                "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
            }
        
        # Ensure inputs are non-empty
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                       if p.strip() and r.strip()]
        
        if not valid_pairs:
            warnings.warn("No valid prediction-reference pairs for ROUGE computation.")
            return {
                "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
            }
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        try:
            scores = self.rouge.get_scores(valid_preds, valid_refs, avg=True)
            return scores
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            logger.error(f"Sample prediction: {valid_preds[0][:100]}")
            logger.error(f"Sample reference: {valid_refs[0][:100]}")
            return {
                "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
            }
    
    def compute_bleu(self, 
                    predictions: List[str], 
                    references: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of generated text
            references: List of reference text
            
        Returns:
            bleu_score: BLEU score
        """
        if not predictions or not references:
            return 0.0
        
        # Tokenize predictions and references
        tokenized_preds = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        tokenized_refs = [[nltk.word_tokenize(ref.lower())] for ref in references]  # BLEU expects list of list of references
        
        try:
            bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=self.smooth)
            return bleu_score
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0
    
    def compute_perplexity(self, 
                          model: nn.Module, 
                          input_ids: torch.Tensor, 
                          labels: torch.Tensor) -> float:
        """
        Compute perplexity of a model on given inputs.
        
        Args:
            model: Model to evaluate
            input_ids: Input token IDs
            labels: Target token IDs
            
        Returns:
            perplexity: Perplexity score
        """
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        
        try:
            perplexity = math.exp(loss.item())
            return perplexity
        except OverflowError:
            return float('inf')
        

class CompressionMetrics:
    """Class to compute compression-specific metrics."""
    
    def __init__(self):
        """Initialize compression metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.original_sizes = []
        self.compressed_sizes = []
        self.compression_times = []
    
    def update(self, 
              original_size: int, 
              compressed_size: int, 
              compression_time: float):
        """
        Update compression metrics.
        
        Args:
            original_size: Original KV cache size
            compressed_size: Compressed KV cache size
            compression_time: Time taken for compression
        """
        self.original_sizes.append(original_size)
        self.compressed_sizes.append(compressed_size)
        self.compression_times.append(compression_time)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get computed compression metrics.
        
        Returns:
            metrics: Dictionary of compression metrics
        """
        if not self.original_sizes:
            return {
                "average_compression_ratio": 0,
                "average_compression_time_ms": 0,
                "compression_overhead_percent": 0
            }
        
        # Compute compression ratio
        compression_ratios = [orig / comp if comp > 0 else 0 
                             for orig, comp in zip(self.original_sizes, self.compressed_sizes)]
        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
        
        # Compute average compression time in milliseconds
        avg_compression_time_ms = sum(self.compression_times) * 1000 / len(self.compression_times)
        
        # Estimate overhead as percentage of total processing time
        # This is a rough estimate - actual overhead depends on total inference time
        total_tokens = sum(self.original_sizes)
        avg_tokens_per_compression = total_tokens / len(self.original_sizes)
        
        # Assuming 10 tokens/second baseline throughput for estimation
        estimated_baseline_time = total_tokens / 10
        total_compression_time = sum(self.compression_times)
        compression_overhead_percent = (total_compression_time / estimated_baseline_time) * 100 if estimated_baseline_time > 0 else 0
        
        return {
            "average_compression_ratio": avg_compression_ratio,
            "average_compression_time_ms": avg_compression_time_ms,
            "compression_overhead_percent": compression_overhead_percent
        }