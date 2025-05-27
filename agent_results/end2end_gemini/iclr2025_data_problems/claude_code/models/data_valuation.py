"""
Data valuation framework for RAG-Informed Dynamic Data Valuation.
"""
import os
import json
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import math
from scipy.stats import pearsonr, spearmanr

class DataValuationMethod:
    """Base class for data valuation methods."""
    
    def __init__(self, name: str):
        """
        Initialize data valuation method.
        
        Args:
            name: Name of the valuation method
        """
        self.name = name
    
    def calculate_value(
        self, 
        data_chunk: Any, 
        attribution_scores: List[Dict[str, Any]] = None,
        retrieval_counts: int = None,
        user_feedback: List[Dict[str, Any]] = None,
        timestamp: float = None,
        additional_info: Dict[str, Any] = None
    ) -> float:
        """
        Calculate the value of a data chunk.
        
        Args:
            data_chunk: The data chunk to value
            attribution_scores: List of attribution scores for the chunk
            retrieval_counts: Number of times the chunk was retrieved
            user_feedback: List of user feedback on answers using this chunk
            timestamp: Current timestamp
            additional_info: Any additional information for valuation
            
        Returns:
            Float value for the chunk
        """
        raise NotImplementedError("Subclasses must implement calculate_value method")

class StaticPricing(DataValuationMethod):
    """Simple static pricing based on chunk size."""
    
    def __init__(self, price_per_token: float = 0.01):
        """
        Initialize static pricing.
        
        Args:
            price_per_token: Price per token in the chunk
        """
        super().__init__("static_pricing")
        self.price_per_token = price_per_token
    
    def calculate_value(
        self, 
        data_chunk: Any, 
        attribution_scores: List[Dict[str, Any]] = None,
        retrieval_counts: int = None,
        user_feedback: List[Dict[str, Any]] = None,
        timestamp: float = None,
        additional_info: Dict[str, Any] = None
    ) -> float:
        """
        Calculate value based on chunk size.
        
        Args:
            data_chunk: The data chunk to value
            
        Returns:
            Value based on token count
        """
        token_count = len(data_chunk.text.split())
        return token_count * self.price_per_token

class PopularityBasedPricing(DataValuationMethod):
    """Pricing based on retrieval frequency."""
    
    def __init__(self, base_price: float = 1.0, log_factor: float = 1.0):
        """
        Initialize popularity-based pricing.
        
        Args:
            base_price: Base price for all chunks
            log_factor: Factor to apply to log of retrieval count
        """
        super().__init__("popularity_pricing")
        self.base_price = base_price
        self.log_factor = log_factor
    
    def calculate_value(
        self, 
        data_chunk: Any, 
        attribution_scores: List[Dict[str, Any]] = None,
        retrieval_counts: int = None,
        user_feedback: List[Dict[str, Any]] = None,
        timestamp: float = None,
        additional_info: Dict[str, Any] = None
    ) -> float:
        """
        Calculate value based on popularity (retrieval count).
        
        Args:
            data_chunk: The data chunk to value
            retrieval_counts: Number of times the chunk was retrieved
            
        Returns:
            Value based on retrieval frequency
        """
        # Use the provided retrieval count or get it from the chunk
        count = retrieval_counts if retrieval_counts is not None else data_chunk.retrieval_count
        
        # Apply log transformation to avoid extreme prices
        return self.base_price + self.log_factor * math.log(count + 1)

class DynamicRAGValuation(DataValuationMethod):
    """
    RAG-Informed Dynamic Data Valuation method.
    
    This method calculates data value based on:
    1. Attribution scores from RAG system
    2. Output quality metrics
    3. Retrieval frequency
    4. User feedback
    5. Temporal factors (recency)
    """
    
    def __init__(
        self,
        attribution_weight: float = 0.4,
        popularity_weight: float = 0.2,
        feedback_weight: float = 0.3,
        recency_weight: float = 0.1,
        base_price: float = 0.5,
        decay_factor: float = 0.01,
        feedback_aggregation: str = "avg"
    ):
        """
        Initialize dynamic RAG valuation.
        
        Args:
            attribution_weight: Weight given to attribution scores
            popularity_weight: Weight given to retrieval frequency
            feedback_weight: Weight given to user feedback
            recency_weight: Weight given to recency factors
            base_price: Base price for all chunks
            decay_factor: Decay factor for time-based discounting (lambda)
            feedback_aggregation: Method to aggregate user feedback ('avg', 'weighted_avg', 'max')
        """
        super().__init__("dynamic_rag_valuation")
        
        self.attribution_weight = attribution_weight
        self.popularity_weight = popularity_weight
        self.feedback_weight = feedback_weight
        self.recency_weight = recency_weight
        self.base_price = base_price
        self.decay_factor = decay_factor
        self.feedback_aggregation = feedback_aggregation
    
    def _calculate_attribution_component(
        self, 
        attribution_scores: List[Dict[str, Any]], 
        current_time: float
    ) -> float:
        """
        Calculate the attribution component of the value.
        
        Args:
            attribution_scores: List of attribution scores
            current_time: Current timestamp
            
        Returns:
            Attribution component value
        """
        if not attribution_scores:
            return 0.0
        
        # Calculate time-weighted attribution score
        weighted_sum = 0.0
        total_weight = 0.0
        
        for attribution in attribution_scores:
            score = attribution["score"]
            timestamp = attribution["timestamp"]
            time_diff = current_time - timestamp
            time_weight = math.exp(-self.decay_factor * time_diff)
            
            weighted_sum += score * time_weight
            total_weight += time_weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_popularity_component(self, retrieval_count: int) -> float:
        """
        Calculate the popularity component of the value.
        
        Args:
            retrieval_count: Number of times the chunk was retrieved
            
        Returns:
            Popularity component value
        """
        # Apply log transformation to avoid extreme prices
        return math.log(retrieval_count + 1) / 10.0  # Normalize somewhat
    
    def _calculate_feedback_component(
        self, 
        user_feedback: List[Dict[str, Any]], 
        current_time: float
    ) -> float:
        """
        Calculate the feedback component of the value.
        
        Args:
            user_feedback: List of user feedback items
            current_time: Current timestamp
            
        Returns:
            Feedback component value
        """
        if not user_feedback:
            return 0.0
        
        if self.feedback_aggregation == "avg":
            # Simple average of feedback scores
            return sum(feedback["score"] for feedback in user_feedback) / len(user_feedback)
        
        elif self.feedback_aggregation == "weighted_avg":
            # Time-weighted average of feedback scores
            weighted_sum = 0.0
            total_weight = 0.0
            
            for feedback in user_feedback:
                score = feedback["score"]
                timestamp = feedback["timestamp"]
                time_diff = current_time - timestamp
                time_weight = math.exp(-self.decay_factor * time_diff)
                
                weighted_sum += score * time_weight
                total_weight += time_weight
            
            if total_weight == 0:
                return 0.0
            
            return weighted_sum / total_weight
        
        elif self.feedback_aggregation == "max":
            # Maximum feedback score
            return max(feedback["score"] for feedback in user_feedback)
        
        else:
            # Default to simple average
            return sum(feedback["score"] for feedback in user_feedback) / len(user_feedback)
    
    def _calculate_recency_component(
        self, 
        attribution_scores: List[Dict[str, Any]], 
        current_time: float
    ) -> float:
        """
        Calculate the recency component of the value.
        
        Args:
            attribution_scores: List of attribution scores with timestamps
            current_time: Current timestamp
            
        Returns:
            Recency component value
        """
        if not attribution_scores:
            return 0.0
        
        # Calculate average recency of usage
        timestamps = [attr["timestamp"] for attr in attribution_scores]
        most_recent = max(timestamps)
        time_diff = current_time - most_recent
        
        # Normalize recency - more recent = higher value
        return math.exp(-self.decay_factor * time_diff)
    
    def calculate_value(
        self, 
        data_chunk: Any, 
        attribution_scores: List[Dict[str, Any]] = None,
        retrieval_counts: int = None,
        user_feedback: List[Dict[str, Any]] = None,
        timestamp: float = None,
        additional_info: Dict[str, Any] = None
    ) -> float:
        """
        Calculate the dynamic RAG-informed value of a data chunk.
        
        Args:
            data_chunk: The data chunk to value
            attribution_scores: List of attribution scores for the chunk
            retrieval_counts: Number of times the chunk was retrieved
            user_feedback: List of user feedback on answers using this chunk
            timestamp: Current timestamp
            additional_info: Any additional information for valuation
            
        Returns:
            Dynamic value for the chunk
        """
        current_time = timestamp if timestamp is not None else time.time()
        
        # Use provided values or get from the chunk
        attribution_scores = attribution_scores or data_chunk.attribution_scores
        retrieval_count = retrieval_counts if retrieval_counts is not None else data_chunk.retrieval_count
        
        # For this simulation, generate synthetic user feedback if not provided
        if user_feedback is None:
            user_feedback = []
            if hasattr(data_chunk, 'quality') and data_chunk.quality is not None:
                # Simulate some user feedback based on the ground truth quality
                for i in range(min(5, retrieval_count)):
                    # Add some noise to the quality score
                    score = min(1.0, max(0.0, data_chunk.quality + np.random.normal(0, 0.1)))
                    user_feedback.append({
                        "score": score,
                        "timestamp": current_time - np.random.uniform(0, 100)  # Random past timestamp
                    })
        
        # Calculate value components
        attribution_component = self._calculate_attribution_component(attribution_scores, current_time)
        popularity_component = self._calculate_popularity_component(retrieval_count)
        feedback_component = self._calculate_feedback_component(user_feedback, current_time)
        recency_component = self._calculate_recency_component(attribution_scores, current_time)
        
        # Combine components with weights
        value = (
            self.base_price +
            self.attribution_weight * attribution_component +
            self.popularity_weight * popularity_component +
            self.feedback_weight * feedback_component +
            self.recency_weight * recency_component
        )
        
        return value

class DataShapleyValuation(DataValuationMethod):
    """
    Simplified Data Shapley valuation method.
    
    Note: Full Data Shapley is computationally expensive, so this is a simplified version
    for demonstration purposes only.
    """
    
    def __init__(self, performance_history: Dict[str, List[Dict[str, Any]]] = None):
        """
        Initialize Data Shapley valuation.
        
        Args:
            performance_history: History of model performance with different datasets
        """
        super().__init__("data_shapley")
        self.performance_history = performance_history or {}
    
    def update_performance_history(self, history_entry: Dict[str, Any]):
        """
        Update the performance history.
        
        Args:
            history_entry: Performance history entry
        """
        task_id = history_entry.get("task_id", "default")
        if task_id not in self.performance_history:
            self.performance_history[task_id] = []
        
        self.performance_history[task_id].append(history_entry)
    
    def calculate_value(
        self, 
        data_chunk: Any, 
        attribution_scores: List[Dict[str, Any]] = None,
        retrieval_counts: int = None,
        user_feedback: List[Dict[str, Any]] = None,
        timestamp: float = None,
        additional_info: Dict[str, Any] = None
    ) -> float:
        """
        Calculate the Data Shapley value of a data chunk.
        
        Args:
            data_chunk: The data chunk to value
            attribution_scores: Not used directly for Data Shapley
            retrieval_counts: Not used directly for Data Shapley
            user_feedback: Not used directly for Data Shapley
            timestamp: Current timestamp
            additional_info: Should include task_id and performance metrics
            
        Returns:
            Data Shapley value for the chunk
        """
        # In a real implementation, we would calculate the marginal contribution
        # of this data chunk to all possible subsets of the dataset
        
        # For this simplified version, we'll use the attribution scores as a proxy
        # for the data chunk's marginal contribution
        
        if not attribution_scores:
            return 0.0
        
        # Calculate average attribution score
        avg_score = sum(attr["score"] for attr in attribution_scores) / len(attribution_scores)
        
        # Apply a scaling factor to make it comparable to other methods
        return avg_score * 10  # Scale up for demonstration


class DataMarketplace:
    """
    Simulated data marketplace for evaluating data valuation methods.
    """
    
    def __init__(
        self, 
        valuation_methods: List[DataValuationMethod],
        data_chunks: List[Any] = None,
        transaction_cost: float = 0.1,
        save_history: bool = True
    ):
        """
        Initialize data marketplace.
        
        Args:
            valuation_methods: List of valuation methods to evaluate
            data_chunks: Initial set of data chunks in the marketplace
            transaction_cost: Cost for each transaction
            save_history: Whether to save history of prices and metrics
        """
        self.valuation_methods = {method.name: method for method in valuation_methods}
        self.data_chunks = data_chunks or []
        self.transaction_cost = transaction_cost
        self.save_history = save_history
        
        # Initialize price history
        self.price_history = defaultdict(lambda: defaultdict(list))
        
        # Transaction history
        self.transactions = []
        
        # Metrics history
        self.metrics_history = defaultdict(list)
    
    def add_chunks(self, chunks: List[Any]):
        """
        Add new chunks to the marketplace.
        
        Args:
            chunks: List of chunks to add
        """
        self.data_chunks.extend(chunks)
    
    def update_values(self, current_time: float = None):
        """
        Update the values of all chunks using all valuation methods.
        
        Args:
            current_time: Current timestamp (if None, use system time)
        """
        current_time = current_time if current_time is not None else time.time()
        
        for chunk in self.data_chunks:
            for method_name, method in self.valuation_methods.items():
                value = method.calculate_value(
                    data_chunk=chunk,
                    timestamp=current_time
                )
                
                # Update price in chunk for the specific method
                if not hasattr(chunk, 'prices'):
                    chunk.prices = {}
                
                chunk.prices[method_name] = value
                
                # Save price history if requested
                if self.save_history:
                    self.price_history[method_name][chunk.chunk_id].append({
                        "timestamp": current_time,
                        "price": value
                    })
    
    def simulate_transaction(
        self, 
        chunk: Any, 
        user_id: str,
        query: str = None,
        answer: str = None,
        attribution_score: float = None,
        user_feedback: float = None,
        method_name: str = None,
        timestamp: float = None
    ):
        """
        Simulate a transaction for a data chunk.
        
        Args:
            chunk: The data chunk involved in the transaction
            user_id: ID of the user making the transaction
            query: The query that led to this transaction
            answer: The answer generated using this chunk
            attribution_score: Attribution score for this chunk in the answer
            user_feedback: User feedback on the answer
            method_name: Valuation method to use for pricing (if None, use all)
            timestamp: Transaction timestamp
        """
        timestamp = timestamp if timestamp is not None else time.time()
        
        transaction = {
            "chunk_id": chunk.chunk_id,
            "contributor_id": chunk.contributor_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "attribution_score": attribution_score,
            "user_feedback": user_feedback,
            "prices": {}
        }
        
        # Record prices from all methods or just the specified one
        if method_name:
            method_names = [method_name]
        else:
            method_names = list(self.valuation_methods.keys())
        
        for name in method_names:
            if hasattr(chunk, 'prices') and name in chunk.prices:
                price = chunk.prices[name]
            else:
                price = self.valuation_methods[name].calculate_value(
                    data_chunk=chunk,
                    timestamp=timestamp
                )
                if not hasattr(chunk, 'prices'):
                    chunk.prices = {}
                chunk.prices[name] = price
            
            transaction["prices"][name] = price
        
        # Add transaction cost
        for name in method_names:
            transaction["prices"][name] += self.transaction_cost
        
        # Record transaction
        self.transactions.append(transaction)
        
        # Update chunk with new data
        chunk.retrieval_count += 1
        
        if attribution_score is not None:
            if not hasattr(chunk, 'attribution_scores'):
                chunk.attribution_scores = []
            
            chunk.attribution_scores.append({
                "query": query,
                "score": attribution_score,
                "timestamp": timestamp
            })
        
        if user_feedback is not None:
            if not hasattr(chunk, 'user_feedback'):
                chunk.user_feedback = []
            
            chunk.user_feedback.append({
                "score": user_feedback,
                "timestamp": timestamp
            })
    
    def calculate_metrics(self, ground_truth_qualities: Dict[str, float] = None):
        """
        Calculate metrics for evaluating valuation methods.
        
        Args:
            ground_truth_qualities: Dictionary mapping chunk IDs to ground truth qualities
        
        Returns:
            Dictionary of metrics
        """
        timestamp = time.time()
        metrics = {}
        
        # If we have ground truth qualities, calculate correlation with prices
        if ground_truth_qualities:
            for method_name in self.valuation_methods:
                chunk_qualities = []
                chunk_prices = []
                
                for chunk in self.data_chunks:
                    if chunk.chunk_id in ground_truth_qualities:
                        chunk_qualities.append(ground_truth_qualities[chunk.chunk_id])
                        
                        if hasattr(chunk, 'prices') and method_name in chunk.prices:
                            chunk_prices.append(chunk.prices[method_name])
                        else:
                            # Calculate price if not already set
                            price = self.valuation_methods[method_name].calculate_value(
                                data_chunk=chunk,
                                timestamp=timestamp
                            )
                            chunk_prices.append(price)
                
                if len(chunk_qualities) > 1:  # Need at least 2 points for correlation
                    pearson_corr, _ = pearsonr(chunk_qualities, chunk_prices)
                    spearman_corr, _ = spearmanr(chunk_qualities, chunk_prices)
                    
                    metrics[f"{method_name}_pearson_correlation"] = pearson_corr
                    metrics[f"{method_name}_spearman_correlation"] = spearman_corr
        
        # Calculate Gini coefficient for each method to measure reward distribution
        for method_name in self.valuation_methods:
            all_prices = []
            for chunk in self.data_chunks:
                if hasattr(chunk, 'prices') and method_name in chunk.prices:
                    all_prices.append(chunk.prices[method_name])
            
            if all_prices:
                gini = self._calculate_gini(all_prices)
                metrics[f"{method_name}_gini_coefficient"] = gini
        
        # Calculate price stability metrics
        for method_name in self.valuation_methods:
            price_stds = []
            
            for chunk_id, history in self.price_history[method_name].items():
                if len(history) > 1:
                    prices = [entry["price"] for entry in history]
                    price_stds.append(np.std(prices))
            
            if price_stds:
                metrics[f"{method_name}_price_volatility"] = np.mean(price_stds)
        
        # Calculate rewards by contributor
        contributor_rewards = defaultdict(lambda: defaultdict(float))
        
        for transaction in self.transactions:
            chunk_id = transaction["chunk_id"]
            contributor_id = transaction["contributor_id"]
            
            for method_name, price in transaction["prices"].items():
                contributor_rewards[method_name][contributor_id] += price
        
        # Calculate total rewards for each method
        for method_name in self.valuation_methods:
            rewards = list(contributor_rewards[method_name].values())
            if rewards:
                metrics[f"{method_name}_total_rewards"] = sum(rewards)
                metrics[f"{method_name}_rewards_gini"] = self._calculate_gini(rewards)
        
        # Save metrics history if requested
        if self.save_history:
            for key, value in metrics.items():
                self.metrics_history[key].append({
                    "timestamp": timestamp,
                    "value": value
                })
        
        return metrics
    
    def _calculate_gini(self, values: List[float]) -> float:
        """
        Calculate the Gini coefficient for a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Gini coefficient
        """
        if not values:
            return 0.0
        
        values = sorted(values)
        n = len(values)
        indices = np.arange(1, n + 1)
        
        return np.sum((2 * indices - n - 1) * values) / (n * np.sum(values))
    
    def save_results(self, output_dir: str):
        """
        Save marketplace results to disk.
        
        Args:
            output_dir: Directory to save results to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save transactions
        with open(os.path.join(output_dir, "transactions.json"), "w") as f:
            json.dump(self.transactions, f, indent=2)
        
        # Save price history
        price_history_flat = []
        for method_name, chunks in self.price_history.items():
            for chunk_id, history in chunks.items():
                for entry in history:
                    price_history_flat.append({
                        "method": method_name,
                        "chunk_id": chunk_id,
                        "timestamp": entry["timestamp"],
                        "price": entry["price"]
                    })
        
        with open(os.path.join(output_dir, "price_history.json"), "w") as f:
            json.dump(price_history_flat, f, indent=2)
        
        # Save metrics history
        metrics_history_flat = []
        for metric_name, history in self.metrics_history.items():
            for entry in history:
                metrics_history_flat.append({
                    "metric": metric_name,
                    "timestamp": entry["timestamp"],
                    "value": entry["value"]
                })
        
        with open(os.path.join(output_dir, "metrics_history.json"), "w") as f:
            json.dump(metrics_history_flat, f, indent=2)
        
        # Save current chunk prices
        chunk_prices = []
        for chunk in self.data_chunks:
            if hasattr(chunk, 'prices'):
                for method_name, price in chunk.prices.items():
                    chunk_prices.append({
                        "chunk_id": chunk.chunk_id,
                        "contributor_id": chunk.contributor_id,
                        "method": method_name,
                        "price": price,
                        "retrieval_count": chunk.retrieval_count,
                        "quality": chunk.quality if hasattr(chunk, 'quality') else None
                    })
        
        with open(os.path.join(output_dir, "chunk_prices.json"), "w") as f:
            json.dump(chunk_prices, f, indent=2)
        
        print(f"Saved marketplace results to {output_dir}")

if __name__ == "__main__":
    # Sample code to test the data valuation framework
    from utils.data_utils import create_synthetic_data
    
    # Create synthetic data for testing
    data_chunks, _ = create_synthetic_data(num_chunks=20)
    
    # Initialize valuation methods
    static_pricing = StaticPricing(price_per_token=0.01)
    popularity_pricing = PopularityBasedPricing(base_price=1.0, log_factor=2.0)
    dynamic_pricing = DynamicRAGValuation(
        attribution_weight=0.4,
        popularity_weight=0.2,
        feedback_weight=0.3,
        recency_weight=0.1,
        base_price=0.5
    )
    
    # Initialize marketplace
    marketplace = DataMarketplace(
        valuation_methods=[static_pricing, popularity_pricing, dynamic_pricing],
        data_chunks=data_chunks
    )
    
    # Simulate some transactions
    for i in range(50):
        chunk = np.random.choice(data_chunks)
        attribution_score = np.random.uniform(0, 1)
        user_feedback = np.random.uniform(0, 1)
        
        marketplace.simulate_transaction(
            chunk=chunk,
            user_id=f"user_{i % 5}",
            query=f"Sample query {i}",
            answer=f"Sample answer {i}",
            attribution_score=attribution_score,
            user_feedback=user_feedback
        )
    
    # Update all values
    marketplace.update_values()
    
    # Calculate metrics
    ground_truth = {chunk.chunk_id: chunk.quality for chunk in data_chunks}
    metrics = marketplace.calculate_metrics(ground_truth_qualities=ground_truth)
    
    # Print some results
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    marketplace.save_results("data_valuation_results")