"""
Evaluation module for model retrieval metrics.
This module implements metrics for evaluating the performance of model retrieval.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import umap
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Local imports
from config import EVAL_CONFIG, VIZ_CONFIG, LOG_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluation")

class RetrievalEvaluator:
    """
    Evaluator for model retrieval tasks.
    """
    
    def __init__(self, k_values=EVAL_CONFIG["k_values"]):
        self.k_values = k_values
        logger.info(f"Initialized RetrievalEvaluator with k_values={k_values}")
    
    def compute_similarity_matrix(self, embeddings):
        """
        Compute pairwise cosine similarity between embeddings.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, dim]
            
        Returns:
            Similarity matrix [num_models, num_models]
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.matmul(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def evaluate_knn_retrieval(self, embeddings, task_labels):
        """
        Evaluate k-nearest neighbor retrieval performance.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, dim]
            task_labels: List of task labels for each model
            
        Returns:
            Dictionary of metrics.
        """
        num_models = len(embeddings)
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Convert task labels to integer indices
        unique_tasks = list(set(task_labels))
        task_to_idx = {task: i for i, task in enumerate(unique_tasks)}
        task_indices = np.array([task_to_idx[task] for task in task_labels])
        
        # Initialize metrics
        metrics = {
            f"precision@{k}": [] for k in self.k_values
        }
        metrics.update({
            f"recall@{k}": [] for k in self.k_values
        })
        metrics.update({
            f"f1@{k}": [] for k in self.k_values
        })
        metrics["mAP"] = []
        
        # Compute metrics for each model
        for i in range(num_models):
            # Get similarities to other models
            similarities = similarity_matrix[i]
            
            # Set similarity to self to -inf
            similarities[i] = -np.inf
            
            # Get top-k indices
            top_k_max = max(self.k_values)
            top_k_indices = np.argsort(similarities)[::-1][:top_k_max]
            
            # Get ground truth label for query model
            query_task = task_indices[i]
            
            # Compute metrics for each k
            for k in self.k_values:
                # Get top-k retrieved models
                retrieved_indices = top_k_indices[:k]
                retrieved_tasks = task_indices[retrieved_indices]
                
                # Create binary relevance labels (1 if same task, 0 otherwise)
                relevance = (retrieved_tasks == query_task).astype(int)
                
                # Compute precision and recall
                if np.sum(relevance) > 0:
                    precision = np.sum(relevance) / k
                    recall = np.sum(relevance) / np.sum(task_indices == query_task)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                else:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                
                metrics[f"precision@{k}"].append(precision)
                metrics[f"recall@{k}"].append(recall)
                metrics[f"f1@{k}"].append(f1)
            
            # Compute mean Average Precision (mAP)
            # For each query, AP is the average of precision@k for each relevant item
            relevant_indices = np.where(task_indices == query_task)[0]
            relevant_indices = relevant_indices[relevant_indices != i]  # Exclude self
            
            if len(relevant_indices) > 0:
                # Get all retrieved models
                retrieved_indices = np.argsort(similarities)[::-1]
                
                # Compute precision at each relevant item
                precisions = []
                for j, idx in enumerate(retrieved_indices):
                    if idx in relevant_indices:
                        # This is a relevant item, compute precision at this point
                        rank = j + 1
                        precision_at_rank = np.sum((retrieved_indices[:rank] == idx) | 
                                                np.isin(retrieved_indices[:rank], relevant_indices)) / rank
                        precisions.append(precision_at_rank)
                
                if precisions:
                    ap = np.mean(precisions)
                else:
                    ap = 0.0
                    
                metrics["mAP"].append(ap)
            else:
                # No relevant items for this query
                metrics["mAP"].append(0.0)
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def evaluate_finetuning_transfer(self, embeddings, model_performance, finetune_budgets=EVAL_CONFIG["finetuning_budgets"]):
        """
        Simulate finetuning transfer and evaluate performance improvement.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, dim]
            model_performance: Numpy array of model performance values
            finetune_budgets: List of finetuning budget values to evaluate
            
        Returns:
            Dictionary of metrics for each budget.
        """
        num_models = len(embeddings)
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Initialize metrics
        metrics = {
            f"perf_improvement@{budget}": [] for budget in finetune_budgets
        }
        
        # Compute metrics for each model
        for i in range(num_models):
            # Get similarities to other models
            similarities = similarity_matrix[i]
            
            # Set similarity to self to -inf
            similarities[i] = -np.inf
            
            # Get base performance of the query model
            base_performance = model_performance[i]
            
            # Simulate finetuning for different budgets
            for budget in finetune_budgets:
                # Get top-k models as transfer sources
                k = budget  # Simplification: k proportional to budget
                top_k_indices = np.argsort(similarities)[::-1][:k]
                
                # Compute weighted performance improvement
                weights = similarities[top_k_indices]
                weights = np.exp(weights) / np.sum(np.exp(weights))  # Softmax normalization
                
                # Simulate improvement based on weighted average of performances
                # and similarity to the source models
                top_k_performances = model_performance[top_k_indices]
                performance_delta = np.sum(weights * (top_k_performances - base_performance))
                
                # Apply a discount factor based on budget (diminishing returns)
                discount = np.log(budget + 1) / np.log(max(finetune_budgets) + 1)
                
                # Final improvement is positive delta scaled by budget discount
                improvement = max(0, performance_delta) * discount
                metrics[f"perf_improvement@{budget}"].append(improvement)
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def evaluate_symmetry_robustness(self, original_embeddings, transformed_embeddings):
        """
        Evaluate robustness to symmetry transformations.
        
        Args:
            original_embeddings: Numpy array of original embeddings [num_models, dim]
            transformed_embeddings: Numpy array of transformed embeddings [num_models, dim]
            
        Returns:
            Dictionary of metrics.
        """
        num_models = len(original_embeddings)
        
        # Compute cosine similarities between original and transformed embeddings
        similarities = []
        for i in range(num_models):
            # Normalize embeddings
            orig_emb = original_embeddings[i] / (np.linalg.norm(original_embeddings[i]) + 1e-8)
            trans_emb = transformed_embeddings[i] / (np.linalg.norm(transformed_embeddings[i]) + 1e-8)
            
            # Compute similarity
            sim = np.dot(orig_emb, trans_emb)
            similarities.append(sim)
        
        # Compute metrics
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)
        
        # Compute embedding distance
        distances = []
        for i in range(num_models):
            dist = np.linalg.norm(original_embeddings[i] - transformed_embeddings[i])
            distances.append(dist)
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        max_distance = np.max(distances)
        
        # Return metrics
        return {
            "mean_similarity": mean_similarity,
            "std_similarity": std_similarity,
            "min_similarity": min_similarity,
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "max_distance": max_distance
        }
    
    def evaluate_clustering_quality(self, embeddings, task_labels):
        """
        Evaluate clustering quality of the embeddings.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, dim]
            task_labels: List of task labels for each model
            
        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score
        
        # Convert task labels to integer indices
        unique_tasks = list(set(task_labels))
        task_to_idx = {task: i for i, task in enumerate(unique_tasks)}
        task_indices = np.array([task_to_idx[task] for task in task_labels])
        
        try:
            # Compute silhouette score
            silhouette = silhouette_score(embeddings, task_indices)
            
            # Compute Davies-Bouldin index
            davies_bouldin = davies_bouldin_score(embeddings, task_indices)
            
            return {
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin
            }
        except:
            logger.warning("Failed to compute clustering metrics, returning zeros")
            return {
                "silhouette_score": 0.0,
                "davies_bouldin_score": 0.0
            }
    
    def visualize_embeddings(self, embeddings, task_labels, title="Embedding Visualization", 
                           method='tsne', save_path=None):
        """
        Visualize embeddings in 2D space.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, dim]
            task_labels: List of task labels for each model
            title: Plot title
            method: Dimensionality reduction method ('tsne' or 'umap')
            save_path: Path to save the visualization
            
        Returns:
            Figure object.
        """
        # Convert embeddings to 2D
        if method == 'tsne':
            tsne = TSNE(
                n_components=2,
                perplexity=VIZ_CONFIG["embedding_vis"]["perplexity"],
                random_state=42
            )
            embeddings_2d = tsne.fit_transform(embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=VIZ_CONFIG["embedding_vis"]["n_neighbors"],
                min_dist=0.1,
                random_state=42
            )
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create figure
        plt.figure(figsize=VIZ_CONFIG["figure_size"])
        
        # Convert task labels to integer indices
        unique_tasks = list(set(task_labels))
        task_to_idx = {task: i for i, task in enumerate(unique_tasks)}
        task_indices = np.array([task_to_idx[task] for task in task_labels])
        
        # Create scatter plot
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=task_indices,
            cmap='tab10',
            alpha=0.8,
            s=100
        )
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=plt.cm.tab10(i), markersize=10, label=task)
                          for i, task in enumerate(unique_tasks)]
        plt.legend(handles=legend_elements, title="Tasks", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add title and labels
        plt.title(title)
        plt.xlabel(f"{method.upper()} Dimension 1")
        plt.ylabel(f"{method.upper()} Dimension 2")
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
        
        return plt.gcf()
    
    def visualize_similarity_matrix(self, embeddings, task_labels, title="Similarity Matrix", 
                                   save_path=None):
        """
        Visualize the similarity matrix of embeddings.
        
        Args:
            embeddings: Numpy array of embeddings [num_models, dim]
            task_labels: List of task labels for each model
            title: Plot title
            save_path: Path to save the visualization
            
        Returns:
            Figure object.
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Convert task labels to integer indices
        unique_tasks = list(set(task_labels))
        task_to_idx = {task: i for i, task in enumerate(unique_tasks)}
        task_indices = np.array([task_to_idx[task] for task in task_labels])
        
        # Sort by task
        sort_indices = np.argsort(task_indices)
        sorted_similarity = similarity_matrix[sort_indices][:, sort_indices]
        sorted_tasks = [task_labels[i] for i in sort_indices]
        
        # Create figure
        plt.figure(figsize=VIZ_CONFIG["figure_size"])
        
        # Create heatmap
        sns.heatmap(
            sorted_similarity,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=False,
            yticklabels=False
        )
        
        # Add task boundary lines
        task_boundaries = []
        prev_task = sorted_tasks[0]
        for i, task in enumerate(sorted_tasks[1:], 1):
            if task != prev_task:
                task_boundaries.append(i)
                prev_task = task
        
        for boundary in task_boundaries:
            plt.axhline(y=boundary, color='k', linestyle='-', linewidth=1)
            plt.axvline(x=boundary, color='k', linestyle='-', linewidth=1)
        
        # Add title and labels
        plt.title(title)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
        
        return plt.gcf()
    
    def generate_retrieval_report(self, model_ids, embeddings, task_labels, model_performance, 
                                metadata, example_queries=5, top_k=5):
        """
        Generate a detailed retrieval report with example queries.
        
        Args:
            model_ids: List of model IDs
            embeddings: Numpy array of embeddings [num_models, dim]
            task_labels: List of task labels for each model
            model_performance: Numpy array of model performance values
            metadata: Dictionary mapping model IDs to metadata
            example_queries: Number of example queries to include
            top_k: Number of top results to show for each query
            
        Returns:
            Report as a string.
        """
        num_models = len(embeddings)
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Convert task labels to integer indices
        unique_tasks = list(set(task_labels))
        task_to_idx = {task: i for i, task in enumerate(unique_tasks)}
        task_indices = np.array([task_to_idx[task] for task in task_labels])
        
        # Select random models for example queries
        query_indices = np.random.choice(num_models, size=example_queries, replace=False)
        
        # Initialize report
        report = "# Model Retrieval Report\n\n"
        
        # Add overall statistics
        report += "## Overall Statistics\n\n"
        report += f"Total models: {num_models}\n"
        report += f"Unique tasks: {len(unique_tasks)}\n"
        report += f"Task distribution:\n"
        
        task_counts = defaultdict(int)
        for task in task_labels:
            task_counts[task] += 1
        
        for task, count in task_counts.items():
            report += f"- {task}: {count} models\n"
        
        report += "\n"
        
        # Add example queries
        report += "## Example Queries\n\n"
        
        for i, query_idx in enumerate(query_indices):
            query_id = model_ids[query_idx]
            query_task = task_labels[query_idx]
            query_perf = model_performance[query_idx]
            
            report += f"### Query {i+1}: {query_id}\n\n"
            report += f"Task: {query_task}\n"
            report += f"Performance: {query_perf:.4f}\n"
            
            # Get metadata for query model
            if query_id in metadata:
                query_meta = metadata[query_id]
                report += "Metadata:\n"
                for key, value in query_meta.items():
                    if key not in ["id", "path", "performance"]:
                        report += f"- {key}: {value}\n"
            
            report += "\n"
            
            # Get top-k results
            similarities = similarity_matrix[query_idx]
            similarities[query_idx] = -np.inf  # Exclude self
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            
            report += f"Top {top_k} Results:\n\n"
            report += "| Model ID | Task | Performance | Similarity |\n"
            report += "|----------|------|-------------|------------|\n"
            
            for idx in top_k_indices:
                model_id = model_ids[idx]
                task = task_labels[idx]
                perf = model_performance[idx]
                sim = similarities[idx]
                
                report += f"| {model_id} | {task} | {perf:.4f} | {sim:.4f} |\n"
            
            report += "\n"
        
        return report


class CrossValidationEvaluator:
    """
    Evaluator for cross-validation experiments.
    """
    
    def __init__(self, num_folds=EVAL_CONFIG["num_folds"]):
        self.num_folds = num_folds
        self.retrieval_evaluator = RetrievalEvaluator()
        logger.info(f"Initialized CrossValidationEvaluator with num_folds={num_folds}")
    
    def create_folds(self, model_ids, task_labels, stratify=True):
        """
        Create cross-validation folds.
        
        Args:
            model_ids: List of model IDs
            task_labels: List of task labels for each model
            stratify: Whether to stratify folds by task
            
        Returns:
            List of (train_indices, test_indices) tuples for each fold.
        """
        num_models = len(model_ids)
        
        if stratify:
            # Stratify by task
            task_to_indices = defaultdict(list)
            for i, task in enumerate(task_labels):
                task_to_indices[task].append(i)
            
            # Create folds
            folds = []
            for fold in range(self.num_folds):
                train_indices = []
                test_indices = []
                
                for task, indices in task_to_indices.items():
                    # Shuffle indices
                    np.random.shuffle(indices)
                    
                    # Split indices for this task
                    num_test = max(1, len(indices) // self.num_folds)
                    start_idx = fold * num_test
                    end_idx = start_idx + num_test
                    
                    # Handle last fold
                    if fold == self.num_folds - 1:
                        end_idx = len(indices)
                    
                    # Split indices
                    fold_test_indices = indices[start_idx:end_idx]
                    fold_train_indices = [idx for idx in indices if idx not in fold_test_indices]
                    
                    # Add to fold
                    train_indices.extend(fold_train_indices)
                    test_indices.extend(fold_test_indices)
                
                folds.append((train_indices, test_indices))
        else:
            # Random splits
            indices = np.arange(num_models)
            np.random.shuffle(indices)
            
            folds = []
            for fold in range(self.num_folds):
                test_size = num_models // self.num_folds
                start_idx = fold * test_size
                end_idx = start_idx + test_size
                
                # Handle last fold
                if fold == self.num_folds - 1:
                    end_idx = num_models
                
                test_indices = indices[start_idx:end_idx]
                train_indices = [idx for idx in indices if idx not in test_indices]
                
                folds.append((train_indices, test_indices))
        
        return folds
    
    def evaluate_model(self, encoder, model_ids, model_data, task_labels, model_performance, metadata=None):
        """
        Evaluate a model using cross-validation.
        
        Args:
            encoder: Model encoder to evaluate
            model_ids: List of model IDs
            model_data: List of model data (weights or graphs)
            task_labels: List of task labels for each model
            model_performance: Numpy array of model performance values
            metadata: Optional dictionary of model metadata
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Create folds
        folds = self.create_folds(model_ids, task_labels)
        
        # Initialize metrics
        all_metrics = []
        
        # Evaluate each fold
        for fold, (train_indices, test_indices) in enumerate(folds):
            logger.info(f"Evaluating fold {fold+1}/{self.num_folds}")
            
            # Get training and test data
            train_data = [model_data[i] for i in train_indices]
            test_data = [model_data[i] for i in test_indices]
            
            # Get test labels and IDs
            test_ids = [model_ids[i] for i in test_indices]
            test_tasks = [task_labels[i] for i in test_indices]
            test_perf = model_performance[test_indices]
            
            # Encode models
            train_embeddings = np.array([encoder.encode(data) for data in train_data])
            test_embeddings = np.array([encoder.encode(data) for data in test_data])
            
            # Evaluate retrieval
            retrieval_metrics = self.retrieval_evaluator.evaluate_knn_retrieval(
                np.concatenate([train_embeddings, test_embeddings], axis=0),
                task_labels=task_labels[train_indices] + task_labels[test_indices]
            )
            
            # Evaluate transfer
            transfer_metrics = self.retrieval_evaluator.evaluate_finetuning_transfer(
                np.concatenate([train_embeddings, test_embeddings], axis=0),
                np.concatenate([model_performance[train_indices], test_perf], axis=0)
            )
            
            # Evaluate clustering
            clustering_metrics = self.retrieval_evaluator.evaluate_clustering_quality(
                np.concatenate([train_embeddings, test_embeddings], axis=0),
                task_labels=task_labels[train_indices] + task_labels[test_indices]
            )
            
            # Combine metrics
            fold_metrics = {
                **retrieval_metrics,
                **transfer_metrics,
                **clustering_metrics
            }
            
            all_metrics.append(fold_metrics)
        
        # Compute average metrics across folds
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([metrics[key] for metrics in all_metrics])
            avg_metrics[f"{key}_std"] = np.std([metrics[key] for metrics in all_metrics])
        
        return avg_metrics
    
    def compare_models(self, encoder_dict, model_ids, model_data, task_labels, model_performance, metadata=None):
        """
        Compare multiple model encoders.
        
        Args:
            encoder_dict: Dictionary mapping encoder names to encoder objects
            model_ids: List of model IDs
            model_data: List of model data (weights or graphs)
            task_labels: List of task labels for each model
            model_performance: Numpy array of model performance values
            metadata: Optional dictionary of model metadata
            
        Returns:
            DataFrame with comparison results.
        """
        # Initialize results
        results = {}
        
        # Evaluate each encoder
        for name, encoder in encoder_dict.items():
            logger.info(f"Evaluating encoder: {name}")
            metrics = self.evaluate_model(encoder, model_ids, model_data, task_labels, model_performance, metadata)
            results[name] = metrics
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df


# Test code
if __name__ == "__main__":
    # Create dummy data for testing
    num_models = 100
    embedding_dim = 32
    
    # Create dummy embeddings
    embeddings = np.random.randn(num_models, embedding_dim)
    
    # Create dummy task labels (3 tasks)
    task_labels = []
    for i in range(num_models):
        if i < 30:
            task_labels.append("classification")
        elif i < 70:
            task_labels.append("detection")
        else:
            task_labels.append("segmentation")
    
    # Create dummy performance values
    performance = np.random.uniform(0.5, 0.95, size=num_models)
    
    # Create evaluator
    evaluator = RetrievalEvaluator()
    
    # Test kNN retrieval evaluation
    print("Testing kNN retrieval evaluation...")
    retrieval_metrics = evaluator.evaluate_knn_retrieval(embeddings, task_labels)
    print(retrieval_metrics)
    
    # Test finetuning transfer evaluation
    print("\nTesting finetuning transfer evaluation...")
    transfer_metrics = evaluator.evaluate_finetuning_transfer(embeddings, performance)
    print(transfer_metrics)
    
    # Test symmetry robustness evaluation
    print("\nTesting symmetry robustness evaluation...")
    transformed_embeddings = embeddings + np.random.randn(num_models, embedding_dim) * 0.01
    robustness_metrics = evaluator.evaluate_symmetry_robustness(embeddings, transformed_embeddings)
    print(robustness_metrics)
    
    # Test clustering quality evaluation
    print("\nTesting clustering quality evaluation...")
    clustering_metrics = evaluator.evaluate_clustering_quality(embeddings, task_labels)
    print(clustering_metrics)
    
    # Test visualization
    print("\nTesting visualization...")
    try:
        fig = evaluator.visualize_embeddings(embeddings, task_labels, save_path="test_embeddings.png")
        print("Saved embedding visualization to test_embeddings.png")
        
        fig = evaluator.visualize_similarity_matrix(embeddings, task_labels, save_path="test_similarity.png")
        print("Saved similarity matrix visualization to test_similarity.png")
    except:
        print("Visualization failed, likely due to missing dependencies or display")
    
    # Test retrieval report
    print("\nTesting retrieval report...")
    model_ids = [f"model_{i}" for i in range(num_models)]
    metadata = {
        model_id: {
            "architecture": f"arch_{i % 3}",
            "dataset": f"dataset_{i % 5}",
            "parameters": 1000000 + i * 10000
        } for i, model_id in enumerate(model_ids)
    }
    
    report = evaluator.generate_retrieval_report(
        model_ids, embeddings, task_labels, performance, metadata, example_queries=2
    )
    print(report[:500] + "...\n[Report truncated]")
    
    # Test cross-validation
    print("\nTesting cross-validation...")
    
    # Mock encoder for testing
    class MockEncoder:
        def encode(self, data):
            return np.random.randn(32)  # Random encoding
    
    cv_evaluator = CrossValidationEvaluator(num_folds=3)
    
    # Mock model data
    model_data = [{"data": i} for i in range(num_models)]
    
    # Create folds
    folds = cv_evaluator.create_folds(model_ids, task_labels)
    print(f"Created {len(folds)} folds")
    
    # Test model evaluation
    encoder = MockEncoder()
    metrics = cv_evaluator.evaluate_model(encoder, model_ids, model_data, task_labels, performance)
    print("Cross-validation metrics:")
    print(metrics)
    
    # Test model comparison
    encoder_dict = {
        "encoder1": MockEncoder(),
        "encoder2": MockEncoder()
    }
    
    comparison = cv_evaluator.compare_models(encoder_dict, model_ids, model_data, task_labels, performance)
    print("\nModel comparison:")
    print(comparison)