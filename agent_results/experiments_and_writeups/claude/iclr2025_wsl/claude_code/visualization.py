"""
Visualization module for model zoo retrieval experiment.
This module provides functions for visualizing experimental results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from pathlib import Path

# Local imports
from config import VIZ_CONFIG, LOG_CONFIG, FIGURES_DIR

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization")

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('tab10')
sns.set_context("paper", font_scale=1.5)

# Ensure the figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

class ResultsVisualizer:
    """
    Visualizer for experimental results.
    """
    
    def __init__(self, figures_dir=FIGURES_DIR, save_format=VIZ_CONFIG["save_format"], dpi=VIZ_CONFIG["dpi"]):
        self.figures_dir = Path(figures_dir)
        self.save_format = save_format
        self.dpi = dpi
        logger.info(f"Initialized ResultsVisualizer with figures_dir={figures_dir}")
    
    def plot_training_history(self, history, title="Training History", save_as="training_history"):
        """
        Plot training history curves.
        
        Args:
            history: Dictionary with training history.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Tuple of figure handles.
        """
        figures = []
        
        # Plot loss curves
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        ax.plot(history["train_loss"], label="Train Loss", marker="o")
        ax.plot(history["val_loss"], label="Validation Loss", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        save_path = self.figures_dir / f"{save_as}_loss.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved loss plot to {save_path}")
        figures.append(fig)
        
        # Plot contrastive loss and metric loss separately
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        ax.plot(history["train_contrastive_loss"], label="Train Contrastive Loss", marker="o")
        ax.plot(history["val_contrastive_loss"], label="Val Contrastive Loss", marker="s")
        ax.plot(history["train_metric_loss"], label="Train Metric Loss", marker="^")
        ax.plot(history["val_metric_loss"], label="Val Metric Loss", marker="x")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Component")
        ax.set_title(f"{title} - Loss Components")
        ax.legend()
        ax.grid(True)
        
        save_path = self.figures_dir / f"{save_as}_components.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved loss components plot to {save_path}")
        figures.append(fig)
        
        # Return figures
        return figures
    
    def plot_retrieval_metrics(self, metrics_dict, title="Retrieval Performance", save_as="retrieval_metrics"):
        """
        Plot retrieval metrics comparison.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Tuple of figure handles.
        """
        figures = []
        
        # Extract retrieval metrics
        precision_metrics = {k: v for k, v in metrics_dict.items() if k.startswith("precision@")}
        recall_metrics = {k: v for k, v in metrics_dict.items() if k.startswith("recall@")}
        f1_metrics = {k: v for k, v in metrics_dict.items() if k.startswith("f1@")}
        
        # Extract k values
        k_values = sorted([int(k.split("@")[1]) for k in precision_metrics.keys()])
        
        # Create DataFrame for plotting
        data = []
        for model_name, model_metrics in metrics_dict.items():
            for k in k_values:
                data.append({
                    "Model": model_name,
                    "k": k,
                    "Precision": model_metrics.get(f"precision@{k}", 0),
                    "Recall": model_metrics.get(f"recall@{k}", 0),
                    "F1": model_metrics.get(f"f1@{k}", 0)
                })
        
        df = pd.DataFrame(data)
        
        # Plot precision@k
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        sns.barplot(data=df, x="k", y="Precision", hue="Model", ax=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("Precision@k")
        ax.set_title(f"{title} - Precision@k")
        ax.legend(title="Model")
        
        save_path = self.figures_dir / f"{save_as}_precision.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved precision plot to {save_path}")
        figures.append(fig)
        
        # Plot recall@k
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        sns.barplot(data=df, x="k", y="Recall", hue="Model", ax=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k")
        ax.set_title(f"{title} - Recall@k")
        ax.legend(title="Model")
        
        save_path = self.figures_dir / f"{save_as}_recall.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved recall plot to {save_path}")
        figures.append(fig)
        
        # Plot F1@k
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        sns.barplot(data=df, x="k", y="F1", hue="Model", ax=ax)
        ax.set_xlabel("k")
        ax.set_ylabel("F1@k")
        ax.set_title(f"{title} - F1@k")
        ax.legend(title="Model")
        
        save_path = self.figures_dir / f"{save_as}_f1.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved F1 plot to {save_path}")
        figures.append(fig)
        
        # Plot mAP
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        mAP_data = [(model, metrics.get("mAP", 0)) 
                    for model, metrics in metrics_dict.items()]
        mAP_df = pd.DataFrame(mAP_data, columns=["Model", "mAP"])
        sns.barplot(data=mAP_df, x="Model", y="mAP", ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("mAP")
        ax.set_title(f"{title} - Mean Average Precision")
        
        save_path = self.figures_dir / f"{save_as}_map.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved mAP plot to {save_path}")
        figures.append(fig)
        
        # Return figures
        return figures
    
    def plot_transfer_performance(self, metrics_dict, title="Transfer Performance", save_as="transfer_performance"):
        """
        Plot transfer learning performance.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Figure handle.
        """
        # Extract transfer metrics
        transfer_metrics = {}
        for model_name, model_metrics in metrics_dict.items():
            transfer_metrics[model_name] = {
                k: v for k, v in model_metrics.items() 
                if k.startswith("perf_improvement@")
            }
        
        # Extract budget values
        budget_values = sorted([int(k.split("@")[1]) 
                              for k in list(transfer_metrics.values())[0].keys()])
        
        # Create DataFrame for plotting
        data = []
        for model_name, model_metrics in transfer_metrics.items():
            for budget in budget_values:
                data.append({
                    "Model": model_name,
                    "Budget": budget,
                    "Performance Improvement": model_metrics.get(f"perf_improvement@{budget}", 0)
                })
        
        df = pd.DataFrame(data)
        
        # Plot performance improvement
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        sns.lineplot(data=df, x="Budget", y="Performance Improvement", hue="Model", 
                   marker="o", ax=ax)
        ax.set_xlabel("Finetuning Budget")
        ax.set_ylabel("Performance Improvement")
        ax.set_title(title)
        ax.legend(title="Model")
        
        save_path = self.figures_dir / f"{save_as}.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved transfer performance plot to {save_path}")
        
        return fig
    
    def plot_symmetry_robustness(self, metrics_dict, title="Symmetry Robustness", save_as="symmetry_robustness"):
        """
        Plot symmetry robustness metrics.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Tuple of figure handles.
        """
        figures = []
        
        # Extract symmetry metrics
        symmetry_metrics = {}
        for model_name, model_metrics in metrics_dict.items():
            symmetry_metrics[model_name] = {
                k: v for k, v in model_metrics.items() 
                if k in ["mean_similarity", "min_similarity", "mean_distance", "max_distance"]
            }
        
        # Create DataFrame for similarity metrics
        sim_data = []
        for model_name, model_metrics in symmetry_metrics.items():
            sim_data.append({
                "Model": model_name,
                "Mean Similarity": model_metrics.get("mean_similarity", 0),
                "Min Similarity": model_metrics.get("min_similarity", 0)
            })
        
        sim_df = pd.DataFrame(sim_data)
        
        # Plot similarity metrics
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        
        # Reshape data for seaborn
        sim_df_melt = pd.melt(sim_df, id_vars=["Model"], 
                            value_vars=["Mean Similarity", "Min Similarity"],
                            var_name="Metric", value_name="Value")
        
        sns.barplot(data=sim_df_melt, x="Model", y="Value", hue="Metric", ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Similarity")
        ax.set_title(f"{title} - Similarity Metrics")
        ax.legend(title="Metric")
        
        save_path = self.figures_dir / f"{save_as}_similarity.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved symmetry similarity plot to {save_path}")
        figures.append(fig)
        
        # Create DataFrame for distance metrics
        dist_data = []
        for model_name, model_metrics in symmetry_metrics.items():
            dist_data.append({
                "Model": model_name,
                "Mean Distance": model_metrics.get("mean_distance", 0),
                "Max Distance": model_metrics.get("max_distance", 0)
            })
        
        dist_df = pd.DataFrame(dist_data)
        
        # Plot distance metrics
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        
        # Reshape data for seaborn
        dist_df_melt = pd.melt(dist_df, id_vars=["Model"], 
                              value_vars=["Mean Distance", "Max Distance"],
                              var_name="Metric", value_name="Value")
        
        sns.barplot(data=dist_df_melt, x="Model", y="Value", hue="Metric", ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Distance")
        ax.set_title(f"{title} - Distance Metrics")
        ax.legend(title="Metric")
        
        save_path = self.figures_dir / f"{save_as}_distance.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved symmetry distance plot to {save_path}")
        figures.append(fig)
        
        # Return figures
        return figures
    
    def plot_clustering_metrics(self, metrics_dict, title="Clustering Quality", save_as="clustering_metrics"):
        """
        Plot clustering quality metrics.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Figure handle.
        """
        # Extract clustering metrics
        clustering_metrics = {}
        for model_name, model_metrics in metrics_dict.items():
            clustering_metrics[model_name] = {
                k: v for k, v in model_metrics.items() 
                if k in ["silhouette_score", "davies_bouldin_score"]
            }
        
        # Create DataFrame for plotting
        data = []
        for model_name, model_metrics in clustering_metrics.items():
            data.append({
                "Model": model_name,
                "Silhouette Score": model_metrics.get("silhouette_score", 0),
                "Davies-Bouldin Score": model_metrics.get("davies_bouldin_score", 0)
            })
        
        df = pd.DataFrame(data)
        
        # Plot clustering metrics
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
        
        # Reshape data for seaborn
        df_melt = pd.melt(df, id_vars=["Model"], 
                         value_vars=["Silhouette Score", "Davies-Bouldin Score"],
                         var_name="Metric", value_name="Value")
        
        sns.barplot(data=df_melt, x="Model", y="Value", hue="Metric", ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(title="Metric")
        
        save_path = self.figures_dir / f"{save_as}.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved clustering metrics plot to {save_path}")
        
        return fig
    
    def plot_overall_comparison(self, metrics_dict, title="Overall Model Comparison", save_as="overall_comparison"):
        """
        Create radar/spider chart with overall model comparison.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Figure handle.
        """
        # Select key metrics for overall comparison
        key_metrics = [
            "precision@1", "precision@5", "mAP", 
            "perf_improvement@50", "mean_similarity", "silhouette_score"
        ]
        
        # Create DataFrame with normalized metrics
        data = {}
        for metric in key_metrics:
            values = [model_metrics.get(metric, 0) for model_metrics in metrics_dict.values()]
            # Normalize between 0 and 1
            if max(values) == min(values):
                normalized = [0.5 for _ in values]
            else:
                if metric == "davies_bouldin_score":  # Lower is better
                    normalized = [1 - (v - min(values)) / (max(values) - min(values)) for v in values]
                else:  # Higher is better
                    normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
            
            data[metric] = normalized
        
        # Create DataFrame
        df = pd.DataFrame(data, index=metrics_dict.keys())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"], subplot_kw=dict(polar=True))
        
        # Get number of metrics
        N = len(key_metrics)
        
        # Set angles for each metric
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for model_name, row in df.iterrows():
            values = row.values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Set metric labels
        metric_labels = [m.replace("@", " @ k=") for m in key_metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        
        # Set y ticks
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(title, size=15, y=1.1)
        
        save_path = self.figures_dir / f"{save_as}.{self.save_format}"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved overall comparison plot to {save_path}")
        
        return fig
    
    def plot_embedding_visualization(self, embeddings_dict, labels, title="Embedding Visualization", 
                                    save_as="embedding_visualization"):
        """
        Plot 2D visualization of embeddings using t-SNE.
        
        Args:
            embeddings_dict: Dictionary mapping model names to embeddings.
            labels: List of task labels for each model.
            title: Plot title.
            save_as: Base filename to save as.
            
        Returns:
            Dictionary mapping model names to figure handles.
        """
        from sklearn.manifold import TSNE
        
        figures = {}
        
        # Process each model's embeddings
        for model_name, embeddings in embeddings_dict.items():
            # Create t-SNE visualization
            tsne = TSNE(
                n_components=2,
                perplexity=VIZ_CONFIG["embedding_vis"]["perplexity"],
                random_state=42
            )
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create figure
            fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"])
            
            # Get unique labels
            unique_labels = list(set(labels))
            
            # Plot each class
            for label in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    label=label,
                    alpha=0.7
                )
            
            ax.set_title(f"{title} - {model_name}")
            ax.legend(title="Task")
            
            save_path = self.figures_dir / f"{save_as}_{model_name}.{self.save_format}"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved embedding visualization for {model_name} to {save_path}")
            
            figures[model_name] = fig
        
        return figures
    
    def create_figures_for_results(self, results_data):
        """
        Create all figures for the experimental results.
        
        Args:
            results_data: Dictionary with experimental results.
            
        Returns:
            List of generated figure paths.
        """
        figure_paths = []
        
        # Plot training history for the proposed model
        if "training_history" in results_data:
            history_figures = self.plot_training_history(
                results_data["training_history"],
                title="Training History - Permutation-Equivariant GNN",
                save_as="training_history"
            )
            figure_paths.extend([f"training_history_loss.{self.save_format}", 
                               f"training_history_components.{self.save_format}"])
        
        # Plot retrieval metrics
        if "retrieval_metrics" in results_data:
            retrieval_figures = self.plot_retrieval_metrics(
                results_data["retrieval_metrics"],
                title="Retrieval Performance",
                save_as="retrieval_metrics"
            )
            figure_paths.extend([f"retrieval_metrics_precision.{self.save_format}", 
                               f"retrieval_metrics_recall.{self.save_format}",
                               f"retrieval_metrics_f1.{self.save_format}",
                               f"retrieval_metrics_map.{self.save_format}"])
        
        # Plot transfer performance
        if "transfer_metrics" in results_data:
            transfer_figure = self.plot_transfer_performance(
                results_data["transfer_metrics"],
                title="Transfer Learning Performance",
                save_as="transfer_performance"
            )
            figure_paths.append(f"transfer_performance.{self.save_format}")
        
        # Plot symmetry robustness
        if "symmetry_metrics" in results_data:
            symmetry_figures = self.plot_symmetry_robustness(
                results_data["symmetry_metrics"],
                title="Symmetry Robustness",
                save_as="symmetry_robustness"
            )
            figure_paths.extend([f"symmetry_robustness_similarity.{self.save_format}", 
                               f"symmetry_robustness_distance.{self.save_format}"])
        
        # Plot clustering metrics
        if "clustering_metrics" in results_data:
            clustering_figure = self.plot_clustering_metrics(
                results_data["clustering_metrics"],
                title="Clustering Quality",
                save_as="clustering_metrics"
            )
            figure_paths.append(f"clustering_metrics.{self.save_format}")
        
        # Plot overall comparison
        if "overall_metrics" in results_data:
            overall_figure = self.plot_overall_comparison(
                results_data["overall_metrics"],
                title="Overall Model Comparison",
                save_as="overall_comparison"
            )
            figure_paths.append(f"overall_comparison.{self.save_format}")
        
        # Plot embedding visualizations
        if "embeddings" in results_data and "labels" in results_data:
            embedding_figures = self.plot_embedding_visualization(
                results_data["embeddings"],
                results_data["labels"],
                title="Embedding Visualization",
                save_as="embedding_visualization"
            )
            for model_name in results_data["embeddings"].keys():
                figure_paths.append(f"embedding_visualization_{model_name}.{self.save_format}")
        
        return figure_paths
    
    def generate_tables_for_results(self, results_data):
        """
        Generate tables for the experimental results in markdown format.
        
        Args:
            results_data: Dictionary with experimental results.
            
        Returns:
            Dictionary mapping table names to markdown table strings.
        """
        tables = {}
        
        # Retrieval metrics table
        if "retrieval_metrics" in results_data:
            # Select metrics to include
            metrics = ["precision@1", "precision@5", "precision@10", "mAP"]
            
            # Create table header
            header = "| Model | " + " | ".join(metrics) + " |\n"
            separator = "|------|" + "|".join(["---" for _ in metrics]) + "|\n"
            
            # Create table rows
            rows = []
            for model_name, model_metrics in results_data["retrieval_metrics"].items():
                row = f"| {model_name} | "
                row += " | ".join([f"{model_metrics.get(metric, 0):.4f}" for metric in metrics])
                row += " |\n"
                rows.append(row)
            
            # Combine into table
            table = header + separator + "".join(rows)
            tables["retrieval_metrics"] = table
        
        # Transfer metrics table
        if "transfer_metrics" in results_data:
            # Select metrics to include
            metrics = ["perf_improvement@10", "perf_improvement@50", "perf_improvement@100"]
            
            # Create table header
            header = "| Model | " + " | ".join([m.replace("perf_improvement@", "Budget ") for m in metrics]) + " |\n"
            separator = "|------|" + "|".join(["---" for _ in metrics]) + "|\n"
            
            # Create table rows
            rows = []
            for model_name, model_metrics in results_data["transfer_metrics"].items():
                row = f"| {model_name} | "
                row += " | ".join([f"{model_metrics.get(metric, 0):.4f}" for metric in metrics])
                row += " |\n"
                rows.append(row)
            
            # Combine into table
            table = header + separator + "".join(rows)
            tables["transfer_metrics"] = table
        
        # Symmetry metrics table
        if "symmetry_metrics" in results_data:
            # Select metrics to include
            metrics = ["mean_similarity", "min_similarity", "mean_distance", "max_distance"]
            
            # Create table header
            header = "| Model | " + " | ".join([m.replace("_", " ").title() for m in metrics]) + " |\n"
            separator = "|------|" + "|".join(["---" for _ in metrics]) + "|\n"
            
            # Create table rows
            rows = []
            for model_name, model_metrics in results_data["symmetry_metrics"].items():
                row = f"| {model_name} | "
                row += " | ".join([f"{model_metrics.get(metric, 0):.4f}" for metric in metrics])
                row += " |\n"
                rows.append(row)
            
            # Combine into table
            table = header + separator + "".join(rows)
            tables["symmetry_metrics"] = table
        
        # Clustering metrics table
        if "clustering_metrics" in results_data:
            # Select metrics to include
            metrics = ["silhouette_score", "davies_bouldin_score"]
            
            # Create table header
            header = "| Model | " + " | ".join([m.replace("_", " ").title() for m in metrics]) + " |\n"
            separator = "|------|" + "|".join(["---" for _ in metrics]) + "|\n"
            
            # Create table rows
            rows = []
            for model_name, model_metrics in results_data["clustering_metrics"].items():
                row = f"| {model_name} | "
                row += " | ".join([f"{model_metrics.get(metric, 0):.4f}" for metric in metrics])
                row += " |\n"
                rows.append(row)
            
            # Combine into table
            table = header + separator + "".join(rows)
            tables["clustering_metrics"] = table
        
        # Hyperparameters table
        if "hyperparameters" in results_data:
            # Create table header
            header = "| Parameter | Value |\n"
            separator = "|----------|-------|\n"
            
            # Create table rows
            rows = []
            for param_name, param_value in results_data["hyperparameters"].items():
                row = f"| {param_name} | {param_value} |\n"
                rows.append(row)
            
            # Combine into table
            table = header + separator + "".join(rows)
            tables["hyperparameters"] = table
        
        # Dataset statistics table
        if "dataset_stats" in results_data:
            # Create table header
            header = "| Statistic | Value |\n"
            separator = "|----------|-------|\n"
            
            # Create table rows
            rows = []
            for stat_name, stat_value in results_data["dataset_stats"].items():
                # Format based on type
                if isinstance(stat_value, (int, float)):
                    value_str = f"{stat_value}"
                elif isinstance(stat_value, dict):
                    value_str = ", ".join([f"{k}: {v}" for k, v in stat_value.items()])
                else:
                    value_str = str(stat_value)
                
                row = f"| {stat_name} | {value_str} |\n"
                rows.append(row)
            
            # Combine into table
            table = header + separator + "".join(rows)
            tables["dataset_stats"] = table
        
        return tables
    
    def generate_results_markdown(self, results_data, output_path):
        """
        Generate a markdown document with all experimental results.
        
        Args:
            results_data: Dictionary with experimental results.
            output_path: Path to save the markdown document.
            
        Returns:
            Path to the generated document.
        """
        # Create tables
        tables = self.generate_tables_for_results(results_data)
        
        # Generate figures
        figure_paths = self.create_figures_for_results(results_data)
        
        # Create markdown document
        markdown = "# Model Zoo Retrieval Experiment Results\n\n"
        
        # Add experiment description
        if "description" in results_data:
            markdown += "## Experiment Description\n\n"
            markdown += results_data["description"] + "\n\n"
        
        # Add dataset statistics
        if "dataset_stats" in tables:
            markdown += "## Dataset Statistics\n\n"
            markdown += tables["dataset_stats"] + "\n\n"
        
        # Add hyperparameters
        if "hyperparameters" in tables:
            markdown += "## Hyperparameters\n\n"
            markdown += tables["hyperparameters"] + "\n\n"
        
        # Add retrieval metrics
        if "retrieval_metrics" in tables:
            markdown += "## Retrieval Performance\n\n"
            markdown += "The following table shows the retrieval performance metrics for different models:\n\n"
            markdown += tables["retrieval_metrics"] + "\n\n"
            
            # Add figures
            if "retrieval_metrics_precision.png" in figure_paths:
                markdown += "### Precision@k\n\n"
                markdown += f"![Precision@k](./retrieval_metrics_precision.{self.save_format})\n\n"
            
            if "retrieval_metrics_recall.png" in figure_paths:
                markdown += "### Recall@k\n\n"
                markdown += f"![Recall@k](./retrieval_metrics_recall.{self.save_format})\n\n"
            
            if "retrieval_metrics_f1.png" in figure_paths:
                markdown += "### F1@k\n\n"
                markdown += f"![F1@k](./retrieval_metrics_f1.{self.save_format})\n\n"
            
            if "retrieval_metrics_map.png" in figure_paths:
                markdown += "### Mean Average Precision\n\n"
                markdown += f"![mAP](./retrieval_metrics_map.{self.save_format})\n\n"
        
        # Add transfer metrics
        if "transfer_metrics" in tables:
            markdown += "## Transfer Learning Performance\n\n"
            markdown += "The following table shows the transfer learning performance for different models:\n\n"
            markdown += tables["transfer_metrics"] + "\n\n"
            
            # Add figure
            if "transfer_performance.png" in figure_paths:
                markdown += f"![Transfer Performance](./transfer_performance.{self.save_format})\n\n"
        
        # Add symmetry metrics
        if "symmetry_metrics" in tables:
            markdown += "## Symmetry Robustness\n\n"
            markdown += "The following table shows the symmetry robustness metrics for different models:\n\n"
            markdown += tables["symmetry_metrics"] + "\n\n"
            
            # Add figures
            if "symmetry_robustness_similarity.png" in figure_paths:
                markdown += "### Similarity Metrics\n\n"
                markdown += f"![Similarity Metrics](./symmetry_robustness_similarity.{self.save_format})\n\n"
            
            if "symmetry_robustness_distance.png" in figure_paths:
                markdown += "### Distance Metrics\n\n"
                markdown += f"![Distance Metrics](./symmetry_robustness_distance.{self.save_format})\n\n"
        
        # Add clustering metrics
        if "clustering_metrics" in tables:
            markdown += "## Clustering Quality\n\n"
            markdown += "The following table shows the clustering quality metrics for different models:\n\n"
            markdown += tables["clustering_metrics"] + "\n\n"
            
            # Add figure
            if "clustering_metrics.png" in figure_paths:
                markdown += f"![Clustering Metrics](./clustering_metrics.{self.save_format})\n\n"
        
        # Add overall comparison
        if "overall_comparison.png" in figure_paths:
            markdown += "## Overall Model Comparison\n\n"
            markdown += f"![Overall Comparison](./overall_comparison.{self.save_format})\n\n"
        
        # Add embedding visualizations
        embedding_figs = [p for p in figure_paths if p.startswith("embedding_visualization_")]
        if embedding_figs:
            markdown += "## Embedding Visualizations\n\n"
            
            for fig_path in embedding_figs:
                model_name = fig_path.replace(f"embedding_visualization_", "").replace(f".{self.save_format}", "")
                markdown += f"### {model_name}\n\n"
                markdown += f"![{model_name} Embeddings](./{fig_path})\n\n"
        
        # Add training history
        if "training_history_loss.png" in figure_paths:
            markdown += "## Training History\n\n"
            markdown += "### Loss Curves\n\n"
            markdown += f"![Training Loss](./training_history_loss.{self.save_format})\n\n"
            
            if "training_history_components.png" in figure_paths:
                markdown += "### Loss Components\n\n"
                markdown += f"![Loss Components](./training_history_components.{self.save_format})\n\n"
        
        # Add conclusions
        if "conclusions" in results_data:
            markdown += "## Conclusions\n\n"
            markdown += results_data["conclusions"] + "\n\n"
        
        # Add limitations and future work
        if "limitations" in results_data:
            markdown += "## Limitations and Future Work\n\n"
            markdown += results_data["limitations"] + "\n\n"
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(markdown)
        
        logger.info(f"Generated results markdown at {output_path}")
        
        return output_path


# Test code
if __name__ == "__main__":
    # Create a sample results data dictionary
    results_data = {
        "description": "This experiment compares the performance of different model encoders "
                      "for the task of neural network weight embedding and retrieval.",
        "dataset_stats": {
            "total_models": 100,
            "unique_tasks": 3,
            "models_by_type": {"vision": 50, "nlp": 30, "scientific": 20},
            "average_params": 1250000
        },
        "hyperparameters": {
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "hidden_dim": 128,
            "output_dim": 256,
            "temperature": 0.07
        },
        "training_history": {
            "train_loss": [0.9, 0.7, 0.5, 0.4, 0.35],
            "val_loss": [0.95, 0.8, 0.6, 0.55, 0.5],
            "train_contrastive_loss": [0.8, 0.6, 0.4, 0.3, 0.25],
            "val_contrastive_loss": [0.85, 0.7, 0.5, 0.45, 0.4],
            "train_metric_loss": [0.1, 0.1, 0.1, 0.1, 0.1],
            "val_metric_loss": [0.1, 0.1, 0.1, 0.1, 0.1]
        },
        "retrieval_metrics": {
            "EquivariantGNN": {
                "precision@1": 0.85,
                "precision@5": 0.75,
                "precision@10": 0.65,
                "recall@1": 0.15,
                "recall@5": 0.45,
                "recall@10": 0.65,
                "f1@1": 0.25,
                "f1@5": 0.55,
                "f1@10": 0.65,
                "mAP": 0.70
            },
            "Transformer": {
                "precision@1": 0.65,
                "precision@5": 0.55,
                "precision@10": 0.45,
                "recall@1": 0.10,
                "recall@5": 0.35,
                "recall@10": 0.50,
                "f1@1": 0.18,
                "f1@5": 0.42,
                "f1@10": 0.47,
                "mAP": 0.55
            },
            "PCA": {
                "precision@1": 0.45,
                "precision@5": 0.40,
                "precision@10": 0.35,
                "recall@1": 0.08,
                "recall@5": 0.25,
                "recall@10": 0.42,
                "f1@1": 0.13,
                "f1@5": 0.30,
                "f1@10": 0.38,
                "mAP": 0.38
            }
        },
        "transfer_metrics": {
            "EquivariantGNN": {
                "perf_improvement@10": 0.05,
                "perf_improvement@50": 0.15,
                "perf_improvement@100": 0.20
            },
            "Transformer": {
                "perf_improvement@10": 0.03,
                "perf_improvement@50": 0.10,
                "perf_improvement@100": 0.15
            },
            "PCA": {
                "perf_improvement@10": 0.02,
                "perf_improvement@50": 0.07,
                "perf_improvement@100": 0.10
            }
        },
        "symmetry_metrics": {
            "EquivariantGNN": {
                "mean_similarity": 0.95,
                "min_similarity": 0.90,
                "mean_distance": 0.05,
                "max_distance": 0.10
            },
            "Transformer": {
                "mean_similarity": 0.75,
                "min_similarity": 0.65,
                "mean_distance": 0.25,
                "max_distance": 0.35
            },
            "PCA": {
                "mean_similarity": 0.60,
                "min_similarity": 0.45,
                "mean_distance": 0.40,
                "max_distance": 0.55
            }
        },
        "clustering_metrics": {
            "EquivariantGNN": {
                "silhouette_score": 0.75,
                "davies_bouldin_score": 0.25
            },
            "Transformer": {
                "silhouette_score": 0.60,
                "davies_bouldin_score": 0.40
            },
            "PCA": {
                "silhouette_score": 0.45,
                "davies_bouldin_score": 0.55
            }
        },
        "conclusions": "The permutation-equivariant GNN encoder outperforms baseline methods "
                      "across all metrics, demonstrating the importance of respecting weight space "
                      "symmetries for effective model retrieval.",
        "limitations": "The current approach has some limitations:\n"
                    "- Limited to fixed model architectures\n"
                    "- Scaling to very large models remains challenging\n"
                    "- Real-world transfer performance needs further validation\n\n"
                    "Future work should address these issues and explore applications in model editing, "
                    "meta-optimization, and security domains."
    }
    
    # Create visualizer
    visualizer = ResultsVisualizer()
    
    # Generate results markdown
    output_path = Path(FIGURES_DIR) / "test_results.md"
    visualizer.generate_results_markdown(results_data, output_path)
    
    print(f"Generated test results at {output_path}")
    
    # Create some dummy embeddings for visualization test
    num_models = 100
    embedding_dim = 32
    
    # Create dummy embeddings
    embeddings = {
        "EquivariantGNN": np.random.randn(num_models, embedding_dim),
        "Transformer": np.random.randn(num_models, embedding_dim),
        "PCA": np.random.randn(num_models, embedding_dim)
    }
    
    # Create dummy task labels
    labels = []
    for i in range(num_models):
        if i < 30:
            labels.append("classification")
        elif i < 70:
            labels.append("detection")
        else:
            labels.append("segmentation")
    
    # Update results data with embeddings
    results_data["embeddings"] = embeddings
    results_data["labels"] = labels
    
    # Regenerate with embeddings
    visualizer.generate_results_markdown(results_data, output_path)
    
    print(f"Updated test results with embedding visualizations at {output_path}")