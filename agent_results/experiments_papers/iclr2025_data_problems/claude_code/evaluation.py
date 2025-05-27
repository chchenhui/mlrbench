import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset
import logging
from utils import Timer, compute_recalls, plot_method_comparison, plot_fairness_metrics
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pandas as pd
from influence_estimation import contrastive_loss, MultiModalModel, train_epoch, evaluate

logger = logging.getLogger("influence_space")

def evaluate_image_text_retrieval(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on image-text retrieval task.
    
    Args:
        model: Trained model
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    all_indices = []
    all_genders = []
    all_ethnicities = []
    
    # Extract embeddings and metadata
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Computing test embeddings"):
            # Extract data
            images = batch["image_embeddings"].to(device)
            texts = batch["text_embeddings"].to(device)
            indices = batch["idx"].tolist()
            
            # Get image and text embeddings
            image_embeddings, text_embeddings = model(images, texts)
            
            # Store embeddings and metadata
            all_image_embeddings.append(image_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
            all_indices.extend(indices)
            
            # Extract demographic information if available
            if "gender" in batch:
                all_genders.extend(batch["gender"])
            if "ethnicity" in batch:
                all_ethnicities.extend(batch["ethnicity"])
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_embeddings, all_text_embeddings.t()).numpy()
    
    # Compute recall metrics
    recall_metrics = compute_recalls(similarity_matrix, ks=[1, 5, 10])
    
    # Compute demographic-specific metrics if available
    demographic_metrics = {}
    
    if all_genders and all_ethnicities:
        # Convert to numpy arrays for easier handling
        genders = np.array(all_genders)
        ethnicities = np.array(all_ethnicities)
        
        # Compute metrics for each gender
        unique_genders = sorted(set(genders))
        for gender in unique_genders:
            gender_indices = np.where(genders == gender)[0]
            if len(gender_indices) >= 2:  # Need at least 2 samples to compute meaningful metrics
                gender_similarity = similarity_matrix[gender_indices][:, gender_indices]
                gender_metrics = compute_recalls(gender_similarity, ks=[1, 5, 10])
                demographic_metrics[f"gender_{gender}"] = gender_metrics
        
        # Compute metrics for each ethnicity
        unique_ethnicities = sorted(set(ethnicities))
        for ethnicity in unique_ethnicities:
            ethnicity_indices = np.where(ethnicities == ethnicity)[0]
            if len(ethnicity_indices) >= 2:  # Need at least 2 samples to compute meaningful metrics
                ethnicity_similarity = similarity_matrix[ethnicity_indices][:, ethnicity_indices]
                ethnicity_metrics = compute_recalls(ethnicity_similarity, ks=[1, 5, 10])
                demographic_metrics[f"ethnicity_{ethnicity}"] = ethnicity_metrics
        
        # Compute performance gaps
        gaps = {}
        
        # Gender gaps
        gender_performances = {gender: metrics["avg_recall@1"] 
                             for gender, metrics in demographic_metrics.items() 
                             if gender.startswith("gender_")}
        if gender_performances:
            gaps["gender"] = max(gender_performances.values()) - min(gender_performances.values())
        
        # Ethnicity gaps
        ethnicity_performances = {ethnicity: metrics["avg_recall@1"] 
                                for ethnicity, metrics in demographic_metrics.items() 
                                if ethnicity.startswith("ethnicity_")}
        if ethnicity_performances:
            gaps["ethnicity"] = max(ethnicity_performances.values()) - min(ethnicity_performances.values())
        
        # Add gaps to metrics
        recall_metrics["demographic_gaps"] = gaps
    
    return recall_metrics

def train_and_evaluate_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    image_dim: int = 512,
    text_dim: int = 512,
    embed_dim: int = 256,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    model_save_path: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Dict[str, float]]:
    """
    Train and evaluate a model on image-text retrieval.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        test_dataloader: DataLoader for test data
        image_dim: Dimension of image embeddings
        text_dim: Dimension of text embeddings
        embed_dim: Dimension of joint embedding space
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        device: Device to run training on
        model_save_path: Path to save the best model
        
    Returns:
        Tuple of (best_model, training_history, test_metrics)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training model on {device} for {num_epochs} epochs")
    
    # Create model
    model = MultiModalModel(image_dim=image_dim, text_dim=text_dim, embed_dim=embed_dim).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_recall@1": [],
        "val_recall@5": [],
        "val_recall@10": []
    }
    
    # Best model and metrics
    best_model = None
    best_val_recall = 0.0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, device)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_recall@1"].append(val_metrics["avg_recall@1"])
        history["val_recall@5"].append(val_metrics["avg_recall@5"])
        history["val_recall@10"].append(val_metrics["avg_recall@10"])
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_metrics['val_loss']:.4f}, "
                  f"Val Recall@1 = {val_metrics['avg_recall@1']:.2f}%, "
                  f"Val Recall@5 = {val_metrics['avg_recall@5']:.2f}%")
        
        # Check if this is the best model
        if val_metrics["avg_recall@1"] > best_val_recall:
            best_val_recall = val_metrics["avg_recall@1"]
            best_model = model.state_dict().copy()
            
            # Save best model if path is provided
            if model_save_path:
                torch.save(best_model, model_save_path)
                logger.info(f"Saved best model to {model_save_path}")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model)
    
    # Evaluate on test set
    test_metrics = evaluate_image_text_retrieval(model, test_dataloader, device)
    
    logger.info(f"Test Recall@1 = {test_metrics['avg_recall@1']:.2f}%, "
              f"Test Recall@5 = {test_metrics['avg_recall@5']:.2f}%, "
              f"Test Recall@10 = {test_metrics['avg_recall@10']:.2f}%")
    
    if "demographic_gaps" in test_metrics:
        gaps = test_metrics["demographic_gaps"]
        for demo, gap in gaps.items():
            logger.info(f"Test {demo.capitalize()} Gap = {gap:.2f}%")
    
    return model, history, test_metrics

def compare_methods(
    method_results: Dict[str, Dict[str, float]],
    output_dir: str = "./",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare different curation methods.
    
    Args:
        method_results: Dictionary mapping method names to their test metrics
        output_dir: Directory to save results
        visualize: Whether to create visualizations
        
    Returns:
        Comparison of methods
    """
    logger.info("Comparing curation methods...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract main metrics for comparison
    main_metrics = ["avg_recall@1", "avg_recall@5", "avg_recall@10"]
    comparison = {metric: {method: results[metric] for method, results in method_results.items()} 
                for metric in main_metrics}
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(output_dir, "method_comparison.csv"))
    
    # Extract demographic gaps for comparison
    demographic_gaps = {}
    for method, results in method_results.items():
        if "demographic_gaps" in results:
            demographic_gaps[method] = results["demographic_gaps"]
    
    # Save demographic gaps to file
    if demographic_gaps:
        with open(os.path.join(output_dir, "demographic_gaps.json"), "w") as f:
            json.dump(demographic_gaps, f, indent=2)
    
    # Visualize comparison
    if visualize:
        # Plot method comparison for each metric
        for metric in main_metrics:
            plot_method_comparison(
                method_results, 
                metric,
                title=f"Method Comparison: {metric}",
                save_path=os.path.join(output_dir, f"comparison_{metric}.png")
            )
        
        # Plot fairness metrics
        if demographic_gaps:
            plot_fairness_metrics(
                demographic_gaps,
                title="Demographic Performance Gaps",
                save_path=os.path.join(output_dir, "fairness_comparison.png")
            )
    
    return comparison

def run_evaluation_pipeline(
    method_dataloaders: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]],
    output_dir: str = "./",
    num_epochs: int = 10,
    image_dim: int = 512,
    text_dim: int = 512,
    embed_dim: int = 256,
    device: Optional[torch.device] = None,
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run the full evaluation pipeline for multiple methods.
    
    Args:
        method_dataloader: Dictionary mapping method names to tuples of (train, val, test) dataloaders
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        image_dim: Dimension of image embeddings
        text_dim: Dimension of text embeddings
        embed_dim: Dimension of joint embedding space
        device: Device to run evaluation on
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary mapping method names to their test metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Results for each method
    method_results = {}
    method_histories = {}
    
    # Evaluate each method
    for method_name, (train_loader, val_loader, test_loader) in method_dataloaders.items():
        logger.info(f"Evaluating {method_name} method...")
        
        # Create method-specific output directory
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Train and evaluate model
        model, history, test_metrics = train_and_evaluate_model(
            train_loader,
            val_loader,
            test_loader,
            image_dim=image_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            num_epochs=num_epochs,
            device=device,
            model_save_path=os.path.join(method_dir, "best_model.pt")
        )
        
        # Store results
        method_results[method_name] = test_metrics
        method_histories[method_name] = history
        
        # Save metrics
        with open(os.path.join(method_dir, "test_metrics.json"), "w") as f:
            # Convert numpy values to Python types for JSON serialization
            serializable_metrics = {}
            for k, v in test_metrics.items():
                if isinstance(v, dict):
                    serializable_metrics[k] = {k2: float(v2) for k2, v2 in v.items()}
                else:
                    serializable_metrics[k] = float(v)
            json.dump(serializable_metrics, f, indent=2)
        
        # Save history
        with open(os.path.join(method_dir, "training_history.json"), "w") as f:
            serializable_history = {k: [float(v2) for v2 in v] for k, v in history.items()}
            json.dump(serializable_history, f, indent=2)
        
        # Visualize training history
        if visualize:
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Val Loss")
            plt.title(f"{method_name}: Loss Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(method_dir, "loss_curves.png"))
            plt.close()
            
            # Plot recall curves
            plt.figure(figsize=(10, 6))
            plt.plot(history["val_recall@1"], label="Recall@1")
            plt.plot(history["val_recall@5"], label="Recall@5")
            plt.plot(history["val_recall@10"], label="Recall@10")
            plt.title(f"{method_name}: Validation Recall Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Recall (%)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(method_dir, "recall_curves.png"))
            plt.close()
    
    # Compare methods
    comparison = compare_methods(method_results, output_dir=output_dir, visualize=visualize)
    
    return method_results

def compute_efficiency_metrics(
    method_cluster_counts: Dict[str, int],
    method_sample_counts: Dict[str, int],
    full_dataset_size: int,
    method_training_times: Dict[str, float],
    output_dir: str = "./",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute efficiency metrics for different methods.
    
    Args:
        method_cluster_counts: Dictionary mapping method names to number of clusters used
        method_sample_counts: Dictionary mapping method names to number of samples used
        full_dataset_size: Size of the full dataset
        method_training_times: Dictionary mapping method names to training times
        output_dir: Directory to save results
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary mapping method names to efficiency metrics
    """
    logger.info("Computing efficiency metrics...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute efficiency metrics
    efficiency_metrics = {}
    
    for method, sample_count in method_sample_counts.items():
        # Data reduction ratio
        reduction_ratio = 1.0 - (sample_count / full_dataset_size)
        
        # Training time (if available)
        training_time = method_training_times.get(method, 0.0)
        
        # Number of clusters (if applicable)
        cluster_count = method_cluster_counts.get(method, 0)
        
        # Store metrics
        efficiency_metrics[method] = {
            "data_reduction_ratio": reduction_ratio,
            "normalized_training_time": training_time / method_training_times.get("Full Dataset", 1.0),
            "cluster_count": cluster_count
        }
    
    # Save metrics to file
    with open(os.path.join(output_dir, "efficiency_metrics.json"), "w") as f:
        json.dump(efficiency_metrics, f, indent=2)
    
    # Create comparison table
    efficiency_df = pd.DataFrame(efficiency_metrics).T
    efficiency_df.to_csv(os.path.join(output_dir, "efficiency_metrics.csv"))
    
    # Visualize efficiency metrics
    if visualize:
        # Plot data reduction ratio
        plt.figure(figsize=(12, 6))
        methods = list(efficiency_metrics.keys())
        reduction_ratios = [metrics["data_reduction_ratio"] * 100 for metrics in efficiency_metrics.values()]
        
        plt.bar(methods, reduction_ratios)
        plt.title("Data Reduction Ratio by Method")
        plt.xlabel("Method")
        plt.ylabel("Data Reduction (%)")
        plt.grid(True, axis='y')
        
        # Add values on top of bars
        for i, value in enumerate(reduction_ratios):
            plt.text(i, value + 1, f"{value:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "data_reduction.png"))
        plt.close()
        
        # Plot normalized training time
        plt.figure(figsize=(12, 6))
        methods = list(efficiency_metrics.keys())
        training_times = [metrics["normalized_training_time"] for metrics in efficiency_metrics.values()]
        
        plt.bar(methods, training_times)
        plt.title("Normalized Training Time by Method")
        plt.xlabel("Method")
        plt.ylabel("Normalized Training Time")
        plt.grid(True, axis='y')
        
        # Add values on top of bars
        for i, value in enumerate(training_times):
            plt.text(i, value + 0.05, f"{value:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_time.png"))
        plt.close()
    
    return efficiency_metrics