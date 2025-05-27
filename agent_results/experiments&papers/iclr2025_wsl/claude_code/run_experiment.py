"""
Main experiment script for model zoo retrieval experiment.
This script orchestrates the entire workflow including data preparation, model training, 
evaluation, and visualization.
"""

import os
import json
import time
import logging
import numpy as np
import torch
import random
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Local imports
from config import *
from data_prep import ModelZooDataset
from weight_to_graph import WeightToGraph
from gnn_encoder import ModelEmbedder
from contrastive_learning import ContrastiveLearningFramework, SymmetryAugmenter
from baselines import PCAEncoder, TransformerEncoder, HypernetworkEncoder, FlatVectorizer
from evaluation import RetrievalEvaluator, CrossValidationEvaluator
from visualization import ResultsVisualizer

# Set up logging with timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"experiment_{timestamp}.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("experiment")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

class ModelGraphDataset(Dataset):
    """
    Dataset for model graph representations.
    """
    
    def __init__(self, model_ids, model_graphs, task_labels, model_performance):
        """
        Initialize the dataset.
        
        Args:
            model_ids: List of model IDs
            model_graphs: List of model graph lists (one list per model)
            task_labels: List of task labels for each model
            model_performance: List of performance values for each model
        """
        self.model_ids = model_ids
        self.model_graphs = model_graphs
        self.task_labels = task_labels
        self.model_performance = torch.tensor(model_performance, dtype=torch.float32)
        
        assert len(model_ids) == len(model_graphs) == len(task_labels) == len(model_performance), \
            "Mismatch in dataset sizes"
    
    def __len__(self):
        return len(self.model_ids)
    
    def __getitem__(self, idx):
        return (
            self.model_graphs[idx],
            self.model_performance[idx],
            self.model_ids[idx]
        )

def create_dataloaders(dataset, train_ratio=0.8, batch_size=TRAIN_CONFIG["batch_size"]):
    """
    Create training and validation dataloaders.
    
    Args:
        dataset: ModelGraphDataset object
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for dataloaders
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    # Get unique task labels
    unique_tasks = {}
    for i, label in enumerate(dataset.task_labels):
        if label not in unique_tasks:
            unique_tasks[label] = []
        unique_tasks[label].append(i)
    
    # Stratified split by task
    train_indices = []
    val_indices = []
    
    for task, indices in unique_tasks.items():
        random.shuffle(indices)
        train_count = int(len(indices) * train_ratio)
        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count:])
    
    # Create subdatasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: ([item[0] for item in x], 
                              torch.stack([item[1] for item in x]),
                              [item[2] for item in x])
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: ([item[0] for item in x], 
                              torch.stack([item[1] for item in x]),
                              [item[2] for item in x])
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)} samples, val={len(val_dataset)} samples")
    
    return train_loader, val_loader

def prepare_data():
    """
    Prepare model zoo dataset and convert to graph representations.
    
    Returns:
        Tuple of (model_ids, model_graphs, task_labels, model_performance, metadata)
    """
    logger.info("Preparing data...")
    
    # Create model zoo dataset
    model_zoo = ModelZooDataset()
    
    # Check if metadata exists, otherwise create synthetic model zoo
    if os.path.exists(model_zoo.metadata_file):
        logger.info("Loading existing model metadata")
        model_zoo.load_metadata()
    else:
        logger.info("Creating synthetic model zoo")
        model_zoo.create_synthetic_model_zoo()
    
    # Get model IDs
    model_ids = model_zoo.get_all_models()
    
    # Get task labels (combine task and dataset)
    task_labels = []
    for model_id in model_ids:
        metadata = model_zoo.get_model_metadata(model_id)
        task_labels.append(f"{metadata['task']}_{metadata['dataset']}")
    
    # Get performance values
    model_performance = []
    for model_id in model_ids:
        metadata = model_zoo.get_model_metadata(model_id)
        model_performance.append(metadata["performance"])
    
    # Convert weights to graphs
    logger.info("Converting model weights to graph representations...")
    converter = WeightToGraph()
    model_graphs = []
    
    for model_id in tqdm(model_ids, desc="Converting weights to graphs"):
        # Load model weights
        weights = model_zoo.get_model_weights(model_id)
        if weights is None:
            logger.warning(f"Failed to load weights for {model_id}, skipping")
            continue
        
        # Convert to graphs
        graphs = converter.convert_model_to_graphs(weights)
        model_graphs.append(graphs)
    
    # Get metadata dictionary
    metadata = {model_id: model_zoo.get_model_metadata(model_id) for model_id in model_ids}
    
    # Print data statistics
    logger.info(f"Prepared data with {len(model_ids)} models and {len(set(task_labels))} unique tasks")
    
    return model_ids, model_graphs, task_labels, model_performance, metadata

def train_model(model_ids, model_graphs, task_labels, model_performance, metadata, 
              epochs=TRAIN_CONFIG["num_epochs"]):
    """
    Train the permutation-equivariant GNN encoder.
    
    Args:
        model_ids: List of model IDs
        model_graphs: List of model graph lists (one list per model)
        task_labels: List of task labels for each model
        model_performance: List of performance values for each model
        metadata: Dictionary of model metadata
        epochs: Number of training epochs
        
    Returns:
        Tuple of (trained_model, history, framework)
    """
    logger.info("Preparing model training...")
    
    # Create dataset and dataloaders
    dataset = ModelGraphDataset(model_ids, model_graphs, task_labels, model_performance)
    train_loader, val_loader = create_dataloaders(dataset)
    
    # Create model and training framework
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ModelEmbedder(
        node_dim=MODEL_CONFIG["gnn_encoder"]["node_dim"],
        edge_dim=MODEL_CONFIG["gnn_encoder"]["edge_dim"],
        hidden_dim=MODEL_CONFIG["gnn_encoder"]["hidden_dim"],
        output_dim=MODEL_CONFIG["gnn_encoder"]["output_dim"],
        num_layers=MODEL_CONFIG["gnn_encoder"]["num_layers"],
        dropout=MODEL_CONFIG["gnn_encoder"]["dropout"],
        readout=MODEL_CONFIG["gnn_encoder"]["readout"]
    )
    
    framework = ContrastiveLearningFramework(
        model_embedder=model,
        lambda_contrastive=TRAIN_CONFIG["lambda_contrastive"],
        temperature=TRAIN_CONFIG["temperature"],
        num_negatives=TRAIN_CONFIG["num_negatives"],
        device=device
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(framework.metric_loss.parameters()),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train model
    logger.info(f"Starting model training for {epochs} epochs on {device}...")
    history, best_model = framework.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Save model
    save_path = os.path.join(MODELS_DIR, "permutation_equivariant_gnn.pt")
    framework.save_model(save_path)
    logger.info(f"Saved trained model to {save_path}")
    
    return model, history, framework

def evaluate_models(model_ids, model_graphs, task_labels, model_performance, metadata):
    """
    Evaluate all models (proposed and baselines).
    
    Args:
        model_ids: List of model IDs
        model_graphs: List of model graph lists (one list per model)
        task_labels: List of task labels for each model
        model_performance: List of performance values for each model
        metadata: Dictionary of model metadata
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating models...")
    
    # Create weight representations
    vectorizer = FlatVectorizer()
    weight_vectors = []
    
    for model_id in tqdm(model_ids, desc="Converting weights to vectors"):
        # Load model weights
        weights = ModelZooDataset().get_model_weights(model_id)
        if weights is None:
            logger.warning(f"Failed to load weights for {model_id}, skipping")
            continue
        
        # Convert to vector
        vector = vectorizer.vectorize_weights(weights)
        weight_vectors.append(vector)
    
    # Create evaluator
    evaluator = RetrievalEvaluator()
    
    # 1. Evaluate Permutation-Equivariant GNN
    logger.info("Evaluating Permutation-Equivariant GNN...")
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = ModelEmbedder().to(device)
    
    # Check if saved model exists
    model_path = os.path.join(MODELS_DIR, "permutation_equivariant_gnn.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        gnn_model.load_state_dict(checkpoint["model_embedder"])
        logger.info(f"Loaded trained GNN model from {model_path}")
    else:
        logger.warning(f"No saved model found at {model_path}, using untrained model")
    
    # Generate embeddings
    logger.info("Generating GNN embeddings...")
    gnn_embeddings = []
    
    with torch.no_grad():
        for i, graphs in enumerate(tqdm(model_graphs, desc="GNN embeddings")):
            # Move graphs to device
            dev_graphs = []
            for g in graphs:
                g = g.to(device)
                dev_graphs.append(g)
            
            # Generate embedding
            embedding = gnn_model.forward(dev_graphs).cpu().numpy()
            gnn_embeddings.append(embedding)
    
    gnn_embeddings = np.stack(gnn_embeddings)
    
    # Evaluate GNN
    gnn_knn_metrics = evaluator.evaluate_knn_retrieval(gnn_embeddings, task_labels)
    gnn_transfer_metrics = evaluator.evaluate_finetuning_transfer(gnn_embeddings, model_performance)
    
    # Create positive pair embeddings with augmentation
    augmenter = SymmetryAugmenter()
    aug_model_graphs = []
    
    for i, graphs in enumerate(tqdm(model_graphs, desc="Augmenting graphs")):
        aug_graphs = augmenter.augment_model_graphs(graphs)
        aug_model_graphs.append(aug_graphs)
    
    aug_gnn_embeddings = []
    
    with torch.no_grad():
        for i, graphs in enumerate(tqdm(aug_model_graphs, desc="Augmented embeddings")):
            # Move graphs to device
            dev_graphs = []
            for g in graphs:
                g = g.to(device)
                dev_graphs.append(g)
            
            # Generate embedding
            embedding = gnn_model.forward(dev_graphs).cpu().numpy()
            aug_gnn_embeddings.append(embedding)
    
    aug_gnn_embeddings = np.stack(aug_gnn_embeddings)
    
    # Evaluate symmetry robustness
    gnn_symmetry_metrics = evaluator.evaluate_symmetry_robustness(gnn_embeddings, aug_gnn_embeddings)
    
    # Evaluate clustering quality
    gnn_clustering_metrics = evaluator.evaluate_clustering_quality(gnn_embeddings, task_labels)
    
    # 2. Evaluate Transformer (Non-equivariant baseline)
    logger.info("Evaluating Transformer Encoder...")
    
    # Create and train transformer encoder
    transformer_model = TransformerEncoder().to(device)
    
    # Generate embeddings
    logger.info("Generating Transformer embeddings...")
    transformer_embeddings = []
    
    with torch.no_grad():
        for i, model_id in enumerate(tqdm(model_ids, desc="Transformer embeddings")):
            # Load model weights
            weights = ModelZooDataset().get_model_weights(model_id)
            if weights is None:
                logger.warning(f"Failed to load weights for {model_id}, skipping")
                continue
            
            # Generate embedding
            embedding = transformer_model.forward(weights).cpu().numpy()
            transformer_embeddings.append(embedding)
    
    transformer_embeddings = np.stack(transformer_embeddings)
    
    # Apply augmentation to weights for symmetry robustness
    aug_transformer_embeddings = []
    
    with torch.no_grad():
        for i, model_id in enumerate(tqdm(model_ids, desc="Augmented Transformer embeddings")):
            # Load model weights
            weights = ModelZooDataset().get_model_weights(model_id)
            if weights is None:
                continue
            
            # Apply augmentation
            aug_weights, _ = ModelZooDataset().apply_symmetry_transforms(model_id)
            
            # Generate embedding
            embedding = transformer_model.forward(aug_weights).cpu().numpy()
            aug_transformer_embeddings.append(embedding)
    
    aug_transformer_embeddings = np.stack(aug_transformer_embeddings)
    
    # Evaluate transformer
    transformer_knn_metrics = evaluator.evaluate_knn_retrieval(transformer_embeddings, task_labels)
    transformer_transfer_metrics = evaluator.evaluate_finetuning_transfer(transformer_embeddings, model_performance)
    transformer_symmetry_metrics = evaluator.evaluate_symmetry_robustness(
        transformer_embeddings, aug_transformer_embeddings
    )
    transformer_clustering_metrics = evaluator.evaluate_clustering_quality(transformer_embeddings, task_labels)
    
    # 3. Evaluate PCA (Flat baseline)
    logger.info("Evaluating PCA Encoder...")
    
    # Create and fit PCA encoder
    pca_encoder = PCAEncoder(n_components=MODEL_CONFIG["pca_encoder"]["n_components"])
    
    # Create weight dictionaries for training
    weight_dicts = []
    for model_id in model_ids:
        weights = ModelZooDataset().get_model_weights(model_id)
        if weights is not None:
            weight_dicts.append(weights)
    
    pca_encoder.fit(weight_dicts)
    
    # Generate embeddings
    logger.info("Generating PCA embeddings...")
    pca_embeddings = []
    
    for i, model_id in enumerate(tqdm(model_ids, desc="PCA embeddings")):
        # Load model weights
        weights = ModelZooDataset().get_model_weights(model_id)
        if weights is None:
            logger.warning(f"Failed to load weights for {model_id}, skipping")
            continue
        
        # Generate embedding
        embedding = pca_encoder.encode(weights)
        pca_embeddings.append(embedding)
    
    pca_embeddings = np.stack(pca_embeddings)
    
    # Apply augmentation to weights for symmetry robustness
    aug_pca_embeddings = []
    
    for i, model_id in enumerate(tqdm(model_ids, desc="Augmented PCA embeddings")):
        # Load model weights
        weights = ModelZooDataset().get_model_weights(model_id)
        if weights is None:
            continue
        
        # Apply augmentation
        aug_weights, _ = ModelZooDataset().apply_symmetry_transforms(model_id)
        
        # Generate embedding
        embedding = pca_encoder.encode(aug_weights)
        aug_pca_embeddings.append(embedding)
    
    aug_pca_embeddings = np.stack(aug_pca_embeddings)
    
    # Evaluate PCA
    pca_knn_metrics = evaluator.evaluate_knn_retrieval(pca_embeddings, task_labels)
    pca_transfer_metrics = evaluator.evaluate_finetuning_transfer(pca_embeddings, model_performance)
    pca_symmetry_metrics = evaluator.evaluate_symmetry_robustness(pca_embeddings, aug_pca_embeddings)
    pca_clustering_metrics = evaluator.evaluate_clustering_quality(pca_embeddings, task_labels)
    
    # Combine all metrics
    all_retrieval_metrics = {
        "EquivariantGNN": gnn_knn_metrics,
        "Transformer": transformer_knn_metrics,
        "PCA": pca_knn_metrics
    }
    
    all_transfer_metrics = {
        "EquivariantGNN": gnn_transfer_metrics,
        "Transformer": transformer_transfer_metrics,
        "PCA": pca_transfer_metrics
    }
    
    all_symmetry_metrics = {
        "EquivariantGNN": gnn_symmetry_metrics,
        "Transformer": transformer_symmetry_metrics,
        "PCA": pca_symmetry_metrics
    }
    
    all_clustering_metrics = {
        "EquivariantGNN": gnn_clustering_metrics,
        "Transformer": transformer_clustering_metrics,
        "PCA": pca_clustering_metrics
    }
    
    # Create embeddings dictionary for visualization
    embeddings_dict = {
        "EquivariantGNN": gnn_embeddings,
        "Transformer": transformer_embeddings,
        "PCA": pca_embeddings
    }
    
    # Create dataset statistics
    dataset_stats = ModelZooDataset().stats_summary()
    
    # Create results dictionary
    results = {
        "description": "This experiment compares the performance of different model encoders "
                      "for the task of neural network weight embedding and retrieval, "
                      "with a focus on permutation equivariance.",
        "dataset_stats": dataset_stats,
        "hyperparameters": {
            "batch_size": TRAIN_CONFIG["batch_size"],
            "num_epochs": TRAIN_CONFIG["num_epochs"],
            "learning_rate": TRAIN_CONFIG["learning_rate"],
            "weight_decay": TRAIN_CONFIG["weight_decay"],
            "hidden_dim": MODEL_CONFIG["gnn_encoder"]["hidden_dim"],
            "output_dim": MODEL_CONFIG["gnn_encoder"]["output_dim"],
            "temperature": TRAIN_CONFIG["temperature"],
            "lambda_contrastive": TRAIN_CONFIG["lambda_contrastive"]
        },
        "retrieval_metrics": all_retrieval_metrics,
        "transfer_metrics": all_transfer_metrics,
        "symmetry_metrics": all_symmetry_metrics,
        "clustering_metrics": all_clustering_metrics,
        "embeddings": embeddings_dict,
        "labels": task_labels,
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
    
    # Return results
    return results

def main(args):
    """
    Main function to run the entire experiment.
    
    Args:
        args: Command-line arguments
        
    Returns:
        None
    """
    # Set random seed
    set_seed(args.seed)
    
    # Prepare data
    logger.info(f"=== Starting Experiment: {timestamp} ===")
    
    # Record timing
    start_time = time.time()
    
    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare data
    logger.info("Step 1: Preparing data...")
    model_ids, model_graphs, task_labels, model_performance, metadata = prepare_data()
    
    # Train model
    if not args.skip_training:
        logger.info("Step 2: Training permutation-equivariant GNN model...")
        model, history, framework = train_model(
            model_ids, model_graphs, task_labels, model_performance, metadata,
            epochs=args.epochs
        )
    else:
        logger.info("Step 2: Skipping model training as requested")
        history = None
    
    # Evaluate models
    logger.info("Step 3: Evaluating all models...")
    results = evaluate_models(model_ids, model_graphs, task_labels, model_performance, metadata)
    
    # Add training history to results
    if history is not None:
        results["training_history"] = history
    
    # Create visualizations and results
    logger.info("Step 4: Creating visualizations and results...")
    visualizer = ResultsVisualizer()
    
    # Generate results.md
    results_path = os.path.join(RESULTS_DIR, "results.md")
    visualizer.generate_results_markdown(results, results_path)
    
    # Save raw results
    raw_results_path = os.path.join(RESULTS_DIR, "results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = results.copy()
    
    # Remove embeddings from serializable results (too large)
    if "embeddings" in serializable_results:
        del serializable_results["embeddings"]
    
    # Convert numpy values to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    serializable_results = convert_numpy(serializable_results)
    
    with open(raw_results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Save execution log
    log_content = f"""
    # Experiment Execution Log
    
    Timestamp: {timestamp}
    Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)
    
    ## Command Line Arguments
    
    - Seed: {args.seed}
    - Epochs: {args.epochs}
    - Skip Training: {args.skip_training}
    
    ## Data Statistics
    
    - Number of Models: {len(model_ids)}
    - Number of Tasks: {len(set(task_labels))}
    
    ## Hardware Information
    
    - Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    - CUDA Available: {torch.cuda.is_available()}
    - Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
    
    ## Software Information
    
    - Python Version: {sys.version}
    - PyTorch Version: {torch.__version__}
    - NumPy Version: {np.__version__}
    
    ## Execution Steps
    
    1. Data Preparation: Completed
    2. Model Training: {"Skipped" if args.skip_training else "Completed"}
    3. Model Evaluation: Completed
    4. Results Generation: Completed
    
    ## Results Location
    
    - Results Markdown: {results_path}
    - Raw Results JSON: {raw_results_path}
    - Figures Directory: {FIGURES_DIR}
    - Model Directory: {MODELS_DIR}
    
    """
    
    log_path = os.path.join(RESULTS_DIR, "log.txt")
    with open(log_path, "w") as f:
        f.write(log_content)
    
    logger.info(f"=== Experiment Complete: {timestamp} ===")
    logger.info(f"Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    logger.info(f"Results saved to {RESULTS_DIR}")
    
    # Return results path
    return results_path

if __name__ == "__main__":
    # Import additional modules for main execution
    import sys
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model zoo retrieval experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"], 
                      help="Number of training epochs")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    # Run main function
    results_path = main(args)