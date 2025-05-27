#!/usr/bin/env python
"""
Simplified experiment runner for the Gradient-Informed Fingerprinting (GIF) method.

This script runs a simplified version of the GIF attribution experiment,
using synthetic data to quickly test the full pipeline.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/simplified_experiment.log")
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from models.probe import ProbeNetwork, ProbeTrainer, FingerprintGenerator
from models.indexing import ANNIndex
from models.influence import InfluenceEstimator, AttributionRefiner
from models.baselines import TRACEMethod
from utils.metrics import AttributionMetrics, LatencyTracker
from utils.visualization import ExperimentVisualizer


def create_synthetic_data(
    num_samples: int = 1000,
    embedding_dim: int = 384,
    num_classes: int = 10,
    seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create synthetic data for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate embeddings for different classes with some clustering
    embeddings = []
    labels = []
    
    # Create centers for each class
    class_centers = np.random.randn(num_classes, embedding_dim)
    
    # Generate samples around class centers
    for i in range(num_samples):
        label = i % num_classes
        noise = np.random.randn(embedding_dim) * 0.1
        embedding = class_centers[label] + noise
        
        embeddings.append(embedding)
        labels.append(label)
    
    # Split into train, val, test
    train_ratio, val_ratio = 0.7, 0.15
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size
    
    # Create datasets
    train_embeddings = np.array(embeddings[:train_size], dtype=np.float32)
    train_labels = np.array(labels[:train_size], dtype=np.int64)
    train_ids = [f"train_{i}" for i in range(train_size)]
    
    val_embeddings = np.array(embeddings[train_size:train_size+val_size], dtype=np.float32)
    val_labels = np.array(labels[train_size:train_size+val_size], dtype=np.int64)
    val_ids = [f"val_{i}" for i in range(val_size)]
    
    test_embeddings = np.array(embeddings[train_size+val_size:], dtype=np.float32)
    test_labels = np.array(labels[train_size+val_size:], dtype=np.int64)
    test_ids = [f"test_{i}" for i in range(test_size)]
    
    # Create text data for TRACE baseline
    texts = []
    for i in range(num_samples):
        label = labels[i]
        text = f"Sample {i}: This is class {label} with some additional text to make it more realistic."
        texts.append(text)
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size+val_size]
    test_texts = texts[train_size+val_size:]
    
    # Create dataloaders
    batch_size = 32
    
    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_embeddings, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_embeddings, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare data dictionary
    data = {
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "train_ids": train_ids,
        "train_texts": train_texts,
        "val_embeddings": val_embeddings,
        "val_labels": val_labels,
        "val_ids": val_ids,
        "val_texts": val_texts,
        "test_embeddings": test_embeddings,
        "test_labels": test_labels,
        "test_ids": test_ids,
        "test_texts": test_texts,
        "embedding_dim": embedding_dim,
        "num_classes": num_classes
    }
    
    # Prepare dataloaders dictionary
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }
    
    return data, dataloaders


def run_simplified_experiment(config_path: str) -> None:
    """Run a simplified experiment using synthetic data."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create synthetic data
    logger.info("Creating synthetic data...")
    data, dataloaders = create_synthetic_data(
        num_samples=config.get("num_samples", 1000),
        embedding_dim=config.get("embedding_dim", 384),
        num_classes=config.get("num_classes", 10),
        seed=config.get("seed", 42)
    )
    
    # Create and train probe network
    logger.info("Training probe network...")
    probe = ProbeNetwork(
        input_dim=data["embedding_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        output_dim=data["num_classes"],
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1)
    )
    
    trainer = ProbeTrainer(
        probe=probe,
        device=device,
        optimizer_cls=optim.AdamW,
        optimizer_kwargs={
            "lr": config.get("learning_rate", 0.001),
            "weight_decay": config.get("weight_decay", 0.00001)
        }
    )
    
    history = trainer.train(
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        num_epochs=config.get("num_epochs", 5),
        early_stopping_patience=config.get("early_stopping_patience", 2),
        model_save_path="models/probe_model.pt"
    )
    
    # Create visualizer
    visualizer = ExperimentVisualizer(results_dir="results")
    
    # Create training curves visualization
    visualizer.create_training_curves(
        training_data={"loss": history["train_loss"], "accuracy": history["train_acc"]},
        validation_data={"loss": history["val_loss"], "accuracy": history["val_acc"]}
    )
    
    # Generate fingerprints
    logger.info("Generating fingerprints...")
    fingerprint_generator = FingerprintGenerator(
        probe=probe,
        projection_dim=config.get("projection_dim", 64),
        device=device,
        fingerprint_type=config.get("fingerprint_type", "combined")
    )
    
    latency_tracker = LatencyTracker()
    
    with latency_tracker.measure("fingerprint_generation"):
        train_fingerprints = fingerprint_generator.create_fingerprints(
            embeddings=data["train_embeddings"],
            labels=data["train_labels"],
            batch_size=config.get("batch_size", 32)
        )
        
        test_fingerprints = fingerprint_generator.create_fingerprints(
            embeddings=data["test_embeddings"],
            labels=data["test_labels"],
            batch_size=config.get("batch_size", 32)
        )
    
    # Build ANN index
    logger.info("Building ANN index...")
    ann_index = ANNIndex(
        index_type=config.get("index_type", "flat"),
        dimension=train_fingerprints.shape[1],
        metric=config.get("metric", "l2")
    )
    
    with latency_tracker.measure("index_building"):
        ann_index.build(
            vectors=train_fingerprints,
            ids=data["train_ids"],
            batch_size=config.get("batch_size", 32)
        )
    
    # Set up influence estimator if enabled
    if config.get("use_influence", True):
        logger.info("Setting up influence estimator...")
        criterion = nn.CrossEntropyLoss()
        influence_estimator = InfluenceEstimator(
            model=probe,
            loss_fn=criterion,
            train_dataloader=dataloaders["train"],
            device=device,
            damping=config.get("damping", 0.01),
            lissa_samples=config.get("lissa_samples", 5),
            lissa_depth=config.get("lissa_depth", 100)
        )
        
        attribution_refiner = AttributionRefiner(
            influence_estimator=influence_estimator,
            top_k=config.get("top_k", 5)
        )
    
    # Set up baseline method (TRACE)
    if config.get("use_trace", True):
        logger.info("Setting up TRACE baseline...")
        trace = TRACEMethod(
            encoder_name="sentence-transformers/all-mpnet-base-v2",
            device=device
        )
        
        # We'll simulate TRACE by using random projections
        # In a real experiment, you would train the TRACE model
        class MockEncoder:
            def encode(self, texts, **kwargs):
                # Generate random embeddings for testing with the correct dimension
                # Use the same dimension as our other embeddings
                return torch.randn(len(texts), data["embedding_dim"])

        trace.encoder = MockEncoder()
        trace.projector = nn.Linear(data["embedding_dim"], data["embedding_dim"]).to(device)
        trace.dimension = data["embedding_dim"]  # Set the correct dimension
        
        # Build index
        trace.build_index(data["train_texts"], data["train_ids"])
    
    # Run attribution experiments
    logger.info("Running attribution experiments...")
    metrics = AttributionMetrics(results_dir="results")
    
    # GIF attribution
    logger.info("Running GIF attribution...")
    num_test = min(config.get("num_test", 30), len(data["test_ids"]))
    k_values = config.get("k_values", [1, 3, 5])
    top_k = max(k_values)
    
    gif_predictions = []
    gif_latencies = []
    
    for i in tqdm(range(num_test)):
        # Get test fingerprint
        test_fingerprint = test_fingerprints[i].reshape(1, -1)
        
        # ANN search
        with latency_tracker.measure("ann_search"):
            results, distances = ann_index.search(test_fingerprint, k=top_k)
        
        # Influence refinement (if enabled)
        if config.get("use_influence", True):
            with latency_tracker.measure("influence_refinement"):
                # Build a simple test sample
                test_sample = {
                    "id": data["test_ids"][i],
                    "input": torch.tensor(data["test_embeddings"][i], dtype=torch.float32).unsqueeze(0),
                    "target": torch.tensor(data["test_labels"][i], dtype=torch.long).unsqueeze(0)
                }
                
                # Get candidate IDs from ANN search
                candidate_ids = [id_ for id_ in results[0] if id_ is not None]
                
                # Create ID to index mapping
                id_to_index = {id_: idx for idx, id_ in enumerate(data["train_ids"])}
                
                # Refine candidates
                refined_results = attribution_refiner.refine_candidates(
                    test_sample, candidate_ids, dataloaders["train_dataset"], id_to_index
                )
                
                # Extract IDs and scores
                refined_ids = [id_ for id_, _ in refined_results]
                refined_scores = [score for _, score in refined_results]
                
                # Use refined results
                predictions = refined_ids
        else:
            # Use ANN results directly
            predictions = results[0]
        
        # Record results
        gif_predictions.append(predictions)
        gif_latencies.append(
            latency_tracker.get_measurements("ann_search")[-1] +
            (latency_tracker.get_measurements("influence_refinement")[-1] 
             if config.get("use_influence", True) else 0)
        )
    
    # Evaluate GIF
    metrics.evaluate_single_method(
        method_name="GIF",
        predictions=gif_predictions,
        true_ids=data["test_ids"][:num_test],
        k_values=k_values,
        latencies=gif_latencies
    )
    
    # TRACE attribution
    if config.get("use_trace", True):
        logger.info("Running TRACE attribution...")
        trace_predictions = []
        trace_latencies = []
        
        for i in tqdm(range(num_test)):
            # Get test text
            test_text = data["test_texts"][i]
            
            # Search with TRACE
            start_time = time.time()
            results, distances = trace.search([test_text], k=top_k)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Record results
            trace_predictions.append(results[0])
            trace_latencies.append(latency)
        
        # Evaluate TRACE
        metrics.evaluate_single_method(
            method_name="TRACE",
            predictions=trace_predictions,
            true_ids=data["test_ids"][:num_test],
            k_values=k_values,
            latencies=trace_latencies
        )
    
    # Generate visualizations
    metrics.plot_precision_at_k()
    metrics.plot_recall_at_k()
    metrics.plot_mrr_comparison()
    metrics.plot_latency_comparison()
    
    # Generate summary table
    metrics.generate_summary_table()
    
    # Save metrics
    metrics.save_results()
    
    # Save latency breakdown
    latency_tracker.plot_latency_breakdown(output_dir="results")
    latency_tracker.plot_cumulative_latency(
        operations=["fingerprint_generation", "ann_search", "influence_refinement"],
        output_dir="results"
    )
    
    # Generate results report
    logger.info("Generating results report...")
    
    report = "# Gradient-Informed Fingerprinting (GIF) Simplified Experiment Results\n\n"
    
    # Add experiment details
    report += "## Experiment Details\n\n"
    report += f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"- Samples: {config.get('num_samples', 1000)}\n"
    report += f"- Embedding Dimension: {config.get('embedding_dim', 384)}\n"
    report += f"- Classes: {config.get('num_classes', 10)}\n"
    report += f"- Fingerprint Type: {config.get('fingerprint_type', 'combined')}\n"
    report += f"- Projection Dimension: {config.get('projection_dim', 64)}\n"
    report += f"- Index Type: {config.get('index_type', 'flat')}\n"
    report += f"- Use Influence Refinement: {config.get('use_influence', True)}\n\n"
    
    # Add performance plots
    report += "## Performance Metrics\n\n"
    report += "### Precision@k\n\n"
    report += "![Precision@k](precision_at_k.png)\n\n"
    
    report += "### Mean Reciprocal Rank\n\n"
    report += "![MRR](mrr_comparison.png)\n\n"
    
    # Add latency analysis
    report += "## Latency Analysis\n\n"
    report += "### Method Comparison\n\n"
    report += "![Latency Comparison](latency_comparison.png)\n\n"
    
    report += "### GIF Component Breakdown\n\n"
    report += "![Latency Breakdown](latency_breakdown.png)\n\n"
    
    # Add conclusions
    report += "## Conclusions\n\n"
    report += "This simplified experiment demonstrates the key components of the Gradient-Informed Fingerprinting (GIF) method using synthetic data. The full experiment would use real datasets and more comprehensive evaluation.\n\n"
    
    # Write report
    with open("results/results.md", "w") as f:
        f.write(report)
    
    logger.info("Experiment complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simplified GIF attribution experiment")
    parser.add_argument("--config", type=str, default="config_simplified.json", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run simplified experiment
    run_simplified_experiment(args.config)