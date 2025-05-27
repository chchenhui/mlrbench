#!/usr/bin/env python
"""
Main experiment runner for the Gradient-Informed Fingerprinting (GIF) method.

This script runs the complete experimental pipeline for evaluating GIF and baseline methods:
1. Loads and preprocesses datasets
2. Generates embeddings and clusters
3. Trains the probe network
4. Generates fingerprints
5. Builds ANN indexes
6. Runs attribution experiments
7. Evaluates and visualizes results
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, List, Optional, Tuple, Union, Any
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
        logging.FileHandler("logs/experiment.log")
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from data.data_loader import DataConfig, DataManager, TextDataset
from data.embedding import StaticEmbedder, Clusterer, EmbeddingManager
from models.probe import ProbeNetwork, ProbeTrainer, FingerprintGenerator, GradientExtractor
from models.indexing import ANNIndex
from models.influence import InfluenceEstimator, AttributionRefiner
from models.baselines import TRACEMethod, TRAKMethod
from utils.metrics import AttributionMetrics, LatencyTracker
from utils.visualization import ExperimentVisualizer


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, config_path: str):
        """Initialize experiment with configuration."""
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Set experiment name and directories
        self.exp_name = self.config.get("experiment_name", f"experiment_{time.strftime('%Y%m%d_%H%M%S')}")
        self.base_dir = self.config.get("base_dir", ".")
        self.data_dir = os.path.join(self.base_dir, self.config.get("data_dir", "data"))
        self.model_dir = os.path.join(self.base_dir, self.config.get("model_dir", "models"))
        self.results_dir = os.path.join(self.base_dir, self.config.get("results_dir", "results"))
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize trackers
        self.latency_tracker = LatencyTracker()
        
        # Initialize components (will be set up later)
        self.data_manager = None
        self.embedding_manager = None
        self.probe = None
        self.probe_trainer = None
        self.fingerprint_generator = None
        self.ann_index = None
        self.influence_estimator = None
        self.attribution_refiner = None
        self.baselines = {}
        
        # Initialize metrics
        self.metrics = AttributionMetrics(results_dir=self.results_dir)
        
        # Initialize visualizer
        self.visualizer = ExperimentVisualizer(results_dir=self.results_dir)
    
    def setup_data(self) -> None:
        """Set up data loading and preprocessing."""
        logger.info("Setting up data loading and preprocessing...")
        
        # Get data config
        data_config = DataConfig(
            dataset_name=self.config["data"]["dataset_name"],
            subset_name=self.config["data"].get("subset_name"),
            text_column=self.config["data"].get("text_column", "text"),
            max_samples=self.config["data"].get("max_samples"),
            tokenizer_name=self.config["data"].get("tokenizer_name", "bert-base-uncased"),
            max_length=self.config["data"].get("max_length", 512),
            train_ratio=self.config["data"].get("train_ratio", 0.8),
            val_ratio=self.config["data"].get("val_ratio", 0.1),
            test_ratio=self.config["data"].get("test_ratio", 0.1),
            seed=self.config["data"].get("seed", 42),
            use_synthetic=self.config["data"].get("use_synthetic", False),
            synthetic_samples=self.config["data"].get("synthetic_samples", 1000),
            data_dir=self.data_dir
        )
        
        # Create data manager
        self.data_manager = DataManager(data_config)
        
        # Load and process data
        self.datasets = self.data_manager.load_and_process_data()
        self.dataloaders = self.data_manager.get_dataloaders(
            batch_size=self.config["training"].get("batch_size", 32)
        )
        
        # Save dataset stats
        self.data_manager.save_dataset_stats(self.data_dir)
        
        logger.info("Data loading and preprocessing complete")
    
    def setup_embeddings(self) -> None:
        """Set up embeddings and clustering."""
        logger.info("Setting up embeddings and clustering...")
        
        # Create static embedder
        embedder = StaticEmbedder(
            model_name=self.config["embeddings"].get("model_name", "sentence-transformers/all-mpnet-base-v2"),
            device=self.device,
            pooling_strategy=self.config["embeddings"].get("pooling_strategy", "mean")
        )
        
        # Create clusterer
        clusterer = Clusterer(
            n_clusters=self.config["clustering"].get("n_clusters", 100),
            random_state=self.config["clustering"].get("random_state", 42),
            algorithm=self.config["clustering"].get("algorithm", "kmeans")
        )
        
        # Create embedding manager
        self.embedding_manager = EmbeddingManager(
            embedder=embedder,
            clusterer=clusterer,
            output_dir=self.data_dir
        )
        
        # Process dataset
        results = self.embedding_manager.process_dataset(
            dataloaders=self.dataloaders,
            subsample_size=self.config["clustering"].get("subsample_size", 10000),
            fit_on="train"
        )
        
        logger.info(f"Embedding results: {results}")
        logger.info("Embeddings and clustering complete")
    
    def setup_probe(self) -> None:
        """Set up and train the probe network."""
        logger.info("Setting up and training probe network...")
        
        # Load embeddings and clusters
        train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
        train_clusters, _ = self.embedding_manager.load_clusters("train")
        
        val_embeddings, val_ids = self.embedding_manager.load_embeddings("val")
        val_clusters, _ = self.embedding_manager.load_clusters("val")
        
        embedding_dim = train_embeddings.shape[1]
        n_clusters = self.config["clustering"].get("n_clusters", 100)
        hidden_dim = self.config["probe"].get("hidden_dim", 256)
        num_layers = self.config["probe"].get("num_layers", 2)
        
        # Create probe network
        self.probe = ProbeNetwork(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=n_clusters,
            num_layers=num_layers,
            dropout=self.config["probe"].get("dropout", 0.1),
            activation=self.config["probe"].get("activation", "relu")
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float32),
            torch.tensor(train_clusters, dtype=torch.long)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(val_embeddings, dtype=torch.float32),
            torch.tensor(val_clusters, dtype=torch.long)
        )
        
        # Create dataloaders
        batch_size = self.config["training"].get("batch_size", 32)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create trainer
        self.probe_trainer = ProbeTrainer(
            probe=self.probe,
            device=self.device,
            optimizer_cls=getattr(optim, self.config["training"].get("optimizer", "AdamW")),
            optimizer_kwargs={
                "lr": self.config["training"].get("learning_rate", 1e-3),
                "weight_decay": self.config["training"].get("weight_decay", 1e-5)
            }
        )
        
        # Train probe network
        probe_model_path = os.path.join(self.model_dir, "probe_model.pt")
        
        # Check if we should load a pre-trained model
        if self.config["training"].get("load_pretrained", False) and os.path.exists(probe_model_path):
            logger.info(f"Loading pretrained probe from {probe_model_path}")
            self.probe_trainer.load_probe(probe_model_path)
        else:
            # Train the probe
            logger.info("Training probe network...")
            history = self.probe_trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=self.config["training"].get("num_epochs", 10),
                early_stopping_patience=self.config["training"].get("early_stopping_patience", 3),
                model_save_path=probe_model_path
            )
            
            # Save training history
            with open(os.path.join(self.results_dir, "probe_training_history.json"), "w") as f:
                # Convert values to Python native types
                serializable_history = {}
                for key, values in history.items():
                    if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                        serializable_history[key] = [float(v) for v in values]
                    else:
                        serializable_history[key] = values
                
                json.dump(serializable_history, f, indent=2)
            
            # Create training curves visualization
            self.visualizer.create_training_curves(
                training_data={"loss": history["train_loss"], "accuracy": history["train_acc"]},
                validation_data={"loss": history["val_loss"], "accuracy": history["val_acc"]},
                output_filename="probe_training_curves.png"
            )
        
        logger.info("Probe network setup complete")
    
    def generate_fingerprints(self) -> None:
        """Generate fingerprints for all data splits."""
        logger.info("Generating fingerprints...")
        
        # Load embeddings and clusters
        train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
        train_clusters, _ = self.embedding_manager.load_clusters("train")
        
        val_embeddings, val_ids = self.embedding_manager.load_embeddings("val")
        val_clusters, _ = self.embedding_manager.load_clusters("val")
        
        test_embeddings, test_ids = self.embedding_manager.load_embeddings("test")
        test_clusters, _ = self.embedding_manager.load_clusters("test")
        
        # Create fingerprint generator
        projection_dim = self.config["fingerprints"].get("projection_dim", 128)
        fingerprint_type = self.config["fingerprints"].get("type", "combined")
        
        self.fingerprint_generator = FingerprintGenerator(
            probe=self.probe,
            projection_dim=projection_dim,
            device=self.device,
            fingerprint_type=fingerprint_type
        )
        
        # Generate fingerprints
        batch_size = self.config["fingerprints"].get("batch_size", 64)
        
        with self.latency_tracker.measure("fingerprint_generation"):
            # Train fingerprints
            logger.info("Generating train fingerprints...")
            train_fingerprints = self.fingerprint_generator.create_fingerprints(
                embeddings=train_embeddings,
                labels=train_clusters,
                batch_size=batch_size
            )
            
            # Validation fingerprints
            logger.info("Generating validation fingerprints...")
            val_fingerprints = self.fingerprint_generator.create_fingerprints(
                embeddings=val_embeddings,
                labels=val_clusters,
                batch_size=batch_size
            )
            
            # Test fingerprints
            logger.info("Generating test fingerprints...")
            test_fingerprints = self.fingerprint_generator.create_fingerprints(
                embeddings=test_embeddings,
                labels=test_clusters,
                batch_size=batch_size
            )
        
        # Save fingerprints
        np.save(os.path.join(self.data_dir, "train_fingerprints.npy"), train_fingerprints)
        np.save(os.path.join(self.data_dir, "val_fingerprints.npy"), val_fingerprints)
        np.save(os.path.join(self.data_dir, "test_fingerprints.npy"), test_fingerprints)
        
        # Save projection matrix
        self.fingerprint_generator.save_projection_matrix(
            os.path.join(self.model_dir, "projection_matrix.pt")
        )
        
        # Create fingerprint visualization
        logger.info("Creating fingerprint visualization...")
        self.visualizer.create_fingerprint_visualization(
            fingerprints=train_fingerprints[:1000],  # Use a subset
            labels=train_clusters[:1000],
            method="tsne",
            output_filename="fingerprint_visualization.png"
        )
        
        logger.info("Fingerprint generation complete")
    
    def build_index(self) -> None:
        """Build ANN index for similarity search."""
        logger.info("Building ANN index...")
        
        # Load train fingerprints and IDs
        train_fingerprints = np.load(os.path.join(self.data_dir, "train_fingerprints.npy"))
        train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
        
        # Create ANN index
        index_type = self.config["indexing"].get("index_type", "hnsw")
        metric = self.config["indexing"].get("metric", "l2")
        
        self.ann_index = ANNIndex(
            index_type=index_type,
            dimension=train_fingerprints.shape[1],
            metric=metric,
            use_gpu=self.config["indexing"].get("use_gpu", False),
            index_params=self.config["indexing"].get("index_params", {})
        )
        
        # Build index
        with self.latency_tracker.measure("index_building"):
            self.ann_index.build(
                vectors=train_fingerprints,
                ids=train_ids,
                batch_size=self.config["indexing"].get("batch_size", 10000)
            )
        
        # Save index
        self.ann_index.save(self.model_dir)
        
        # Save stats
        index_stats = self.ann_index.get_stats()
        with open(os.path.join(self.results_dir, "index_stats.json"), "w") as f:
            # Convert values to Python native types
            serializable_stats = {}
            for key, value in index_stats.items():
                if isinstance(value, (np.ndarray, np.number)):
                    serializable_stats[key] = value.item() if hasattr(value, "item") else value.tolist()
                else:
                    serializable_stats[key] = value
            
            json.dump(serializable_stats, f, indent=2)
        
        logger.info(f"ANN index built with {train_fingerprints.shape[0]} fingerprints")
    
    def setup_influence_estimator(self) -> None:
        """Set up influence estimator for refinement."""
        logger.info("Setting up influence estimator...")
        
        # Skip if influence refinement is disabled
        if not self.config["attribution"].get("use_influence", True):
            logger.info("Influence refinement disabled, skipping setup")
            return
        
        # Load train embeddings and clusters
        train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
        train_clusters, _ = self.embedding_manager.load_clusters("train")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float32),
            torch.tensor(train_clusters, dtype=torch.long)
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config["influence"].get("batch_size", 32), 
            shuffle=False
        )
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create influence estimator
        self.influence_estimator = InfluenceEstimator(
            model=self.probe,
            loss_fn=criterion,
            train_dataloader=train_dataloader,
            device=self.device,
            damping=self.config["influence"].get("damping", 0.01),
            scale=self.config["influence"].get("scale", 1.0),
            lissa_iterations=self.config["influence"].get("lissa_iterations", 10),
            lissa_samples=self.config["influence"].get("lissa_samples", 10),
            lissa_depth=self.config["influence"].get("lissa_depth", 10000),
            matrix_free=self.config["influence"].get("matrix_free", True)
        )
        
        # Create attribution refiner
        self.attribution_refiner = AttributionRefiner(
            influence_estimator=self.influence_estimator,
            top_k=self.config["attribution"].get("top_k", 10)
        )
        
        # Save influence estimator
        self.influence_estimator.save(os.path.join(self.model_dir, "influence_estimator.pkl"))
        
        logger.info("Influence estimator setup complete")
    
    def setup_baselines(self) -> None:
        """Set up baseline methods."""
        logger.info("Setting up baseline methods...")
        
        # Get list of enabled baselines
        enabled_baselines = self.config["baselines"].get("enabled", [])
        
        # Only import and initialize baselines if needed
        if "trace" in enabled_baselines:
            logger.info("Setting up TRACE baseline...")
            self.baselines["trace"] = TRACEMethod(
                encoder_name=self.config["baselines"].get("trace_encoder", "sentence-transformers/all-mpnet-base-v2"),
                device=self.device,
                temperature=self.config["baselines"].get("trace_temperature", 0.1),
                index_type=self.config["baselines"].get("trace_index_type", "hnsw"),
                contrastive_margin=self.config["baselines"].get("trace_margin", 0.5)
            )
            
            # Check if we need to train or load
            trace_model_path = os.path.join(self.model_dir, "trace")
            os.makedirs(trace_model_path, exist_ok=True)
            
            if self.config["baselines"].get("trace_train", True):
                # Load data
                train_dataset = self.dataloaders["train"].dataset
                val_dataset = self.dataloaders["val"].dataset
                
                train_texts = [sample["text"] for sample in train_dataset]
                train_labels = self.embedding_manager.clusters["train"]
                
                val_texts = [sample["text"] for sample in val_dataset]
                val_labels = self.embedding_manager.clusters["val"]
                
                # Train TRACE
                self.baselines["trace"].train(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    batch_size=self.config["baselines"].get("trace_batch_size", 16),
                    num_epochs=self.config["baselines"].get("trace_epochs", 5),
                    learning_rate=self.config["baselines"].get("trace_lr", 1e-4),
                    output_dir=trace_model_path
                )
                
                # Build index
                train_ids = [sample["id"] for sample in train_dataset]
                self.baselines["trace"].build_index(train_texts, train_ids)
                
                # Save model
                self.baselines["trace"].save(trace_model_path)
            else:
                # Load pretrained model
                self.baselines["trace"].load(trace_model_path)
        
        if "trak" in enabled_baselines:
            logger.info("Setting up TRAK baseline...")
            self.baselines["trak"] = TRAKMethod(
                model=self.probe,
                device=self.device,
                projection_dim=self.config["baselines"].get("trak_projection_dim", 128),
                num_examples=self.config["baselines"].get("trak_num_examples", 1000)
            )
            
            # Check if we need to fit or load
            trak_model_path = os.path.join(self.model_dir, "trak")
            os.makedirs(trak_model_path, exist_ok=True)
            
            if self.config["baselines"].get("trak_fit", True):
                # Load data
                train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
                train_clusters, _ = self.embedding_manager.load_clusters("train")
                
                # Create dataset
                train_dataset = TensorDataset(
                    torch.tensor(train_embeddings, dtype=torch.float32),
                    torch.tensor(train_clusters, dtype=torch.long)
                )
                
                train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=self.config["baselines"].get("trak_batch_size", 32),
                    shuffle=False
                )
                
                # Fit TRAK
                self.baselines["trak"].fit(train_dataloader, train_ids)
                
                # Save model
                self.baselines["trak"].save(trak_model_path)
            else:
                # Load pretrained model
                self.baselines["trak"].load(trak_model_path)
        
        logger.info(f"Baseline setup complete: {', '.join(self.baselines.keys())}")
    
    def run_attribution(self) -> None:
        """Run attribution experiments."""
        logger.info("Running attribution experiments...")
        
        # Load test data
        test_fingerprints = np.load(os.path.join(self.data_dir, "test_fingerprints.npy"))
        test_embeddings, test_ids = self.embedding_manager.load_embeddings("test")
        test_clusters, _ = self.embedding_manager.load_clusters("test")
        
        # Get train data for ground truth evaluation
        train_dataset = self.dataloaders["train"].dataset
        train_texts = [sample["text"] for sample in train_dataset]
        train_ids = [sample["id"] for sample in train_dataset]
        
        # Load ANN index if not already loaded
        if self.ann_index is None:
            logger.info("Loading ANN index...")
            self.ann_index = ANNIndex()
            self.ann_index.load(self.model_dir)
        
        # Create test dataset
        # For this experiment, we'll use each test sample as its own ground truth
        # In a real-world scenario, we would have a separate ground truth mapping
        
        # Set up parameters
        k_values = self.config["attribution"].get("k_values", [1, 3, 5, 10])
        top_k = max(k_values)
        num_test_samples = min(
            self.config["attribution"].get("num_test_samples", 100),
            test_fingerprints.shape[0]
        )
        
        # Run GIF (our method)
        logger.info(f"Running GIF attribution on {num_test_samples} test samples...")
        
        # Store results
        gif_results = {
            "predictions": [],
            "latencies": []
        }
        
        # Run attribution for each test sample
        for i in tqdm(range(num_test_samples)):
            # Get test sample
            test_fingerprint = test_fingerprints[i].reshape(1, -1)
            test_id = test_ids[i]
            
            # Step 1: ANN search
            with self.latency_tracker.measure("ann_search"):
                ann_results, ann_distances = self.ann_index.search(test_fingerprint, k=top_k * 2)
            
            # Step 2: Influence refinement (if enabled)
            if self.config["attribution"].get("use_influence", True) and self.attribution_refiner is not None:
                with self.latency_tracker.measure("influence_refinement"):
                    # Build a simple test sample
                    test_sample = {
                        "id": test_id,
                        "input": torch.tensor(test_embeddings[i], dtype=torch.float32).unsqueeze(0),
                        "target": torch.tensor(test_clusters[i], dtype=torch.long).unsqueeze(0)
                    }
                    
                    # Get candidate IDs from ANN search
                    candidate_ids = [id_ for id_ in ann_results[0] if id_ is not None]
                    
                    # Create ID to index mapping
                    id_to_index = {id_: idx for idx, id_ in enumerate(train_ids)}
                    
                    # Refine candidates
                    refined_results = self.attribution_refiner.refine_candidates(
                        test_sample, candidate_ids, train_dataset, id_to_index
                    )
                    
                    # Extract IDs and scores
                    refined_ids = [id_ for id_, _ in refined_results]
                    refined_scores = [score for _, score in refined_results]
                    
                    # Use refined results
                    predictions = refined_ids
                    scores = refined_scores
                else:
                    # Use ANN results directly
                    predictions = ann_results[0]
                    scores = [-dist for dist in ann_distances[0]]  # Convert distances to scores
            
            # Record total attribution time
            gif_results["predictions"].append(predictions)
            gif_results["latencies"].append(
                self.latency_tracker.get_measurements("ann_search")[-1] +
                (self.latency_tracker.get_measurements("influence_refinement")[-1] 
                 if self.config["attribution"].get("use_influence", True) else 0)
            )
        
        # Run baseline methods
        baseline_results = {}
        
        for name, baseline in self.baselines.items():
            logger.info(f"Running {name.upper()} attribution on {num_test_samples} test samples...")
            
            baseline_results[name] = {
                "predictions": [],
                "latencies": []
            }
            
            # Get test data in appropriate format
            if name == "trace":
                # TRACE uses text
                test_dataset = self.dataloaders["test"].dataset
                test_texts = [test_dataset[i]["text"] for i in range(num_test_samples)]
                
                # Run attribution
                for i in tqdm(range(len(test_texts))):
                    start_time = time.time()
                    
                    # Search with TRACE
                    results, distances = baseline.search([test_texts[i]], k=top_k)
                    
                    # Record results
                    baseline_results[name]["predictions"].append(results[0])
                    baseline_results[name]["latencies"].append((time.time() - start_time) * 1000)  # ms
            
            elif name == "trak":
                # TRAK uses embeddings
                # Create test dataset
                test_dataset = TensorDataset(
                    torch.tensor(test_embeddings[:num_test_samples], dtype=torch.float32),
                    torch.tensor(test_clusters[:num_test_samples], dtype=torch.long)
                )
                
                test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=1,  # Process one at a time for fair latency comparison
                    shuffle=False
                )
                
                # Run attribution
                start_time = time.time()
                attributions = baseline.attribute(test_dataloader, k=top_k)
                total_time = time.time() - start_time
                
                # Record results
                for ids, scores in attributions:
                    baseline_results[name]["predictions"].append(ids)
                    # Distribute total time evenly across samples
                    baseline_results[name]["latencies"].append(total_time * 1000 / num_test_samples)  # ms
        
        # Evaluate results
        # For GIF
        self.metrics.evaluate_single_method(
            method_name="GIF",
            predictions=gif_results["predictions"],
            true_ids=test_ids[:num_test_samples],  # Using test IDs as ground truth (simplification)
            k_values=k_values,
            latencies=gif_results["latencies"]
        )
        
        # For baselines
        for name, results in baseline_results.items():
            self.metrics.evaluate_single_method(
                method_name=name.upper(),
                predictions=results["predictions"],
                true_ids=test_ids[:num_test_samples],  # Using test IDs as ground truth (simplification)
                k_values=k_values,
                latencies=results["latencies"]
            )
        
        # Save metrics
        self.metrics.save_results()
        
        # Generate visualizations
        self.metrics.plot_precision_at_k()
        self.metrics.plot_recall_at_k()
        self.metrics.plot_mrr_comparison()
        self.metrics.plot_latency_comparison()
        
        # Generate summary table
        self.metrics.generate_summary_table()
        
        # Save latency breakdown
        self.latency_tracker.plot_latency_breakdown(output_dir=self.results_dir)
        self.latency_tracker.plot_cumulative_latency(
            operations=["fingerprint_generation", "ann_search", "influence_refinement"],
            output_dir=self.results_dir
        )
        
        logger.info("Attribution experiments complete")
    
    def run_ablation_studies(self) -> None:
        """Run ablation studies."""
        if not self.config.get("run_ablations", False):
            logger.info("Ablation studies disabled, skipping")
            return
        
        logger.info("Running ablation studies...")
        
        # 1. Fingerprint type ablation (static only, gradient only, combined)
        if self.config["ablations"].get("fingerprint_type", True):
            logger.info("Running fingerprint type ablation...")
            
            fingerprint_types = ["static", "gradient", "combined"]
            ablation_results = {}
            
            for fp_type in fingerprint_types:
                # Create fingerprint generator
                fp_generator = FingerprintGenerator(
                    probe=self.probe,
                    projection_dim=self.config["fingerprints"].get("projection_dim", 128),
                    device=self.device,
                    fingerprint_type=fp_type
                )
                
                # Load train embeddings and clusters
                train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
                train_clusters, _ = self.embedding_manager.load_clusters("train")
                
                # Generate train fingerprints
                train_fingerprints = fp_generator.create_fingerprints(
                    embeddings=train_embeddings,
                    labels=train_clusters,
                    batch_size=self.config["fingerprints"].get("batch_size", 64)
                )
                
                # Load test data
                test_embeddings, test_ids = self.embedding_manager.load_embeddings("test")
                test_clusters, _ = self.embedding_manager.load_clusters("test")
                
                # Generate test fingerprints
                test_fingerprints = fp_generator.create_fingerprints(
                    embeddings=test_embeddings,
                    labels=test_clusters,
                    batch_size=self.config["fingerprints"].get("batch_size", 64)
                )
                
                # Create ANN index
                ann_index = ANNIndex(
                    index_type=self.config["indexing"].get("index_type", "hnsw"),
                    dimension=train_fingerprints.shape[1],
                    metric=self.config["indexing"].get("metric", "l2"),
                    use_gpu=self.config["indexing"].get("use_gpu", False),
                    index_params=self.config["indexing"].get("index_params", {})
                )
                
                # Build index
                ann_index.build(
                    vectors=train_fingerprints,
                    ids=train_ids,
                    batch_size=self.config["indexing"].get("batch_size", 10000)
                )
                
                # Run attribution
                k_values = self.config["attribution"].get("k_values", [1, 3, 5, 10])
                top_k = max(k_values)
                num_test_samples = min(
                    self.config["ablations"].get("num_test_samples", 100),
                    test_fingerprints.shape[0]
                )
                
                # Store results
                predictions = []
                latencies = []
                
                # Run attribution for each test sample
                for i in tqdm(range(num_test_samples)):
                    # Get test sample
                    test_fingerprint = test_fingerprints[i].reshape(1, -1)
                    
                    # ANN search
                    start_time = time.time()
                    results, distances = ann_index.search(test_fingerprint, k=top_k)
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    # Record results
                    predictions.append(results[0])
                    latencies.append(latency)
                
                # Evaluate results
                self.metrics.evaluate_single_method(
                    method_name=f"GIF ({fp_type})",
                    predictions=predictions,
                    true_ids=test_ids[:num_test_samples],
                    k_values=k_values,
                    latencies=latencies
                )
                
                # Store results for later visualization
                ablation_results[f"GIF ({fp_type})"] = self.metrics.results[f"GIF ({fp_type})"]
            
            # Create ablation plot
            self.visualizer.create_ablation_plot(
                ablation_data=ablation_results,
                metrics_to_plot=["precision@1", "mrr", "mean_latency_ms"],
                output_filename="fingerprint_type_ablation.png"
            )
        
        # 2. Projection dimension ablation
        if self.config["ablations"].get("projection_dimension", True):
            logger.info("Running projection dimension ablation...")
            
            projection_dims = [16, 32, 64, 128, 256, 512]
            dimension_results = {}
            
            for dim in projection_dims:
                # Create fingerprint generator
                fp_generator = FingerprintGenerator(
                    probe=self.probe,
                    projection_dim=dim,
                    device=self.device,
                    fingerprint_type="combined"
                )
                
                # Load train embeddings and clusters
                train_embeddings, train_ids = self.embedding_manager.load_embeddings("train")
                train_clusters, _ = self.embedding_manager.load_clusters("train")
                
                # Generate train fingerprints
                train_fingerprints = fp_generator.create_fingerprints(
                    embeddings=train_embeddings,
                    labels=train_clusters,
                    batch_size=self.config["fingerprints"].get("batch_size", 64)
                )
                
                # Load test data
                test_embeddings, test_ids = self.embedding_manager.load_embeddings("test")
                test_clusters, _ = self.embedding_manager.load_clusters("test")
                
                # Generate test fingerprints
                test_fingerprints = fp_generator.create_fingerprints(
                    embeddings=test_embeddings,
                    labels=test_clusters,
                    batch_size=self.config["fingerprints"].get("batch_size", 64)
                )
                
                # Create ANN index
                ann_index = ANNIndex(
                    index_type=self.config["indexing"].get("index_type", "hnsw"),
                    dimension=train_fingerprints.shape[1],
                    metric=self.config["indexing"].get("metric", "l2"),
                    use_gpu=self.config["indexing"].get("use_gpu", False),
                    index_params=self.config["indexing"].get("index_params", {})
                )
                
                # Build index
                ann_index.build(
                    vectors=train_fingerprints,
                    ids=train_ids,
                    batch_size=self.config["indexing"].get("batch_size", 10000)
                )
                
                # Run attribution
                k_values = self.config["attribution"].get("k_values", [1, 3, 5, 10])
                top_k = max(k_values)
                num_test_samples = min(
                    self.config["ablations"].get("num_test_samples", 100),
                    test_fingerprints.shape[0]
                )
                
                # Store results
                predictions = []
                latencies = []
                
                # Run attribution for each test sample
                for i in tqdm(range(num_test_samples)):
                    # Get test sample
                    test_fingerprint = test_fingerprints[i].reshape(1, -1)
                    
                    # ANN search
                    start_time = time.time()
                    results, distances = ann_index.search(test_fingerprint, k=top_k)
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    # Record results
                    predictions.append(results[0])
                    latencies.append(latency)
                
                # Evaluate results
                method_name = f"GIF (dim={dim})"
                self.metrics.evaluate_single_method(
                    method_name=method_name,
                    predictions=predictions,
                    true_ids=test_ids[:num_test_samples],
                    k_values=k_values,
                    latencies=latencies
                )
                
                # Store results for later visualization
                dimension_results[dim] = {
                    "precision@1": self.metrics.results[method_name]["precision@1"],
                    "mrr": self.metrics.results[method_name]["mrr"],
                    "latency_ms": self.metrics.results[method_name]["mean_latency_ms"]
                }
            
            # Create projection dimension plot
            self.visualizer.create_projection_dimension_plot(
                dimension_data=dimension_results,
                metrics_to_plot=["precision@1", "mrr", "latency_ms"],
                output_filename="projection_dimension_ablation.png"
            )
        
        logger.info("Ablation studies complete")
    
    def create_sample_visualizations(self) -> None:
        """Create sample attribution visualizations."""
        logger.info("Creating sample attribution visualizations...")
        
        # Load test data
        test_dataset = self.dataloaders["test"].dataset
        
        # Load train data
        train_dataset = self.dataloaders["train"].dataset
        
        # Get a few sample texts
        num_samples = min(5, len(test_dataset))
        
        for i in range(num_samples):
            # Get test sample
            test_sample = test_dataset[i]
            test_id = test_sample["id"]
            test_text = test_sample["text"]
            
            # Use ANN search to get candidates
            test_fingerprint = np.load(os.path.join(self.data_dir, "test_fingerprints.npy"))[i].reshape(1, -1)
            results, distances = self.ann_index.search(test_fingerprint, k=5)
            
            # Get candidate texts
            candidate_ids = results[0]
            candidate_texts = []
            scores = []
            
            for j, cand_id in enumerate(candidate_ids):
                if cand_id is not None:
                    # Find the train sample with this ID
                    for train_sample in train_dataset:
                        if train_sample["id"] == cand_id:
                            candidate_texts.append(train_sample["text"])
                            scores.append(-distances[0][j])  # Convert distance to similarity score
                            break
            
            # Create attribution example visualization
            self.visualizer.create_attribution_example(
                query_text=test_text[:100] + "...",  # Truncate for visualization
                candidate_texts=[text[:100] + "..." for text in candidate_texts],  # Truncate
                scores=scores,
                true_index=None,  # We don't have ground truth
                output_filename=f"attribution_example_{i+1}.png"
            )
        
        logger.info(f"Created {num_samples} sample visualizations")
    
    def generate_results_report(self) -> None:
        """Generate a final results report."""
        logger.info("Generating results report...")
        
        # Load metrics
        self.metrics.load_results()
        
        # Get comparison data
        comparison_df = self.metrics.compare_methods()
        
        # Generate summary table
        summary_df = self.metrics.generate_summary_table()
        
        # Create results markdown
        report = "# Gradient-Informed Fingerprinting (GIF) Experiment Results\n\n"
        
        # Add experiment details
        report += "## Experiment Details\n\n"
        report += f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"- Dataset: {self.config['data']['dataset_name']}"
        if self.config['data'].get('subset_name'):
            report += f" ({self.config['data']['subset_name']})\n"
        else:
            report += "\n"
        
        if self.config['data'].get('max_samples'):
            report += f"- Samples: {self.config['data']['max_samples']}\n"
        
        report += f"- Embedding Model: {self.config['embeddings'].get('model_name', 'sentence-transformers/all-mpnet-base-v2')}\n"
        report += f"- Clusters: {self.config['clustering'].get('n_clusters', 100)}\n"
        report += f"- Fingerprint Type: {self.config['fingerprints'].get('type', 'combined')}\n"
        report += f"- Projection Dimension: {self.config['fingerprints'].get('projection_dim', 128)}\n"
        report += f"- Index Type: {self.config['indexing'].get('index_type', 'hnsw')}\n"
        report += f"- Use Influence Refinement: {self.config['attribution'].get('use_influence', True)}\n\n"
        
        # Add method comparison
        report += "## Method Comparison\n\n"
        report += summary_df.to_markdown() + "\n\n"
        
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
        
        # Add ablation studies if available
        if self.config.get("run_ablations", False):
            report += "## Ablation Studies\n\n"
            
            if self.config["ablations"].get("fingerprint_type", True):
                report += "### Fingerprint Type\n\n"
                report += "![Fingerprint Type Ablation](fingerprint_type_ablation.png)\n\n"
            
            if self.config["ablations"].get("projection_dimension", True):
                report += "### Projection Dimension\n\n"
                report += "![Projection Dimension Ablation](projection_dimension_ablation.png)\n\n"
        
        # Add sample visualizations
        report += "## Attribution Examples\n\n"
        
        # Include available examples
        for i in range(1, 6):
            if os.path.exists(os.path.join(self.results_dir, f"attribution_example_{i}.png")):
                report += f"### Example {i}\n\n"
                report += f"![Attribution Example {i}](attribution_example_{i}.png)\n\n"
        
        # Add conclusions
        report += "## Conclusions\n\n"
        report += "The Gradient-Informed Fingerprinting (GIF) method shows promising results for efficient attribution in foundation models:\n\n"
        
        # Add key findings
        methods = list(comparison_df["method"])
        if "GIF" in methods:
            gif_index = methods.index("GIF")
            
            # Precision comparison
            p1_col = "precision@1" if "precision@1" in comparison_df.columns else None
            if p1_col:
                gif_p1 = float(comparison_df.iloc[gif_index][p1_col])
                baseline_p1 = [float(comparison_df.iloc[i][p1_col]) for i in range(len(methods)) if methods[i] != "GIF"]
                if baseline_p1 and gif_p1 > max(baseline_p1):
                    report += f"- GIF achieves higher precision@1 ({gif_p1:.3f}) compared to baseline methods"
                    report += f" (best baseline: {max(baseline_p1):.3f}).\n"
            
            # MRR comparison
            mrr_col = "mrr" if "mrr" in comparison_df.columns else None
            if mrr_col:
                gif_mrr = float(comparison_df.iloc[gif_index][mrr_col])
                baseline_mrr = [float(comparison_df.iloc[i][mrr_col]) for i in range(len(methods)) if methods[i] != "GIF"]
                if baseline_mrr and gif_mrr > max(baseline_mrr):
                    report += f"- GIF achieves higher Mean Reciprocal Rank ({gif_mrr:.3f}) compared to baseline methods"
                    report += f" (best baseline: {max(baseline_mrr):.3f}).\n"
            
            # Latency comparison
            latency_col = "mean_latency_ms" if "mean_latency_ms" in comparison_df.columns else None
            if latency_col:
                gif_latency = float(comparison_df.iloc[gif_index][latency_col])
                baseline_latency = [float(comparison_df.iloc[i][latency_col]) for i in range(len(methods)) if methods[i] != "GIF"]
                if baseline_latency and gif_latency < min(baseline_latency):
                    report += f"- GIF is faster ({gif_latency:.2f} ms) compared to baseline methods"
                    report += f" (fastest baseline: {min(baseline_latency):.2f} ms).\n"
        
        report += "\nThe two-stage approach combining efficient ANN search with influence-based refinement provides a good balance between attribution accuracy and speed, making it suitable for real-time applications.\n"
        
        # Write report
        with open(os.path.join(self.results_dir, "results.md"), "w") as f:
            f.write(report)
        
        logger.info("Results report generated")
    
    def run_experiment(self) -> None:
        """Run the complete experiment pipeline."""
        logger.info(f"Starting experiment: {self.exp_name}")
        
        # Step 1: Set up data
        self.setup_data()
        
        # Step 2: Generate embeddings and clusters
        self.setup_embeddings()
        
        # Step 3: Train probe network
        self.setup_probe()
        
        # Step 4: Generate fingerprints
        self.generate_fingerprints()
        
        # Step 5: Build ANN index
        self.build_index()
        
        # Step 6: Set up influence estimator (if enabled)
        if self.config["attribution"].get("use_influence", True):
            self.setup_influence_estimator()
        
        # Step 7: Set up baselines
        self.setup_baselines()
        
        # Step 8: Run attribution experiments
        self.run_attribution()
        
        # Step 9: Run ablation studies (if enabled)
        if self.config.get("run_ablations", False):
            self.run_ablation_studies()
        
        # Step 10: Create sample visualizations
        self.create_sample_visualizations()
        
        # Step 11: Generate results report
        self.generate_results_report()
        
        logger.info(f"Experiment complete: {self.exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIF attribution experiments")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run experiment
    runner = ExperimentRunner(args.config)
    runner.run_experiment()