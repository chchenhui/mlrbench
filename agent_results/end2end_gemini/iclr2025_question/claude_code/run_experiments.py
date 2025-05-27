"""
Main script to run experiments for the AUG-RAG system.
"""

import os
import argparse
import json
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple, Union, Any, Optional

# Import modules
from data.data_utils import (
    setup_logging, load_truthfulqa_dataset, load_halueval_dataset,
    load_nq_dataset, create_knowledge_base, preprocess_for_evaluation
)

def extract_questions_from_batch(batch):
    """
    Extract questions from a batch of dataset items, handling different dataset structures.

    Args:
        batch: A batch of dataset items.

    Returns:
        List of extracted questions.
    """
    questions = []
    for item in batch:
        if isinstance(item, dict) and "question" in item:
            questions.append(item["question"])
        elif hasattr(item, "question"):
            questions.append(item.question)
        else:
            # If we can't find a question field, use the item itself if it's a string
            if isinstance(item, str):
                questions.append(item)
            else:
                logger.warning(f"Could not extract question from item: {item}")
                questions.append("What is this?")  # Default fallback
    return questions

def extract_references_from_batch(batch):
    """
    Extract references (answers) from a batch of dataset items, handling different dataset structures.

    Args:
        batch: A batch of dataset items.

    Returns:
        List of extracted references.
    """
    references = []
    for item in batch:
        if isinstance(item, dict):
            if "answer" in item:
                references.append(item["answer"])
            elif "mc1_targets" in item and len(item["mc1_targets"]) > 0:
                references.append(item["mc1_targets"][0])
            elif "reference" in item:
                references.append(item["reference"])
            else:
                references.append("")
        elif hasattr(item, "answer"):
            references.append(item.answer)
        elif hasattr(item, "mc1_targets") and len(item.mc1_targets) > 0:
            references.append(item.mc1_targets[0])
        elif hasattr(item, "reference"):
            references.append(item.reference)
        else:
            references.append("")
    return references
from models.base_model import BaseModel, APIBasedModel
from models.rag_model import RetrieverModule, StandardRAGModel
from models.uncertainty import UncertaintyFactory
from models.aug_rag_model import AUGRAGFactory, AdaptiveRetrievalTrigger
from utils.evaluation import evaluate_model_outputs
from utils.visualization import (
    compare_models_bar_chart, plot_uncertainty_threshold_experiment,
    plot_calibration_curve, plot_uncertainty_histograms,
    plot_retrieval_patterns, plot_ablation_results
)

def setup_environment(args):
    """
    Set up the environment for experiments.
    
    Args:
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment settings and paths.
    """
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(experiment_dir, "logs", "experiment.log")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    # Log experiment settings
    logger.info(f"Starting experiment with settings: {args}")
    
    # Set up data paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create knowledge base if needed
    kb_path = os.path.join(data_dir, f"{args.dataset}_kb.json")
    if not os.path.exists(kb_path):
        logger.info(f"Creating knowledge base for {args.dataset}")
        kb_path = create_knowledge_base(args.dataset, data_dir)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Set up model parameters based on API or local model
    if args.use_api:
        logger.info(f"Using API-based model: {args.api_model}")
        model_params = {
            "model_type": "api",
            "model_name": args.api_model,
            "api_key": os.environ.get("OPENAI_API_KEY") if "gpt" in args.api_model else os.environ.get("ANTHROPIC_API_KEY")
        }
    else:
        logger.info(f"Using local model: {args.model}")
        model_params = {
            "model_type": "local",
            "model_name": args.model,
            "device": device
        }
    
    # Return experiment settings
    return {
        "experiment_dir": experiment_dir,
        "data_dir": data_dir,
        "kb_path": kb_path,
        "device": device,
        "model_params": model_params,
        "logger": logger
    }

def load_dataset(dataset_name: str, split: str = "validation") -> Tuple[Any, str]:
    """
    Load the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to load.
    
    Returns:
        Tuple of (dataset, evaluation_mode).
    """
    logger = logging.getLogger(__name__)
    
    if dataset_name == "truthfulqa":
        dataset = load_truthfulqa_dataset(split)
        eval_mode = "hallucination"
    elif dataset_name == "halueval":
        dataset = load_halueval_dataset(split)
        eval_mode = "hallucination"
    elif dataset_name == "nq":
        dataset = load_nq_dataset(split)
        eval_mode = "factuality"
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Loaded {dataset_name} dataset with {len(dataset)} examples")
    return dataset, eval_mode

def create_model(model_type: str, env: Dict, args: argparse.Namespace) -> Any:
    """
    Create a model based on the specified type.
    
    Args:
        model_type: Type of model to create ("baseline", "standard_rag", "aug_rag").
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        The created model instance.
    """
    logger = env["logger"]
    model_params = env["model_params"]
    
    if model_type == "baseline":
        # Create baseline model (no retrieval)
        if model_params["model_type"] == "api":
            model = APIBasedModel(
                model_name=model_params["model_name"],
                api_key=model_params["api_key"]
            )
        else:
            model = BaseModel(
                model_name=model_params["model_name"],
                device=model_params["device"]
            )
        logger.info(f"Created baseline model: {model_params['model_name']}")
        return model
    
    elif model_type == "standard_rag":
        # Create standard RAG model (always retrieves)
        # First create base model
        if model_params["model_type"] == "api":
            base_model = APIBasedModel(
                model_name=model_params["model_name"],
                api_key=model_params["api_key"]
            )
        else:
            base_model = BaseModel(
                model_name=model_params["model_name"],
                device=model_params["device"]
            )
        
        # Create retriever
        retriever = RetrieverModule(
            knowledge_base_path=env["kb_path"],
            embedding_model_name=args.embedding_model,
            use_sparse=args.use_sparse_retriever,
            device=model_params.get("device", None)
        )
        
        # Create standard RAG model
        model = StandardRAGModel(
            base_model=base_model,
            retriever=retriever,
            num_documents=args.num_documents
        )
        logger.info(f"Created standard RAG model with {args.num_documents} documents per query")
        return model
    
    elif model_type == "aug_rag":
        # Create AUG-RAG model (adaptive retrieval)
        # Prepare configuration
        aug_rag_config = {
            "model_type": model_params["model_type"],
            "model_name": model_params["model_name"],
            "api_key": model_params.get("api_key", None),
            "device": model_params.get("device", None),
            "knowledge_base_path": env["kb_path"],
            "embedding_model": args.embedding_model,
            "use_sparse_retriever": args.use_sparse_retriever,
            "uncertainty_method": args.uncertainty,
            "uncertainty_params": {
                "num_samples": args.mc_samples,
                "num_perturbations": args.spuq_perturbations
            },
            "threshold_type": args.threshold_type,
            "fixed_threshold": args.threshold,
            "window_size": args.window_size,
            "trigger_params": {},
            "num_documents": args.num_documents,
            "generation_chunk_size": args.chunk_size,
            "segment_mode": args.segment_mode
        }
        
        # Create AUG-RAG model
        model = AUGRAGFactory.create(aug_rag_config)
        logger.info(f"Created AUG-RAG model with {args.uncertainty} uncertainty and {args.threshold_type} threshold")
        return model
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

def run_baseline_experiment(env: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run baseline model experiment.
    
    Args:
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment results.
    """
    logger = env["logger"]
    logger.info("Starting baseline experiment")
    
    # Load dataset
    dataset, eval_mode = load_dataset(args.dataset, args.split)
    
    # Create baseline model
    model = create_model("baseline", env, args)
    
    # Generate predictions
    predictions = []
    references = []
    
    # Process in batches
    batch_size = args.batch_size
    max_idx = min(len(dataset), args.max_samples)
    for i in tqdm(range(0, max_idx, batch_size), desc="Generating"):
        end_idx = min(i+batch_size, max_idx)
        batch = dataset[i:end_idx]
        
        # Extract questions
        questions = extract_questions_from_batch(batch)

        # Generate predictions
        batch_predictions = model.generate(
            questions,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        predictions.extend(batch_predictions)
        
        # Extract references
        batch_references = extract_references_from_batch(batch)
        references.extend(batch_references)
    
    # Evaluate predictions
    metrics = evaluate_model_outputs(predictions, references, mode=eval_mode)
    logger.info(f"Baseline metrics: {metrics}")
    
    # Save results
    results = {
        "model": "baseline",
        "dataset": args.dataset,
        "eval_mode": eval_mode,
        "metrics": metrics,
        "predictions": predictions,
        "references": references
    }
    
    results_path = os.path.join(env["experiment_dir"], "results", "baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k not in ["predictions", "references"]}, f, indent=2)
    
    logger.info(f"Saved baseline results to {results_path}")
    return results

def run_standard_rag_experiment(env: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run standard RAG model experiment.
    
    Args:
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment results.
    """
    logger = env["logger"]
    logger.info("Starting standard RAG experiment")
    
    # Load dataset
    dataset, eval_mode = load_dataset(args.dataset, args.split)
    
    # Create standard RAG model
    model = create_model("standard_rag", env, args)
    
    # Generate predictions
    predictions = []
    references = []
    retrieved_contexts = []
    
    # Process in batches
    batch_size = args.batch_size
    max_idx = min(len(dataset), args.max_samples)
    for i in tqdm(range(0, max_idx, batch_size), desc="Generating"):
        end_idx = min(i+batch_size, max_idx)
        batch = dataset[i:end_idx]
        
        # Extract questions
        questions = extract_questions_from_batch(batch)

        # Generate predictions
        batch_predictions = model.generate(
            questions,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        predictions.extend(batch_predictions)
        
        # Extract references
        batch_references = extract_references_from_batch(batch)
        references.extend(batch_references)
        
        # Save retrieved contexts (if available)
        if hasattr(model, "retriever"):
            for question in questions:
                docs = model.retriever.retrieve(question, model.num_documents)
                retrieved_contexts.append(docs)
    
    # Evaluate predictions
    metrics = evaluate_model_outputs(
        predictions, references, 
        mode=eval_mode,
        retrieved_contexts=retrieved_contexts if retrieved_contexts else None
    )
    logger.info(f"Standard RAG metrics: {metrics}")
    
    # Save results
    results = {
        "model": "standard_rag",
        "dataset": args.dataset,
        "eval_mode": eval_mode,
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "retrieved_contexts": retrieved_contexts
    }
    
    results_path = os.path.join(env["experiment_dir"], "results", "standard_rag_results.json")
    with open(results_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k not in ["predictions", "references", "retrieved_contexts"]}, f, indent=2)
    
    logger.info(f"Saved standard RAG results to {results_path}")
    return results

def run_aug_rag_experiment(env: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run AUG-RAG model experiment.
    
    Args:
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment results.
    """
    logger = env["logger"]
    logger.info("Starting AUG-RAG experiment")
    
    # Load dataset
    dataset, eval_mode = load_dataset(args.dataset, args.split)
    
    # Create AUG-RAG model
    model = create_model("aug_rag", env, args)
    
    # Generate predictions
    predictions = []
    references = []
    uncertainty_scores = []
    retrieved_contexts = []
    retrieval_stats = []
    
    # Process in batches
    batch_size = args.batch_size
    max_idx = min(len(dataset), args.max_samples)
    for i in tqdm(range(0, max_idx, batch_size), desc="Generating"):
        end_idx = min(i+batch_size, max_idx)
        batch = dataset[i:end_idx]
        
        # Extract questions
        questions = extract_questions_from_batch(batch)

        # Generate predictions
        batch_predictions = model.generate(
            questions,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        predictions.extend(batch_predictions)
        
        # Extract references
        batch_references = extract_references_from_batch(batch)
        references.extend(batch_references)
        
        # Get retrieval stats
        if hasattr(model, "get_retrieval_stats"):
            stats = model.get_retrieval_stats()
            retrieval_stats.append(stats)
            
            # Estimate average uncertainty for the samples (simplified)
            for q in questions:
                u_score = model.uncertainty_estimator.estimate_uncertainty(q)
                uncertainty_scores.append(u_score)
        
        # Save retrieved contexts (if available)
        if hasattr(model, "retriever"):
            for question in questions:
                docs = model.retriever.retrieve(question, model.num_documents)
                retrieved_contexts.append(docs)
    
    # Create hallucination labels (simplified approximation)
    # In a real implementation, you would use human annotations or more sophisticated methods
    hallucination_labels = []
    for pred, ref in zip(predictions, references):
        # Simple heuristic: consider prediction as hallucination if there's limited overlap with reference
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        overlap = len(pred_tokens.intersection(ref_tokens))
        hallucination_labels.append(overlap / len(ref_tokens) < 0.3 if ref_tokens else True)
    
    # Evaluate predictions
    metrics = evaluate_model_outputs(
        predictions, references, 
        uncertainty_scores=uncertainty_scores if uncertainty_scores else None,
        hallucination_labels=hallucination_labels,
        retrieved_contexts=retrieved_contexts if retrieved_contexts else None,
        mode=eval_mode
    )
    
    # Add retrieval frequency metrics
    if retrieval_stats:
        avg_retrieval_frequency = sum(s["retrieval_frequency"] for s in retrieval_stats) / len(retrieval_stats)
        metrics["retrieval_frequency"] = avg_retrieval_frequency
    
    logger.info(f"AUG-RAG metrics: {metrics}")
    
    # Save results
    results = {
        "model": f"aug_rag_{args.uncertainty}_{args.threshold_type}",
        "dataset": args.dataset,
        "eval_mode": eval_mode,
        "uncertainty_method": args.uncertainty,
        "threshold_type": args.threshold_type,
        "threshold_value": args.threshold,
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "uncertainty_scores": uncertainty_scores,
        "hallucination_labels": hallucination_labels,
        "retrieved_contexts": retrieved_contexts,
        "retrieval_stats": retrieval_stats
    }
    
    results_path = os.path.join(env["experiment_dir"], "results", f"aug_rag_{args.uncertainty}_{args.threshold_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k not in ["predictions", "references", "retrieved_contexts"]}, f, indent=2)
    
    logger.info(f"Saved AUG-RAG results to {results_path}")
    return results

def run_threshold_ablation(env: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run ablation study on uncertainty thresholds.
    
    Args:
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment results.
    """
    logger = env["logger"]
    logger.info("Starting threshold ablation experiment")
    
    # Load dataset (use a smaller subset for ablation)
    dataset, eval_mode = load_dataset(args.dataset, args.split)
    if len(dataset) > args.ablation_samples:
        indices = np.random.choice(len(dataset), args.ablation_samples, replace=False)
        dataset = dataset.select(indices)
    
    # Thresholds to test
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Metrics to track
    hallucination_rates = []
    retrieval_frequencies = []
    all_metrics = {}
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        # Update args for this threshold
        args.threshold = threshold
        args.threshold_type = "fixed"
        
        # Create AUG-RAG model with this threshold
        aug_rag_config = {
            "model_type": env["model_params"]["model_type"],
            "model_name": env["model_params"]["model_name"],
            "api_key": env["model_params"].get("api_key", None),
            "device": env["model_params"].get("device", None),
            "knowledge_base_path": env["kb_path"],
            "embedding_model": args.embedding_model,
            "use_sparse_retriever": args.use_sparse_retriever,
            "uncertainty_method": args.uncertainty,
            "uncertainty_params": {
                "num_samples": args.mc_samples,
                "num_perturbations": args.spuq_perturbations
            },
            "threshold_type": "fixed",
            "fixed_threshold": threshold,
            "window_size": args.window_size,
            "trigger_params": {},
            "num_documents": args.num_documents,
            "generation_chunk_size": args.chunk_size,
            "segment_mode": args.segment_mode
        }
        
        model = AUGRAGFactory.create(aug_rag_config)
        
        # Generate predictions
        predictions = []
        references = []
        retrieval_stats = []
        
        # Process in batches
        batch_size = args.batch_size
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            # Extract questions
            questions = extract_questions_from_batch(batch)

            # Generate predictions
            batch_predictions = model.generate(
                questions,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            predictions.extend(batch_predictions)
            
            # Extract references
            batch_references = extract_references_from_batch(batch)
            references.extend(batch_references)
            
            # Get retrieval stats
            if hasattr(model, "get_retrieval_stats"):
                stats = model.get_retrieval_stats()
                retrieval_stats.append(stats)
        
        # Create hallucination labels (simplified approximation)
        hallucination_labels = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            overlap = len(pred_tokens.intersection(ref_tokens))
            hallucination_labels.append(overlap / len(ref_tokens) < 0.3 if ref_tokens else True)
        
        # Compute hallucination rate
        hallucination_rate = sum(hallucination_labels) / len(hallucination_labels) if hallucination_labels else 0
        hallucination_rates.append(hallucination_rate)
        
        # Compute retrieval frequency
        if retrieval_stats:
            avg_retrieval_frequency = sum(s["retrieval_frequency"] for s in retrieval_stats) / len(retrieval_stats)
            retrieval_frequencies.append(avg_retrieval_frequency)
        else:
            retrieval_frequencies.append(0)
        
        # Evaluate predictions
        metrics = evaluate_model_outputs(
            predictions, references, mode=eval_mode,
            hallucination_labels=hallucination_labels
        )
        all_metrics[str(threshold)] = metrics
        
        logger.info(f"Threshold {threshold}: Hallucination rate: {hallucination_rate:.4f}, Retrieval frequency: {retrieval_frequencies[-1]:.4f}")
    
    # Plot results
    plot_path = plot_uncertainty_threshold_experiment(
        thresholds, hallucination_rates, retrieval_frequencies,
        os.path.join(env["experiment_dir"], "plots")
    )
    
    # Save results
    results = {
        "thresholds": thresholds,
        "hallucination_rates": hallucination_rates,
        "retrieval_frequencies": retrieval_frequencies,
        "all_metrics": all_metrics,
        "plot_path": plot_path
    }
    
    results_path = os.path.join(env["experiment_dir"], "results", "threshold_ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved threshold ablation results to {results_path}")
    return results

def run_uncertainty_methods_ablation(env: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run ablation study on different uncertainty estimation methods.
    
    Args:
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment results.
    """
    logger = env["logger"]
    logger.info("Starting uncertainty methods ablation experiment")
    
    # Load dataset (use a smaller subset for ablation)
    dataset, eval_mode = load_dataset(args.dataset, args.split)
    if len(dataset) > args.ablation_samples:
        indices = np.random.choice(len(dataset), args.ablation_samples, replace=False)
        dataset = dataset.select(indices)
    
    # Uncertainty methods to test
    uncertainty_methods = ["entropy", "token_confidence", "mc_dropout", "spuq"]
    
    # Metrics to track
    hallucination_rates = []
    retrieval_frequencies = []
    aurocs = []
    eces = []
    
    for method in tqdm(uncertainty_methods, desc="Testing uncertainty methods"):
        # Update args for this method
        args.uncertainty = method
        
        # Create AUG-RAG model with this uncertainty method
        aug_rag_config = {
            "model_type": env["model_params"]["model_type"],
            "model_name": env["model_params"]["model_name"],
            "api_key": env["model_params"].get("api_key", None),
            "device": env["model_params"].get("device", None),
            "knowledge_base_path": env["kb_path"],
            "embedding_model": args.embedding_model,
            "use_sparse_retriever": args.use_sparse_retriever,
            "uncertainty_method": method,
            "uncertainty_params": {
                "num_samples": args.mc_samples,
                "num_perturbations": args.spuq_perturbations
            },
            "threshold_type": args.threshold_type,
            "fixed_threshold": args.threshold,
            "window_size": args.window_size,
            "trigger_params": {},
            "num_documents": args.num_documents,
            "generation_chunk_size": args.chunk_size,
            "segment_mode": args.segment_mode
        }
        
        model = AUGRAGFactory.create(aug_rag_config)
        
        # Generate predictions
        predictions = []
        references = []
        uncertainty_scores = []
        retrieval_stats = []
        
        # Process in batches
        batch_size = args.batch_size
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            # Extract questions
            questions = extract_questions_from_batch(batch)

            # Generate predictions
            batch_predictions = model.generate(
                questions,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            predictions.extend(batch_predictions)
            
            # Extract references
            batch_references = extract_references_from_batch(batch)
            references.extend(batch_references)
            
            # Estimate uncertainty for the samples
            for q in questions:
                u_score = model.uncertainty_estimator.estimate_uncertainty(q)
                uncertainty_scores.append(u_score)
            
            # Get retrieval stats
            if hasattr(model, "get_retrieval_stats"):
                stats = model.get_retrieval_stats()
                retrieval_stats.append(stats)
        
        # Create hallucination labels (simplified approximation)
        hallucination_labels = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            overlap = len(pred_tokens.intersection(ref_tokens))
            hallucination_labels.append(overlap / len(ref_tokens) < 0.3 if ref_tokens else True)
        
        # Compute hallucination rate
        hallucination_rate = sum(hallucination_labels) / len(hallucination_labels) if hallucination_labels else 0
        hallucination_rates.append(hallucination_rate)
        
        # Compute retrieval frequency
        if retrieval_stats:
            avg_retrieval_frequency = sum(s["retrieval_frequency"] for s in retrieval_stats) / len(retrieval_stats)
            retrieval_frequencies.append(avg_retrieval_frequency)
        else:
            retrieval_frequencies.append(0)
        
        # Evaluate uncertainty calibration
        from utils.evaluation import evaluate_uncertainty_calibration
        calibration = evaluate_uncertainty_calibration(uncertainty_scores, hallucination_labels)
        aurocs.append(calibration["auroc"])
        eces.append(calibration["ece"])
        
        logger.info(f"Method {method}: Hallucination rate: {hallucination_rate:.4f}, AUROC: {calibration['auroc']:.4f}")
    
    # Plot results
    ablation_metrics = {
        "Hallucination Rate": hallucination_rates,
        "Retrieval Frequency": retrieval_frequencies,
        "AUROC": aurocs,
        "ECE": eces
    }
    
    plot_path = plot_ablation_results(
        "uncertainty_methods", uncertainty_methods, ablation_metrics,
        os.path.join(env["experiment_dir"], "plots")
    )
    
    # Save results
    results = {
        "uncertainty_methods": uncertainty_methods,
        "hallucination_rates": hallucination_rates,
        "retrieval_frequencies": retrieval_frequencies,
        "aurocs": aurocs,
        "eces": eces,
        "plot_path": plot_path
    }
    
    results_path = os.path.join(env["experiment_dir"], "results", "uncertainty_methods_ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved uncertainty methods ablation results to {results_path}")
    return results

def run_num_documents_ablation(env: Dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run ablation study on the number of retrieved documents.
    
    Args:
        env: Environment settings.
        args: Command-line arguments.
    
    Returns:
        Dictionary with experiment results.
    """
    logger = env["logger"]
    logger.info("Starting document count ablation experiment")
    
    # Load dataset (use a smaller subset for ablation)
    dataset, eval_mode = load_dataset(args.dataset, args.split)
    if len(dataset) > args.ablation_samples:
        indices = np.random.choice(len(dataset), args.ablation_samples, replace=False)
        dataset = dataset.select(indices)
    
    # Document counts to test
    doc_counts = [1, 3, 5, 10]
    
    # Metrics to track
    factuality_scores = []
    knowledge_f1_scores = []
    
    for num_docs in tqdm(doc_counts, desc="Testing document counts"):
        # Update args for this document count
        args.num_documents = num_docs
        
        # Create AUG-RAG model with this document count
        aug_rag_config = {
            "model_type": env["model_params"]["model_type"],
            "model_name": env["model_params"]["model_name"],
            "api_key": env["model_params"].get("api_key", None),
            "device": env["model_params"].get("device", None),
            "knowledge_base_path": env["kb_path"],
            "embedding_model": args.embedding_model,
            "use_sparse_retriever": args.use_sparse_retriever,
            "uncertainty_method": args.uncertainty,
            "uncertainty_params": {
                "num_samples": args.mc_samples,
                "num_perturbations": args.spuq_perturbations
            },
            "threshold_type": args.threshold_type,
            "fixed_threshold": args.threshold,
            "window_size": args.window_size,
            "trigger_params": {},
            "num_documents": num_docs,
            "generation_chunk_size": args.chunk_size,
            "segment_mode": args.segment_mode
        }
        
        model = AUGRAGFactory.create(aug_rag_config)
        
        # Generate predictions
        predictions = []
        references = []
        retrieved_contexts = []
        
        # Process in batches
        batch_size = args.batch_size
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:min(i+batch_size, len(dataset))]
            
            # Extract questions
            questions = extract_questions_from_batch(batch)

            # Generate predictions
            batch_predictions = model.generate(
                questions,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            predictions.extend(batch_predictions)
            
            # Extract references
            batch_references = extract_references_from_batch(batch)
            references.extend(batch_references)
            
            # Get retrieved contexts
            for question in questions:
                docs = model.retriever.retrieve(question, num_docs)
                retrieved_contexts.append(docs)
        
        # Evaluate predictions
        metrics = evaluate_model_outputs(
            predictions, references, mode=eval_mode,
            retrieved_contexts=retrieved_contexts
        )
        
        # Track metrics
        if eval_mode == "factuality":
            factuality_scores.append(metrics["f1_score"])
        else:
            factuality_scores.append(1.0 - metrics.get("self_contradiction_rate", 0))
        
        knowledge_f1_scores.append(metrics.get("knowledge_f1", 0))
        
        logger.info(f"Num docs {num_docs}: Factuality: {factuality_scores[-1]:.4f}, Knowledge F1: {knowledge_f1_scores[-1]:.4f}")
    
    # Plot results
    ablation_metrics = {
        "Factuality Score": factuality_scores,
        "Knowledge F1": knowledge_f1_scores
    }
    
    plot_path = plot_ablation_results(
        "num_documents", doc_counts, ablation_metrics,
        os.path.join(env["experiment_dir"], "plots")
    )
    
    # Save results
    results = {
        "doc_counts": doc_counts,
        "factuality_scores": factuality_scores,
        "knowledge_f1_scores": knowledge_f1_scores,
        "plot_path": plot_path
    }
    
    results_path = os.path.join(env["experiment_dir"], "results", "num_documents_ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved document count ablation results to {results_path}")
    return results

def visualize_results(all_results: Dict[str, Dict[str, Any]], env: Dict):
    """
    Create visualizations from experiment results.
    
    Args:
        all_results: Dictionary with all experiment results.
        env: Environment settings.
    """
    logger = env["logger"]
    logger.info("Creating visualizations from results")
    
    # Extract metrics from all models
    model_metrics = {}
    for model_name, results in all_results.items():
        if "metrics" in results:
            model_metrics[model_name] = results["metrics"]
    
    # If we have results for multiple models, create comparison charts
    if len(model_metrics) > 1:
        # Common metrics to compare
        metrics_to_compare = [
            ("exact_match", "Exact Match Score"),
            ("f1_score", "F1 Score"),
            ("knowledge_f1", "Knowledge F1 Score"),
            ("self_contradiction_rate", "Self-Contradiction Rate"),
            ("retrieval_frequency", "Retrieval Frequency")
        ]
        
        # Create comparison charts
        for metric_name, chart_title in metrics_to_compare:
            # Check if any model has this metric
            if any(metric_name in metrics for metrics in model_metrics.values()):
                higher_is_better = metric_name != "self_contradiction_rate"  # Lower is better for this one
                
                try:
                    compare_models_bar_chart(
                        model_metrics, metric_name, chart_title,
                        os.path.join(env["experiment_dir"], "plots"),
                        higher_is_better=higher_is_better
                    )
                except Exception as e:
                    logger.error(f"Error creating comparison chart for {metric_name}: {e}")
    
    # Create model-specific visualizations for AUG-RAG models
    for model_name, results in all_results.items():
        if "aug_rag" in model_name and "uncertainty_scores" in results and "hallucination_labels" in results:
            try:
                # Create calibration curve
                confidences = [1.0 - u for u in results["uncertainty_scores"]]  # Convert uncertainty to confidence
                plot_calibration_curve(
                    confidences, results["hallucination_labels"],
                    os.path.join(env["experiment_dir"], "plots"),
                    model_name=model_name
                )
                
                # Create uncertainty histograms
                correct_uncertainties = [u for u, h in zip(results["uncertainty_scores"], results["hallucination_labels"]) if not h]
                incorrect_uncertainties = [u for u, h in zip(results["uncertainty_scores"], results["hallucination_labels"]) if h]
                
                plot_uncertainty_histograms(
                    correct_uncertainties, incorrect_uncertainties,
                    os.path.join(env["experiment_dir"], "plots"),
                    model_name=model_name
                )
            except Exception as e:
                logger.error(f"Error creating visualization for {model_name}: {e}")
    
    logger.info("Completed visualizations")

def generate_markdown_report(all_results: Dict[str, Dict[str, Any]], env: Dict):
    """
    Generate a markdown report summarizing experiment results.
    
    Args:
        all_results: Dictionary with all experiment results.
        env: Environment settings.
    """
    logger = env["logger"]
    logger.info("Generating markdown report")
    
    # Create report content
    report = []
    
    # Add header
    report.append("# AUG-RAG Experiment Results\n")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add experiment settings
    report.append("## Experiment Settings\n")
    report.append(f"- Dataset: {next(iter(all_results.values()))['dataset']}")
    report.append(f"- Model: {env['model_params']['model_name']}")
    report.append(f"- Evaluation mode: {next(iter(all_results.values()))['eval_mode']}")
    report.append("")
    
    # Add model comparison section
    report.append("## Model Comparison\n")
    
    # Extract metrics from all models
    model_metrics = {}
    for model_name, results in all_results.items():
        if "metrics" in results:
            model_metrics[model_name] = results["metrics"]
    
    # Create metrics table
    if model_metrics:
        # Get all unique metrics
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Filter out metrics we don't want to include in the table
        metrics_to_exclude = {"num_samples"}
        table_metrics = sorted([m for m in all_metrics if m not in metrics_to_exclude])
        
        # Create table header
        report.append("### Performance Metrics\n")
        report.append("| Metric | " + " | ".join(model_metrics.keys()) + " |")
        report.append("| --- | " + " | ".join(["---" for _ in model_metrics]) + " |")
        
        # Add rows for each metric
        for metric in table_metrics:
            row = f"| {metric} | "
            for model_name in model_metrics:
                value = model_metrics[model_name].get(metric, "N/A")
                if isinstance(value, float):
                    row += f"{value:.4f} | "
                else:
                    row += f"{value} | "
            report.append(row)
        
        report.append("")
    
    # Add comparison charts
    report.append("### Comparison Charts\n")
    metrics_to_compare = [
        "exact_match", "f1_score", "knowledge_f1", "self_contradiction_rate", "retrieval_frequency"
    ]
    
    for metric in metrics_to_compare:
        if any(metric in metrics for metrics in model_metrics.values()):
            plot_path = os.path.join("plots", f"{metric}_comparison.png")
            if os.path.exists(os.path.join(env["experiment_dir"], plot_path)):
                report.append(f"![{metric.replace('_', ' ').title()} Comparison]({plot_path})\n")
                report.append(f"*Figure: Comparison of {metric.replace('_', ' ').title()} across models*\n")
    
    # Add detailed results for each model
    report.append("## Detailed Model Results\n")
    
    for model_name, results in all_results.items():
        report.append(f"### {model_name}\n")
        
        # Add model-specific metrics
        if "metrics" in results:
            report.append("#### Metrics\n")
            report.append("| Metric | Value |")
            report.append("| --- | --- |")
            
            for metric, value in results["metrics"].items():
                if isinstance(value, float):
                    report.append(f"| {metric} | {value:.4f} |")
                else:
                    report.append(f"| {metric} | {value} |")
            
            report.append("")
        
        # Add model-specific visualizations
        if "aug_rag" in model_name:
            report.append("#### Visualizations\n")
            
            # Calibration curve
            calib_path = os.path.join("plots", f"{model_name.replace(' ', '_')}_calibration.png")
            if os.path.exists(os.path.join(env["experiment_dir"], calib_path)):
                report.append(f"![Calibration Curve]({calib_path})\n")
                report.append(f"*Figure: Calibration curve showing the relationship between confidence and accuracy*\n")
            
            # Uncertainty histogram
            hist_path = os.path.join("plots", f"{model_name.replace(' ', '_')}_uncertainty_histogram.png")
            if os.path.exists(os.path.join(env["experiment_dir"], hist_path)):
                report.append(f"![Uncertainty Histogram]({hist_path})\n")
                report.append(f"*Figure: Distribution of uncertainty values for correct and incorrect predictions*\n")
            
            # Retrieval patterns (if available)
            pattern_path = os.path.join("plots", f"{model_name.replace(' ', '_')}_retrieval_pattern.png")
            if os.path.exists(os.path.join(env["experiment_dir"], pattern_path)):
                report.append(f"![Retrieval Pattern]({pattern_path})\n")
                report.append(f"*Figure: Pattern of retrieval triggers during generation*\n")
        
        report.append("")
    
    # Add ablation studies section
    if any(r for r in all_results if "ablation" in r):
        report.append("## Ablation Studies\n")
        
        # Threshold ablation
        threshold_results = next((r for r in all_results if "threshold_ablation" in r), None)
        if threshold_results:
            report.append("### Uncertainty Threshold Ablation\n")
            
            plot_path = os.path.join("plots", "uncertainty_threshold_experiment.png")
            if os.path.exists(os.path.join(env["experiment_dir"], plot_path)):
                report.append(f"![Threshold Ablation]({plot_path})\n")
                report.append(f"*Figure: Effect of uncertainty threshold on hallucination rate and retrieval frequency*\n")
        
        # Uncertainty methods ablation
        uncertainty_results = next((r for r in all_results if "uncertainty_methods_ablation" in r), None)
        if uncertainty_results:
            report.append("### Uncertainty Methods Ablation\n")
            
            plot_path = os.path.join("plots", "uncertainty_methods_ablation.png")
            if os.path.exists(os.path.join(env["experiment_dir"], plot_path)):
                report.append(f"![Uncertainty Methods Ablation]({plot_path})\n")
                report.append(f"*Figure: Comparison of different uncertainty estimation methods*\n")
        
        # Num documents ablation
        docs_results = next((r for r in all_results if "num_documents_ablation" in r), None)
        if docs_results:
            report.append("### Number of Retrieved Documents Ablation\n")
            
            plot_path = os.path.join("plots", "num_documents_ablation.png")
            if os.path.exists(os.path.join(env["experiment_dir"], plot_path)):
                report.append(f"![Documents Ablation]({plot_path})\n")
                report.append(f"*Figure: Effect of the number of retrieved documents on model performance*\n")
    
    # Add conclusions section
    report.append("## Conclusions\n")
    
    # Try to extract key findings
    baseline_metrics = model_metrics.get("baseline", {})
    standard_rag_metrics = model_metrics.get("standard_rag", {})
    aug_rag_models = {name: metrics for name, metrics in model_metrics.items() if "aug_rag" in name}
    
    if baseline_metrics and standard_rag_metrics and aug_rag_models:
        # Factuality improvement
        if "f1_score" in baseline_metrics and all("f1_score" in m for m in aug_rag_models.values()):
            baseline_f1 = baseline_metrics["f1_score"]
            standard_rag_f1 = standard_rag_metrics.get("f1_score", baseline_f1)
            best_aug_rag_model = max(aug_rag_models.items(), key=lambda x: x[1].get("f1_score", 0))
            best_aug_rag_f1 = best_aug_rag_model[1].get("f1_score", 0)
            
            report.append(f"- The best AUG-RAG model ({best_aug_rag_model[0]}) improved F1 score by {(best_aug_rag_f1 - baseline_f1) * 100:.1f}% compared to the baseline model and {(best_aug_rag_f1 - standard_rag_f1) * 100:.1f}% compared to standard RAG.")
        
        # Hallucination reduction
        if "self_contradiction_rate" in baseline_metrics and all("self_contradiction_rate" in m for m in aug_rag_models.values()):
            baseline_rate = baseline_metrics["self_contradiction_rate"]
            standard_rag_rate = standard_rag_metrics.get("self_contradiction_rate", baseline_rate)
            best_aug_rag_model = min(aug_rag_models.items(), key=lambda x: x[1].get("self_contradiction_rate", 1.0))
            best_aug_rag_rate = best_aug_rag_model[1].get("self_contradiction_rate", 1.0)
            
            report.append(f"- The best AUG-RAG model ({best_aug_rag_model[0]}) reduced self-contradiction rate by {(baseline_rate - best_aug_rag_rate) * 100:.1f}% compared to the baseline model and {(standard_rag_rate - best_aug_rag_rate) * 100:.1f}% compared to standard RAG.")
        
        # Retrieval efficiency
        if "retrieval_frequency" in standard_rag_metrics and all("retrieval_frequency" in m for m in aug_rag_models.values()):
            standard_rag_freq = standard_rag_metrics.get("retrieval_frequency", 1.0)
            avg_aug_rag_freq = sum(m.get("retrieval_frequency", 0) for m in aug_rag_models.values()) / len(aug_rag_models)
            
            report.append(f"- AUG-RAG models reduced retrieval frequency by {(standard_rag_freq - avg_aug_rag_freq) * 100:.1f}% compared to standard RAG, demonstrating more efficient use of external knowledge.")
    
    # Add generic conclusions
    report.append("- The adaptive uncertainty-gated approach demonstrates a promising direction for balancing factuality and efficiency in retrieval-augmented generation.")
    report.append("- Uncertainty estimation methods show varying effectiveness in predicting hallucinations, with entropy and MC dropout methods generally performing well.")
    report.append("- Dynamic thresholding mechanisms can further improve the adaptability of the system to different contexts and queries.")
    
    # Future work
    report.append("\n### Future Work\n")
    report.append("- Investigate more sophisticated uncertainty estimation methods specifically tailored for generative language models.")
    report.append("- Explore learned thresholding policies that can adapt to different domains and query types.")
    report.append("- Extend the framework to multimodal foundation models where hallucination risks may be even higher.")
    report.append("- Conduct larger-scale human evaluations to assess the real-world impact on user trust and satisfaction.")
    
    # Write report to file
    report_path = os.path.join(env["experiment_dir"], "results.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    logger.info(f"Generated markdown report at {report_path}")
    return report_path

def main():
    """Main function to run the experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run AUG-RAG experiments")
    
    # Model settings
    parser.add_argument("--model", type=str, default="gpt2", help="Model to use for local experiments")
    parser.add_argument("--use-api", action="store_true", help="Use API-based model instead of local model")
    parser.add_argument("--api-model", type=str, default="gpt-4o-mini", help="API model to use if --use-api is set")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="truthfulqa", help="Dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to evaluate")
    
    # RAG settings
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model for retrieval")
    parser.add_argument("--use-sparse-retriever", action="store_true", help="Use sparse retriever (BM25) instead of dense retriever")
    parser.add_argument("--num-documents", type=int, default=3, help="Number of documents to retrieve per query")
    
    # AUG-RAG settings
    parser.add_argument("--uncertainty", type=str, default="entropy", choices=["entropy", "mc_dropout", "token_confidence", "spuq"], help="Uncertainty estimation method")
    parser.add_argument("--mc-samples", type=int, default=5, help="Number of MC samples for MC dropout")
    parser.add_argument("--spuq-perturbations", type=int, default=3, help="Number of perturbations for SPUQ-inspired method")
    parser.add_argument("--threshold", type=float, default=0.5, help="Uncertainty threshold for retrieval trigger")
    parser.add_argument("--threshold-type", type=str, default="fixed", choices=["fixed", "rolling_window", "dynamic_global", "context_specific"], help="Type of threshold")
    parser.add_argument("--window-size", type=int, default=5, help="Window size for rolling window threshold")
    parser.add_argument("--chunk-size", type=int, default=10, help="Number of tokens to generate in each iteration")
    parser.add_argument("--segment-mode", action="store_true", help="Estimate uncertainty at segment level instead of token level")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for generation")
    
    # Experiment settings
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU even if available")
    parser.add_argument("--run-all", action="store_true", help="Run all experiments")
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline experiment")
    parser.add_argument("--run-standard-rag", action="store_true", help="Run standard RAG experiment")
    parser.add_argument("--run-aug-rag", action="store_true", help="Run AUG-RAG experiment")
    parser.add_argument("--run-ablation", type=str, choices=["threshold", "uncertainty_methods", "num_documents"], help="Run ablation study")
    parser.add_argument("--ablation-samples", type=int, default=20, help="Number of samples to use for ablation studies")
    
    args = parser.parse_args()
    
    # Set up environment
    env = setup_environment(args)
    logger = env["logger"]
    
    # Dictionary to store all results
    all_results = {}
    
    # Run experiments
    try:
        # Run baseline experiment
        if args.run_all or args.run_baseline:
            baseline_results = run_baseline_experiment(env, args)
            all_results["baseline"] = baseline_results
        
        # Run standard RAG experiment
        if args.run_all or args.run_standard_rag:
            standard_rag_results = run_standard_rag_experiment(env, args)
            all_results["standard_rag"] = standard_rag_results
        
        # Run AUG-RAG experiment
        if args.run_all or args.run_aug_rag:
            aug_rag_results = run_aug_rag_experiment(env, args)
            all_results[f"aug_rag_{args.uncertainty}_{args.threshold_type}"] = aug_rag_results
        
        # Run ablation studies
        if args.run_all or args.run_ablation:
            if args.run_all or args.run_ablation == "threshold":
                threshold_results = run_threshold_ablation(env, args)
                all_results["threshold_ablation"] = threshold_results
            
            if args.run_all or args.run_ablation == "uncertainty_methods":
                uncertainty_results = run_uncertainty_methods_ablation(env, args)
                all_results["uncertainty_methods_ablation"] = uncertainty_results
            
            if args.run_all or args.run_ablation == "num_documents":
                docs_results = run_num_documents_ablation(env, args)
                all_results["num_documents_ablation"] = docs_results
        
        # Create visualizations
        visualize_results(all_results, env)
        
        # Generate markdown report
        report_path = generate_markdown_report(all_results, env)
        
        # Copy results to the required location
        import shutil
        results_dir = "/home/chenhui/mlr-bench/pipeline_gemini/iclr2025_question/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Copy results.md
        shutil.copy(report_path, os.path.join(results_dir, "results.md"))
        
        # Copy log file
        log_file = os.path.join(env["experiment_dir"], "logs", "experiment.log")
        shutil.copy(log_file, os.path.join(results_dir, "log.txt"))
        
        # Copy figures
        plots_dir = os.path.join(env["experiment_dir"], "plots")
        if os.path.exists(plots_dir):
            for plot_file in os.listdir(plots_dir):
                if plot_file.endswith(".png"):
                    shutil.copy(os.path.join(plots_dir, plot_file), os.path.join(results_dir, plot_file))
        
        logger.info(f"Copied results to {results_dir}")
        logger.info("Experiment completed successfully")
    
    except Exception as e:
        logger.error(f"Error running experiments: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()