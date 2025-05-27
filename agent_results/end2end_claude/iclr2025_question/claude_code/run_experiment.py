"""
Main experiment runner for Reasoning Uncertainty Networks (RUNs) evaluation.
"""
import os
import json
import logging
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    OUTPUTS_DIR, 
    DATASET_CONFIG, 
    EVAL_CONFIG, 
    RUNS_CONFIG,
    BASELINE_CONFIG,
    EXPERIMENT_CONFIG
)
from data import DatasetLoader, load_all_datasets
from model import LLMInterface, ReasoningUncertaintyNetwork
from uncertainty import create_uq_method
from evaluation import ModelEvaluator, ComparisonEvaluator, StatisticalAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUTS_DIR, "log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(use_gpu: bool = True) -> None:
    """
    Set up the experiment environment.
    
    Args:
        use_gpu: Whether to use GPU acceleration
    """
    logger.info("Setting up experiment...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Set up GPU if available and requested
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("GPU not available, using CPU")
        except ImportError:
            logger.info("PyTorch not available, using CPU")
    else:
        logger.info("Using CPU as requested")
    
    # Log experiment configuration
    logger.info(f"Experiment name: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info(f"Random seed: {EVAL_CONFIG['random_seed']}")
    
    # Set random seeds for reproducibility
    np.random.seed(EVAL_CONFIG['random_seed'])
    try:
        import torch
        torch.manual_seed(EVAL_CONFIG['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(EVAL_CONFIG['random_seed'])
    except ImportError:
        pass

def load_datasets():
    """
    Load and prepare datasets for the experiment.
    
    Returns:
        Dictionary of datasets
    """
    logger.info("Loading datasets...")
    
    # Load all datasets (scientific, legal, medical)
    datasets = load_all_datasets(
        test_size=0.2,
        val_size=0.1,
        seed=EVAL_CONFIG['random_seed']
    )
    
    # Log dataset statistics
    for dataset_type, splits in datasets.items():
        logger.info(f"Dataset: {dataset_type}")
        for split_name, dataset in splits.items():
            if split_name != "hallucination_indices":
                logger.info(f"  {split_name}: {len(dataset)} examples")
        
        # Log hallucination distribution in test set
        test_dataset = splits["test"]
        num_hallucinations = sum(test_dataset["contains_hallucination"])
        hallucination_rate = num_hallucinations / len(test_dataset)
        logger.info(f"  Test set hallucination rate: {hallucination_rate:.2f} ({num_hallucinations}/{len(test_dataset)})")
    
    return datasets

def run_runs_experiment(datasets: Dict, dataset_type: str = "scientific"):
    """
    Run the Reasoning Uncertainty Networks experiment.
    
    Args:
        datasets: Dictionary of datasets
        dataset_type: Type of dataset to use ("scientific", "legal", "medical")
        
    Returns:
        Tuple of (evaluator, model)
    """
    logger.info(f"Running RUNs experiment on {dataset_type} dataset...")
    
    # Initialize LLM interface
    llm = LLMInterface()
    
    # Initialize RUNs model
    runs_model = ReasoningUncertaintyNetwork(llm)
    
    # Get test dataset
    test_dataset = datasets[dataset_type]["test"]
    
    # Initialize evaluator
    evaluator = ModelEvaluator("runs", "runs")
    
    # Process each test example
    for i in tqdm(range(EVAL_CONFIG["num_test_examples"]), desc="Evaluating RUNs"):
        if i >= len(test_dataset):
            break
        
        # Get test example
        example = test_dataset[i]
        question = example["question"]
        context = example["context"]
        contains_hallucination = example["contains_hallucination"]
        
        # Process through RUNs
        try:
            graph, hallucination_nodes = runs_model.process(
                question, 
                context,
                visualize=(i < 5),  # Only visualize the first 5 examples
                output_dir=os.path.join(OUTPUTS_DIR, f"runs_example_{i}")
            )
            
            # Get hallucination detection result
            is_hallucination = len(hallucination_nodes) > 0
            
            # Get hallucination score (maximum across nodes)
            if hallucination_nodes:
                hallucination_score = max([
                    graph.nodes[node_id].get("hallucination_score", 0)
                    for node_id in hallucination_nodes
                ])
            else:
                hallucination_score = 0.0
            
            # Add to evaluator
            evaluator.add_prediction(is_hallucination, contains_hallucination, hallucination_score)
            
            # Save detailed results for the example
            if i < 10:  # Save details for the first 10 examples
                runs_model.save_results(
                    os.path.join(OUTPUTS_DIR, f"runs_example_{i}_results.json")
                )
        
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    logger.info(f"RUNs metrics: {metrics}")
    
    # Create visualizations
    evaluator.visualize_calibration(os.path.join(OUTPUTS_DIR, "runs_calibration.png"))
    evaluator.visualize_confusion_matrix(os.path.join(OUTPUTS_DIR, "runs_confusion.png"))
    evaluator.visualize_roc_curve(os.path.join(OUTPUTS_DIR, "runs_roc.png"))
    evaluator.visualize_pr_curve(os.path.join(OUTPUTS_DIR, "runs_pr.png"))
    
    # Save results
    evaluator.save_results(os.path.join(OUTPUTS_DIR, "runs_results.json"))
    
    return evaluator, runs_model

def run_baseline_experiment(datasets: Dict, baseline_name: str, dataset_type: str = "scientific"):
    """
    Run a baseline method experiment.
    
    Args:
        datasets: Dictionary of datasets
        baseline_name: Name of the baseline method
        dataset_type: Type of dataset to use ("scientific", "legal", "medical")
        
    Returns:
        ModelEvaluator instance
    """
    logger.info(f"Running {baseline_name} experiment on {dataset_type} dataset...")
    
    # Initialize LLM interface
    llm = LLMInterface()
    
    # Initialize baseline method
    baseline_method = create_uq_method(baseline_name, llm)
    
    # Get test dataset
    test_dataset = datasets[dataset_type]["test"]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(baseline_name, "baseline")
    
    # Special handling for calibration-based method
    if baseline_name == "calibration" and BASELINE_CONFIG["calibration"]["validation_size"] > 0:
        # Get validation dataset for calibration
        val_dataset = datasets[dataset_type]["val"]
        
        # Create validation data for calibration
        val_size = min(len(val_dataset), BASELINE_CONFIG["calibration"]["validation_size"])
        val_data = []
        
        for i in range(val_size):
            example = val_dataset[i]
            val_data.append((
                example["question"],
                example["context"],
                example["contains_hallucination"]
            ))
        
        # Calibrate the method
        baseline_method.calibrate(val_data)
    
    # Process each test example
    for i in tqdm(range(EVAL_CONFIG["num_test_examples"]), desc=f"Evaluating {baseline_name}"):
        if i >= len(test_dataset):
            break
        
        # Get test example
        example = test_dataset[i]
        question = example["question"]
        context = example["context"]
        contains_hallucination = example["contains_hallucination"]
        
        # Process through baseline method
        try:
            is_hallucination, hallucination_score, details = baseline_method.detect_hallucination(
                question, context
            )
            
            # Add to evaluator
            evaluator.add_prediction(is_hallucination, contains_hallucination, hallucination_score)
            
            # Save detailed results for the example
            if i < 5:  # Save details for the first 5 examples
                with open(os.path.join(OUTPUTS_DIR, f"{baseline_name}_example_{i}_results.json"), "w") as f:
                    json.dump({
                        "question": question,
                        "context": context[:500] + "..." if len(context) > 500 else context,
                        "contains_hallucination": contains_hallucination,
                        "predicted_hallucination": is_hallucination,
                        "hallucination_score": hallucination_score,
                        "details": {k: v for k, v in details.items() if k != "samples" and k != "responses"}
                    }, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error processing example {i} with {baseline_name}: {e}")
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    logger.info(f"{baseline_name} metrics: {metrics}")
    
    # Create visualizations
    evaluator.visualize_calibration(os.path.join(OUTPUTS_DIR, f"{baseline_name}_calibration.png"))
    evaluator.visualize_confusion_matrix(os.path.join(OUTPUTS_DIR, f"{baseline_name}_confusion.png"))
    evaluator.visualize_roc_curve(os.path.join(OUTPUTS_DIR, f"{baseline_name}_roc.png"))
    evaluator.visualize_pr_curve(os.path.join(OUTPUTS_DIR, f"{baseline_name}_pr.png"))
    
    # Save results
    evaluator.save_results(os.path.join(OUTPUTS_DIR, f"{baseline_name}_results.json"))
    
    return evaluator

def compare_models(dataset_type: str = "scientific"):
    """
    Compare all models and create visualizations.
    
    Args:
        dataset_type: Type of dataset used for evaluation
    """
    logger.info("Comparing all models...")
    
    # List of models to compare
    models = ["runs", "selfcheckgpt", "multidim_uq", "calibration", "hudex", "metaqa"]
    model_types = ["runs", "baseline", "baseline", "baseline", "baseline", "baseline"]
    
    # Initialize comparison evaluator
    comparison = ComparisonEvaluator(models, model_types)
    
    # Load results
    comparison.load_results(OUTPUTS_DIR)
    
    # Create comparison table
    comparison_df = comparison.compare_metrics(os.path.join(OUTPUTS_DIR, "comparison_table.csv"))
    logger.info(f"Comparison table:\n{comparison_df}")
    
    # Create visualizations for individual metrics
    for metric in EVAL_CONFIG["metrics"]:
        if metric in ["precision", "recall", "f1", "auroc", "auprc"]:
            comparison.visualize_metric_comparison(
                metric,
                os.path.join(OUTPUTS_DIR, f"comparison_{metric}.png")
            )
    
    # Create combined visualization for precision, recall, F1
    comparison._visualize_combined_metrics(
        ["precision", "recall", "f1"],
        os.path.join(OUTPUTS_DIR, "comparison_prf1.png")
    )
    
    # Create combined visualization for AUROC and AUPRC
    comparison._visualize_combined_metrics(
        ["auroc", "auprc"],
        os.path.join(OUTPUTS_DIR, "comparison_roc_pr.png")
    )
    
    # Create visualization for false positive and false negative rates
    comparison._visualize_combined_metrics(
        ["false_positive_rate", "false_negative_rate"],
        os.path.join(OUTPUTS_DIR, "comparison_error_rates.png")
    )
    
    # Plot hallucination rate vs model
    create_hallucination_rate_plot(
        comparison_df,
        os.path.join(OUTPUTS_DIR, "comparison_hallucination_rate.png")
    )

def create_hallucination_rate_plot(comparison_df: pd.DataFrame, output_path: str):
    """
    Create a plot comparing hallucination detection rates across models.
    
    Args:
        comparison_df: DataFrame with model comparison results
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Extract the metrics
    df = comparison_df.copy()
    
    # If we have true_positives and false_positives columns, compute hallucination rate
    if "true_positives" in df.columns and "false_positives" in df.columns:
        df["predicted_hallucination_rate"] = (df["true_positives"] + df["false_positives"]) / (
            df["true_positives"] + df["false_positives"] + df["true_negatives"] + df["false_negatives"]
        )
        
        # Sort by model type and hallucination rate
        df = df.sort_values(by=["model_type", "predicted_hallucination_rate"], ascending=[True, False])
        
        # Create bar colors based on model type
        colors = df["model_type"].map({"runs": "blue", "baseline": "orange"})
        
        # Create the bar chart
        bars = plt.bar(
            range(len(df)), 
            df["predicted_hallucination_rate"],
            color=colors
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=10
            )
        
        # Add a legend for model types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="blue", label="RUNs"),
            Patch(facecolor="orange", label="Baseline")
        ]
        plt.legend(handles=legend_elements)
        
        plt.xticks(range(len(df)), df.index, rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Predicted Hallucination Rate")
        plt.title("Comparison of Hallucination Detection Rates")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

def create_results_summary():
    """
    Create a summary of experiment results.
    
    Returns:
        Summary string
    """
    logger.info("Creating results summary...")
    
    # Get results files
    results_files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith("_results.json")]
    
    # Load results
    results = {}
    for file in results_files:
        if "example" in file:
            continue  # Skip example results
        
        model_name = file.split("_results.json")[0]
        
        with open(os.path.join(OUTPUTS_DIR, file), "r") as f:
            results[model_name] = json.load(f)
    
    # Create summary
    summary = f"# Reasoning Uncertainty Networks (RUNs) Experiment Results\n\n"
    summary += f"Experiment: {EXPERIMENT_CONFIG['name']}\n"
    summary += f"Description: {EXPERIMENT_CONFIG['description']}\n"
    summary += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add experiment configuration
    summary += f"## Experiment Configuration\n\n"
    summary += f"- Evaluation metrics: {', '.join(EVAL_CONFIG['metrics'])}\n"
    summary += f"- Number of test examples: {EVAL_CONFIG['num_test_examples']}\n"
    summary += f"- Random seed: {EVAL_CONFIG['random_seed']}\n"
    summary += f"- Number of runs: {EVAL_CONFIG['num_runs']}\n\n"
    
    # Add main results table
    summary += f"## Main Results\n\n"
    summary += f"### Performance on Hallucination Detection\n\n"
    
    summary += "| Model | Precision | Recall | F1 | AUROC | AUPRC |\n"
    summary += "| ----- | --------- | ------ | -- | ----- | ----- |\n"
    
    # Sort models with runs first, then alphabetically
    model_names = sorted(results.keys(), key=lambda x: (0 if x == "runs" else 1, x))
    
    for model_name in model_names:
        if "metrics" not in results[model_name]:
            continue
        
        metrics = results[model_name]["metrics"]
        
        precision = metrics.get("precision", "N/A")
        if isinstance(precision, float):
            precision = f"{precision:.4f}"
        
        recall = metrics.get("recall", "N/A")
        if isinstance(recall, float):
            recall = f"{recall:.4f}"
        
        f1 = metrics.get("f1", "N/A")
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
        
        auroc = metrics.get("auroc", "N/A")
        if isinstance(auroc, float):
            auroc = f"{auroc:.4f}"
        
        auprc = metrics.get("auprc", "N/A")
        if isinstance(auprc, float):
            auprc = f"{auprc:.4f}"
        
        summary += f"| {model_name} | {precision} | {recall} | {f1} | {auroc} | {auprc} |\n"
    
    summary += "\n"
    
    # Add error rate comparison
    summary += f"### Error Rates\n\n"
    
    summary += "| Model | False Positive Rate | False Negative Rate |\n"
    summary += "| ----- | ------------------- | ------------------- |\n"
    
    for model_name in model_names:
        if "metrics" not in results[model_name]:
            continue
        
        metrics = results[model_name]["metrics"]
        
        fpr = metrics.get("false_positive_rate", "N/A")
        if isinstance(fpr, float):
            fpr = f"{fpr:.4f}"
        
        fnr = metrics.get("false_negative_rate", "N/A")
        if isinstance(fnr, float):
            fnr = f"{fnr:.4f}"
        
        summary += f"| {model_name} | {fpr} | {fnr} |\n"
    
    summary += "\n"
    
    # Add visualization references
    summary += f"## Visualizations\n\n"
    
    summary += f"### Performance Comparison\n\n"
    summary += f"![Precision, Recall, F1 Comparison](comparison_prf1.png)\n\n"
    summary += f"![ROC and PR Curve Comparison](comparison_roc_pr.png)\n\n"
    summary += f"![Error Rate Comparison](comparison_error_rates.png)\n\n"
    
    # Add RUNs-specific visualizations
    summary += f"### RUNs Visualizations\n\n"
    summary += f"![RUNs Calibration](runs_calibration.png)\n\n"
    summary += f"![RUNs Confusion Matrix](runs_confusion.png)\n\n"
    
    # Add Hallucination Rate comparison
    summary += f"### Hallucination Rate Comparison\n\n"
    summary += f"![Hallucination Rate Comparison](comparison_hallucination_rate.png)\n\n"
    
    # Add analysis and discussion
    summary += f"## Analysis and Discussion\n\n"
    
    # Extract key metrics for RUNs
    runs_metrics = results.get("runs", {}).get("metrics", {})
    runs_f1 = runs_metrics.get("f1", 0)
    
    # Find best baseline
    best_baseline = None
    best_baseline_f1 = 0
    
    for model_name in results:
        if model_name == "runs" or "metrics" not in results[model_name]:
            continue
        
        f1 = results[model_name]["metrics"].get("f1", 0)
        if f1 > best_baseline_f1:
            best_baseline_f1 = f1
            best_baseline = model_name
    
    # Add discussion of results
    if runs_f1 > best_baseline_f1:
        summary += f"The Reasoning Uncertainty Networks (RUNs) approach outperformed all baseline methods "
        summary += f"in terms of F1 score, achieving {runs_f1:.4f} compared to the best baseline "
        summary += f"({best_baseline}) at {best_baseline_f1:.4f}. This represents a "
        summary += f"{(runs_f1 - best_baseline_f1) / best_baseline_f1 * 100:.1f}% improvement.\n\n"
    else:
        summary += f"The Reasoning Uncertainty Networks (RUNs) approach achieved an F1 score of {runs_f1:.4f}, "
        summary += f"which was {'comparable to' if runs_f1 > 0.95 * best_baseline_f1 else 'lower than'} "
        summary += f"the best baseline ({best_baseline}) at {best_baseline_f1:.4f}.\n\n"
    
    summary += "### Key Findings\n\n"
    
    summary += "1. **Uncertainty Propagation:** The RUNs approach explicitly represents and propagates uncertainty "
    summary += "through the reasoning chain, providing more transparency into the sources of uncertainty.\n\n"
    
    summary += "2. **Explanatory Power:** Unlike black-box methods, RUNs provides explanations for why certain "
    summary += "statements might be hallucinated, tracing back to uncertain premises or logical inconsistencies.\n\n"
    
    summary += "3. **Fine-grained Detection:** The graph-based approach allows for detection of hallucinations at "
    summary += "specific points in the reasoning chain, rather than simply classifying entire responses.\n\n"
    
    summary += "### Limitations\n\n"
    
    summary += "1. **Computational Overhead:** The RUNs approach requires multiple LLM calls for constructing the "
    summary += "reasoning graph and initializing uncertainties, which increases latency compared to simpler methods.\n\n"
    
    summary += "2. **Dependence on Initial Graph Construction:** The quality of the reasoning graph construction "
    summary += "directly impacts the effectiveness of uncertainty propagation. Future work could explore more robust "
    summary += "methods for extracting reasoning structures from LLM outputs.\n\n"
    
    summary += "3. **Domain Adaptation:** While our approach aims to be domain-agnostic, optimal performance may "
    summary += "require domain-specific adaptations, particularly in specialized fields like medicine or law.\n\n"
    
    summary += "### Future Work\n\n"
    
    summary += "1. **Integration with Retrieval-Augmented Generation:** Future work could explore tighter integration "
    summary += "between our uncertainty propagation framework and retrieval-augmented generation approaches, potentially "
    summary += "enabling dynamic retrieval based on identified high-uncertainty nodes.\n\n"
    
    summary += "2. **Hierarchical Reasoning Graphs:** As reasoning chains become extremely complex, the graph "
    summary += "representation may become unwieldy. Future work could explore hierarchical graph representations "
    summary += "or other abstractions to manage complexity.\n\n"
    
    summary += "3. **Interactive Human-AI Collaboration:** The transparent representation of uncertainty enables "
    summary += "more effective collaboration between humans and AI systems. Future work could explore interfaces "
    summary += "that allow human experts to interact with the reasoning graph and provide targeted corrections.\n\n"
    
    return summary

def create_results_folder():
    """
    Create the results folder and move relevant files into it.
    """
    logger.info("Creating results folder...")
    
    # Create results folder
    results_dir = os.path.join(os.path.dirname(os.path.dirname(OUTPUTS_DIR)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy results.md
    shutil.copy(
        os.path.join(OUTPUTS_DIR, "results.md"),
        os.path.join(results_dir, "results.md")
    )
    
    # Copy log.txt
    shutil.copy(
        os.path.join(OUTPUTS_DIR, "log.txt"),
        os.path.join(results_dir, "log.txt")
    )
    
    # Copy all PNG figures
    for file in os.listdir(OUTPUTS_DIR):
        if file.endswith(".png"):
            shutil.copy(
                os.path.join(OUTPUTS_DIR, file),
                os.path.join(results_dir, file)
            )
    
    logger.info(f"Results folder created at {results_dir}")

def run_experiment(args):
    """
    Run the full experiment.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    logger.info("Starting experiment...")
    
    # Setup experiment environment
    setup_experiment(use_gpu=args.use_gpu)
    
    # Load datasets
    datasets = load_datasets()
    
    # Run RUNs experiment
    runs_evaluator, runs_model = run_runs_experiment(datasets, args.dataset)
    
    # Run baseline experiments
    baseline_evaluators = {}
    for baseline in args.baselines:
        baseline_evaluators[baseline] = run_baseline_experiment(datasets, baseline, args.dataset)
    
    # Compare all models
    compare_models(args.dataset)
    
    # Create results summary
    summary = create_results_summary()
    
    # Save summary to results.md
    with open(os.path.join(OUTPUTS_DIR, "results.md"), "w") as f:
        f.write(summary)
    
    # Create results folder
    create_results_folder()
    
    # Log experiment completion
    total_time = time.time() - start_time
    logger.info(f"Experiment completed in {total_time:.2f} seconds")

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run Reasoning Uncertainty Networks experiment")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["scientific", "legal", "medical"],
        default="scientific",
        help="Dataset type to use for evaluation"
    )
    
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        choices=["selfcheckgpt", "multidim_uq", "calibration", "hudex", "metaqa"],
        default=["selfcheckgpt", "multidim_uq", "calibration", "hudex", "metaqa"],
        help="Baseline methods to include in the experiment"
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of test examples to use (overrides EVAL_CONFIG)"
    )
    
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU acceleration if available"
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.num_examples is not None:
        EVAL_CONFIG["num_test_examples"] = args.num_examples
    
    return args

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)