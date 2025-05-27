"""
Test script for the Reasoning Uncertainty Networks (RUNs) experiment.

This script runs a smaller version of the experiment for testing purposes.
"""
import os
import json
import logging
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    OUTPUTS_DIR, 
    DATASET_CONFIG, 
    EVAL_CONFIG, 
    RUNS_CONFIG,
    BASELINE_CONFIG,
    EXPERIMENT_CONFIG
)
from data import load_all_datasets
from model import LLMInterface, ReasoningUncertaintyNetwork
from uncertainty import (
    SelfCheckGPT, 
    MultiDimensionalUQ, 
    CalibrationBasedUQ, 
    HuDEx, 
    MetaQA
)
from evaluation import ModelEvaluator, ComparisonEvaluator
from utils import (
    set_random_seed, 
    setup_visualization_style,
    visualize_reasoning_graph,
    visualize_uncertainty_distributions,
    create_radar_chart,
    create_hallucination_types_chart,
    generate_synthetic_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUTS_DIR, "test_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_runs_components():
    """Test the RUNs model components individually."""
    logger.info("Testing RUNs components...")
    
    # Initialize LLM interface
    llm = LLMInterface()
    
    # Test questions
    test_questions = [
        (
            "What happens when you mix baking soda and vinegar?",
            "Baking soda is sodium bicarbonate (NaHCO3). Vinegar contains acetic acid (CH3COOH)."
        ),
        (
            "Why does the sky appear blue?",
            "Sunlight reaches Earth's atmosphere and is scattered in all directions by all the gases and particles in the air."
        )
    ]
    
    # Initialize RUNs model
    runs_model = ReasoningUncertaintyNetwork(llm)
    
    # Test on each question
    for i, (question, context) in enumerate(test_questions):
        logger.info(f"Testing question {i+1}: {question}")
        
        try:
            # Process through RUNs
            graph, hallucination_nodes = runs_model.process(
                question, 
                context,
                visualize=True,
                output_dir=os.path.join(OUTPUTS_DIR, f"test_runs_q{i+1}")
            )
            
            logger.info(f"Generated graph with {len(graph.nodes())} nodes")
            logger.info(f"Detected {len(hallucination_nodes)} potential hallucination nodes")
            
            # Print explanation
            explanation = runs_model.generate_explanation()
            logger.info(f"Explanation: {explanation}")
            
            # Save results
            runs_model.save_results(
                os.path.join(OUTPUTS_DIR, f"test_runs_q{i+1}_results.json")
            )
        
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
    
    logger.info("RUNs component test completed")

def test_baseline_methods():
    """Test the baseline uncertainty quantification methods."""
    logger.info("Testing baseline methods...")
    
    # Initialize LLM interface
    llm = LLMInterface()
    
    # Test questions
    test_questions = [
        (
            "What happens when you mix baking soda and vinegar?",
            "Baking soda is sodium bicarbonate (NaHCO3). Vinegar contains acetic acid (CH3COOH).",
            False
        ),
        (
            "Is the Earth flat?",
            "The Earth is approximately 4.5 billion years old and formed through accretion from the solar nebula.",
            True  # This should be detected as a hallucination since the answer will likely contain information not in the context
        )
    ]
    
    # Initialize baseline methods
    baselines = {
        "selfcheckgpt": SelfCheckGPT(llm, {"num_samples": 2, "temperature": 0.7, "similarity_threshold": 0.8}),
        "multidim_uq": MultiDimensionalUQ(llm, {"num_responses": 2, "num_dimensions": 3}),
        "calibration": CalibrationBasedUQ(llm, {"method": "temperature_scaling"}),
        "hudex": HuDEx(llm, {"threshold": 0.65}),
        "metaqa": MetaQA(llm, {"num_mutations": 2, "similarity_threshold": 0.75})
    }
    
    # Test each baseline on each question
    results = {}
    
    for baseline_name, baseline_method in baselines.items():
        logger.info(f"Testing {baseline_name}...")
        
        baseline_results = []
        
        for i, (question, context, expected_hallucination) in enumerate(test_questions):
            logger.info(f"  Question {i+1}: {question}")
            
            try:
                # Process through baseline method
                is_hallucination, hallucination_score, details = baseline_method.detect_hallucination(
                    question, context
                )
                
                # Log results
                logger.info(f"  Detected hallucination: {is_hallucination}, Score: {hallucination_score:.4f}")
                
                # Save results
                baseline_results.append({
                    "question": question,
                    "context": context,
                    "expected_hallucination": expected_hallucination,
                    "detected_hallucination": is_hallucination,
                    "hallucination_score": hallucination_score,
                    "details": {k: v for k, v in details.items() if k not in ["samples", "responses"]}
                })
            
            except Exception as e:
                logger.error(f"  Error processing question {i+1} with {baseline_name}: {e}")
        
        # Save baseline results
        results[baseline_name] = baseline_results
    
    # Save all results
    with open(os.path.join(OUTPUTS_DIR, "test_baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Baseline methods test completed")

def test_evaluation_components():
    """Test the evaluation components."""
    logger.info("Testing evaluation components...")
    
    # Create synthetic test data
    num_samples = 200
    np.random.seed(42)
    
    ground_truth = np.random.binomial(1, 0.3, num_samples).tolist()  # 30% hallucinations
    
    # Model 1 (RUNs) - Better performance
    runs_scores = np.random.beta(2, 5, num_samples)  # Scores biased toward lower values (good)
    runs_scores[np.array(ground_truth) == 1] += np.random.beta(5, 2, sum(ground_truth))  # Boost scores for hallucinations
    runs_scores = np.clip(runs_scores, 0, 1).tolist()
    runs_preds = [score > 0.5 for score in runs_scores]
    
    # Model 2 (Baseline 1) - Medium performance
    baseline1_scores = np.random.beta(1, 1, num_samples)  # Uniform scores (less good)
    baseline1_scores[np.array(ground_truth) == 1] += 0.2  # Smaller boost for hallucinations
    baseline1_scores = np.clip(baseline1_scores, 0, 1).tolist()
    baseline1_preds = [score > 0.5 for score in baseline1_scores]
    
    # Model 3 (Baseline 2) - Worst performance
    baseline2_scores = np.random.beta(1, 1, num_samples)  # Uniform scores
    baseline2_scores = np.clip(baseline2_scores, 0, 1).tolist()
    baseline2_preds = [score > 0.5 for score in baseline2_scores]
    
    # Evaluate models
    evaluator_runs = ModelEvaluator("runs", "runs")
    evaluator_runs.add_batch(runs_preds, ground_truth, runs_scores)
    runs_metrics = evaluator_runs.compute_metrics()
    logger.info(f"RUNs metrics: {runs_metrics}")
    
    evaluator_baseline1 = ModelEvaluator("baseline1", "baseline")
    evaluator_baseline1.add_batch(baseline1_preds, ground_truth, baseline1_scores)
    baseline1_metrics = evaluator_baseline1.compute_metrics()
    logger.info(f"Baseline 1 metrics: {baseline1_metrics}")
    
    evaluator_baseline2 = ModelEvaluator("baseline2", "baseline")
    evaluator_baseline2.add_batch(baseline2_preds, ground_truth, baseline2_scores)
    baseline2_metrics = evaluator_baseline2.compute_metrics()
    logger.info(f"Baseline 2 metrics: {baseline2_metrics}")
    
    # Create and save visualizations
    evaluator_runs.visualize_calibration(os.path.join(OUTPUTS_DIR, "test_runs_calibration.png"))
    evaluator_runs.visualize_confusion_matrix(os.path.join(OUTPUTS_DIR, "test_runs_confusion.png"))
    evaluator_runs.visualize_roc_curve(os.path.join(OUTPUTS_DIR, "test_runs_roc.png"))
    evaluator_runs.visualize_pr_curve(os.path.join(OUTPUTS_DIR, "test_runs_pr.png"))
    
    # Save evaluation results
    evaluator_runs.save_results(os.path.join(OUTPUTS_DIR, "test_runs_results.json"))
    evaluator_baseline1.save_results(os.path.join(OUTPUTS_DIR, "test_baseline1_results.json"))
    evaluator_baseline2.save_results(os.path.join(OUTPUTS_DIR, "test_baseline2_results.json"))
    
    # Compare models
    comparison = ComparisonEvaluator(["runs", "baseline1", "baseline2"], ["runs", "baseline", "baseline"])
    comparison.load_results(OUTPUTS_DIR)
    
    # Create comparison visualizations
    comparison.visualize_metric_comparison("f1", os.path.join(OUTPUTS_DIR, "test_comparison_f1.png"))
    comparison.visualize_metric_comparison("auroc", os.path.join(OUTPUTS_DIR, "test_comparison_auroc.png"))
    comparison._visualize_combined_metrics(
        ["precision", "recall", "f1"],
        os.path.join(OUTPUTS_DIR, "test_comparison_prf1.png")
    )
    
    logger.info("Evaluation components test completed")

def test_data_loading():
    """Test data loading and processing."""
    logger.info("Testing data loading...")
    
    # Set a small number of examples for testing
    original_val = EVAL_CONFIG["num_test_examples"]
    EVAL_CONFIG["num_test_examples"] = 5
    
    try:
        # Load datasets
        datasets = load_all_datasets(
            test_size=0.2,
            val_size=0.1,
            seed=42
        )
        
        # Log dataset statistics
        for dataset_type, splits in datasets.items():
            logger.info(f"Dataset: {dataset_type}")
            for split_name, dataset in splits.items():
                if split_name != "hallucination_indices":
                    logger.info(f"  {split_name}: {len(dataset)} examples")
        
        # Sample from test sets
        for dataset_type, splits in datasets.items():
            test_dataset = splits["test"]
            
            logger.info(f"\nSamples from {dataset_type} test set:")
            for i in range(min(2, len(test_dataset))):
                logger.info(f"\nExample {i}:")
                logger.info(f"  Question: {test_dataset[i]['question']}")
                logger.info(f"  Context: {test_dataset[i]['context'][:100]}...")
                logger.info(f"  Answer: {test_dataset[i]['answer'][:100]}...")
                logger.info(f"  Contains hallucination: {test_dataset[i]['contains_hallucination']}")
                if test_dataset[i]['contains_hallucination']:
                    logger.info(f"  Hallucination type: {test_dataset[i]['hallucination_type']}")
    
    finally:
        # Restore original value
        EVAL_CONFIG["num_test_examples"] = original_val
    
    logger.info("Data loading test completed")

def run_test(args):
    """
    Run the test script.
    
    Args:
        args: Command-line arguments
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Set up visualization style
    setup_visualization_style()
    
    # Run selected tests
    if args.all or args.data:
        test_data_loading()
    
    if args.all or args.runs:
        test_runs_components()
    
    if args.all or args.baselines:
        test_baseline_methods()
    
    if args.all or args.eval:
        test_evaluation_components()
    
    logger.info("All tests completed")

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test Reasoning Uncertainty Networks components")
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    parser.add_argument(
        "--data",
        action="store_true",
        help="Test data loading"
    )
    
    parser.add_argument(
        "--runs",
        action="store_true",
        help="Test RUNs components"
    )
    
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Test baseline methods"
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Test evaluation components"
    )
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all tests
    if not (args.all or args.data or args.runs or args.baselines or args.eval):
        args.all = True
    
    return args

if __name__ == "__main__":
    args = parse_args()
    run_test(args)