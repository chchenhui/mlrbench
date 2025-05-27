#!/usr/bin/env python
"""
Main experiment runner for the SSCSteer framework.

This script sets up and runs the experiments to evaluate the effectiveness
of the Syntactic and Semantic Conformance Steering (SSCSteer) framework.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the experiment modules
from src.ssm import SyntacticSteeringModule
from src.sesm import SemanticSteeringModule
from src.sscsteer import SSCSteer
from src.baselines import VanillaLLMGenerator, PostHocSyntaxValidator, FeedbackBasedRefinement
from src.datasets import load_datasets, save_datasets
from src.evaluation import evaluate_on_dataset
from src.visualization import visualize_results

# Import LLM interface
# This will be replaced with actual LLM API calls
from src.llm_interface import get_llm_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/experiment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SSCSteer-Experiment")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SSCSteer experiments")
    
    parser.add_argument(
        "--llm", 
        type=str, 
        default="openai",
        choices=["openai", "claude", "codellama", "qwen"],
        help="LLM provider to use for code generation"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini",
        help="Specific model to use for code generation"
    )
    
    parser.add_argument(
        "--num_problems", 
        type=int, 
        default=10,
        help="Number of problems to evaluate per dataset"
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--run_baselines", 
        action="store_true",
        help="Run baseline methods"
    )
    
    parser.add_argument(
        "--run_ablation", 
        action="store_true",
        help="Run ablation study"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def setup_approaches(llm: str, model: str):
    """
    Set up the different code generation approaches to evaluate.
    
    Args:
        llm: LLM provider to use
        model: Specific model to use
        
    Returns:
        Dictionary mapping approach names to approach functions
    """
    # Get LLM generator function
    llm_generator = get_llm_generator(llm, model)
    
    # Create approach functions
    approaches = {}
    
    # 1. Vanilla LLM
    vanilla_llm = VanillaLLMGenerator(max_tokens=512)
    approaches["Vanilla LLM"] = lambda prompt, _: vanilla_llm.generate_code(prompt, llm_generator)
    
    # 2. Post-hoc Syntax Validation
    post_hoc = PostHocSyntaxValidator(max_tokens=512, max_attempts=3)
    approaches["Post-hoc Syntax"] = lambda prompt, _: post_hoc.generate_code(prompt, llm_generator)
    
    # 3. Feedback-based Refinement
    feedback = FeedbackBasedRefinement(max_tokens=512, max_iterations=2)
    approaches["Feedback Refinement"] = lambda prompt, _: feedback.generate_code(prompt, llm_generator)
    
    # 4. SSM-only (Syntactic Steering only)
    ssm_only = SSCSteer(
        use_syntactic_steering=True,
        use_semantic_steering=False,
        beam_width=3,
        max_tokens=512
    )
    approaches["SSM-only"] = lambda prompt, _: ssm_only.generate_code(prompt, llm_generator)
    
    # 5. SeSM-only (Semantic Steering only)
    sesm_only = SSCSteer(
        use_syntactic_steering=False,
        use_semantic_steering=True,
        semantic_check_frequency=3,
        beam_width=3,
        max_tokens=512
    )
    approaches["SeSM-only"] = lambda prompt, _: sesm_only.generate_code(prompt, llm_generator)
    
    # 6. Full SSCSteer
    full_sscsteer = SSCSteer(
        use_syntactic_steering=True,
        use_semantic_steering=True,
        semantic_check_frequency=3,
        beam_width=5,
        max_tokens=512
    )
    approaches["Full SSCSteer"] = lambda prompt, _: full_sscsteer.generate_code(prompt, llm_generator)
    
    return approaches


def run_evaluation(datasets, approaches, num_problems, llm_generator):
    """
    Run evaluation of approaches on datasets.
    
    Args:
        datasets: Dictionary of datasets
        approaches: Dictionary of approaches
        num_problems: Number of problems to evaluate per dataset
        llm_generator: LLM generator function
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Evaluate on HumanEval
    if 'humaneval' in datasets:
        logger.info("Evaluating on HumanEval dataset")
        humaneval_results = evaluate_on_dataset(
            datasets['humaneval'],
            approaches,
            llm_generator,
            num_samples=min(num_problems, len(datasets['humaneval']))
        )
        results['humaneval'] = humaneval_results
    
    # Evaluate on Semantic Tasks
    if 'semantic' in datasets:
        logger.info("Evaluating on Semantic Tasks dataset")
        semantic_results = evaluate_on_dataset(
            datasets['semantic'],
            approaches,
            llm_generator,
            num_samples=min(num_problems, len(datasets['semantic']))
        )
        results['semantic'] = semantic_results
    
    # Combine results from datasets
    combined_results = {}
    combined_detailed = {}
    
    for dataset_name, dataset_results in results.items():
        # Extract comparison dataframe
        if 'comparison' in dataset_results:
            df = dataset_results['comparison']
            df['dataset'] = dataset_name
            if 'combined_comparison' not in combined_results:
                combined_results['combined_comparison'] = []
            combined_results['combined_comparison'].append(df)
            
        # Extract detailed results
        if 'detailed_results' in dataset_results:
            for approach, approach_results in dataset_results['detailed_results'].items():
                if approach not in combined_detailed:
                    combined_detailed[approach] = []
                combined_detailed[approach].extend(approach_results)
    
    # Combine comparison dataframes
    if 'combined_comparison' in combined_results:
        combined_results['comparison'] = pd.concat(combined_results['combined_comparison']).groupby('approach').mean()
        combined_results['comparison'] = combined_results['comparison'].reset_index()
    
    # Add detailed results
    combined_results['detailed_results'] = combined_detailed
    
    # Add dataset results for dataset comparison
    combined_results['dataset_results'] = {
        dataset_name: dataset_results['detailed_results'] 
        for dataset_name, dataset_results in results.items()
        if 'detailed_results' in dataset_results
    }
    
    return combined_results


def run_ablation_study(llm, model, num_problems):
    """
    Run ablation study for SSCSteer components.
    
    Args:
        llm: LLM provider to use
        model: Specific model to use
        num_problems: Number of problems to evaluate per dataset
        
    Returns:
        Dictionary with ablation study results
    """
    logger.info("Running ablation study")
    
    # Get LLM generator function
    llm_generator = get_llm_generator(llm, model)
    
    # Load datasets
    datasets = load_datasets()
    
    # Use only semantic tasks for ablation to focus on semantic improvements
    if 'semantic' in datasets:
        dataset = datasets['semantic']
    else:
        dataset = next(iter(datasets.values()))
    
    # Sample problems
    if num_problems < len(dataset):
        sampled_problems = np.random.choice(dataset, size=num_problems, replace=False)
    else:
        sampled_problems = dataset
    
    # Define ablation configurations
    configurations = {
        "BaseSSCSteer": {
            "use_syntactic_steering": True,
            "use_semantic_steering": True,
            "semantic_check_frequency": 3,
            "beam_width": 3
        },
        "NoBugPatternChecks": {
            "use_syntactic_steering": True,
            "use_semantic_steering": True,
            "semantic_check_frequency": 5,  # Less frequent checks
            "beam_width": 3,
            "use_smt": False  # Disable formal verification
        },
        "NoNullChecks": {
            "use_syntactic_steering": True,
            "use_semantic_steering": True,
            "semantic_check_frequency": 3,
            "beam_width": 3,
            "disable_checks": ["null_dereference"]
        },
        "NoBeamSearch": {
            "use_syntactic_steering": True,
            "use_semantic_steering": True,
            "semantic_check_frequency": 3,
            "beam_width": 1  # No beam search, just greedy
        },
        "SmallBeam": {
            "use_syntactic_steering": True,
            "use_semantic_steering": True,
            "semantic_check_frequency": 3,
            "beam_width": 2  # Smaller beam
        },
        "LargeBeam": {
            "use_syntactic_steering": True,
            "use_semantic_steering": True,
            "semantic_check_frequency": 3,
            "beam_width": 10  # Larger beam
        }
    }
    
    # Run evaluations
    ablation_results = {}
    
    for config_name, config in configurations.items():
        logger.info(f"Evaluating configuration: {config_name}")
        
        # Create SSCSteer instance with this configuration
        sscsteer = SSCSteer(**config)
        
        # Evaluate on sampled problems
        config_results = []
        
        for problem in sampled_problems:
            prompt = problem["prompt"]
            test_cases = problem["test_cases"]
            
            try:
                # Generate code
                generation_result = sscsteer.generate_code(prompt, llm_generator)
                code = generation_result["code"]
                
                # Evaluate code
                from src.evaluation import evaluate_code_against_tests, calculate_code_quality_metrics
                
                test_result = evaluate_code_against_tests(code, test_cases)
                quality_metrics = calculate_code_quality_metrics(code)
                
                # Combine results
                result = {
                    "pass_rate": test_result["pass_rate"],
                    "is_valid": quality_metrics["is_valid"],
                    "pylint_score": quality_metrics["pylint_score"],
                    "generation_time": generation_result["metrics"]["generation_time"]
                }
                
                config_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {config_name}: {e}")
        
        # Calculate average metrics
        if config_results:
            ablation_results[config_name] = {
                "pass_rate": np.mean([r["pass_rate"] for r in config_results]),
                "syntactic_validity": np.mean([r["is_valid"] for r in config_results]),
                "pylint_score": np.mean([r["pylint_score"] for r in config_results]),
                "generation_time": np.mean([r["generation_time"] for r in config_results])
            }
        else:
            ablation_results[config_name] = {
                "pass_rate": 0.0,
                "syntactic_validity": 0.0,
                "pylint_score": 0.0,
                "generation_time": 0.0
            }
    
    return ablation_results


def main():
    """Main function to run the experiments."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save start time
    start_time = time.time()
    
    # Load datasets
    logger.info("Loading datasets")
    datasets = load_datasets("data")
    
    # Create LLM generator
    logger.info(f"Using LLM: {args.llm} with model: {args.model}")
    llm_generator = get_llm_generator(args.llm, args.model)
    
    # Run evaluations
    results = {}
    
    if args.run_baselines:
        # Set up approaches
        logger.info("Setting up approaches")
        approaches = setup_approaches(args.llm, args.model)
        
        # Run evaluation
        logger.info("Running evaluation")
        results = run_evaluation(datasets, approaches, args.num_problems, llm_generator)
    else:
        # Just evaluate SSCSteer without baselines
        logger.info("Evaluating only SSCSteer (no baselines)")
        
        # Create SSCSteer instance
        sscsteer = SSCSteer(
            use_syntactic_steering=True,
            use_semantic_steering=True,
            semantic_check_frequency=3,
            beam_width=5,
            max_tokens=512
        )
        
        # Set up limited approaches
        approaches = {
            "Full SSCSteer": lambda prompt, _: sscsteer.generate_code(prompt, llm_generator)
        }
        
        # Run evaluation
        results = run_evaluation(datasets, approaches, args.num_problems, llm_generator)
    
    # Run ablation study if requested
    if args.run_ablation:
        logger.info("Running ablation study")
        ablation_results = run_ablation_study(args.llm, args.model, min(5, args.num_problems))
        results['ablation_results'] = ablation_results
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Experiments completed in {elapsed_time:.2f} seconds")
    
    # Save results
    results_path = os.path.join(args.save_dir, "experiment_results.json")
    with open(results_path, 'w') as f:
        # Convert DataFrame to dict for JSON serialization
        if 'comparison' in results and isinstance(results['comparison'], pd.DataFrame):
            results['comparison'] = results['comparison'].to_dict(orient='records')
            
        json.dump(results, f, indent=2, default=lambda x: str(x))
    
    logger.info(f"Results saved to {results_path}")
    
    # Visualize results
    logger.info("Visualizing results")
    visualize_results(results, args.save_dir)
    
    # Log completion
    logger.info("Experiment runner completed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())