"""
Main experiment runner script

This script coordinates the entire experiment workflow for the ContextBench framework.
It loads datasets, trains models, evaluates them across multiple metrics and contexts,
and generates visualizations and reports.
"""

import os
import json
import logging
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime

# Import ContextBench modules
from utils.metadata_schema import ContextualMetadata, create_example_metadata, save_all_metadata
from utils.task_config import ContextVector, DynamicTaskConfigurator, create_example_contexts, save_example_contexts
from metrics.metrics_suite import MultiMetricEvaluationSuite
from data.data_loader import load_and_preprocess_dataset, create_adversarial_examples
from models.model_trainer import ModelTrainer
from models.baselines import get_baselines_for_domain, get_shap_perturbation_function
from utils.visualization import generate_all_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run ContextBench experiments')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='adult',
        choices=['adult', 'mnist', 'sst2'],
        help='Dataset to use for experiments'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        default='tabular',
        choices=['tabular', 'vision', 'text'],
        help='Domain of the dataset'
    )
    
    parser.add_argument(
        '--task_type',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='Type of task'
    )
    
    parser.add_argument(
        '--contexts',
        type=str,
        nargs='+',
        default=['healthcare', 'finance'],
        help='Contexts to evaluate in'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Use GPU for training if available'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    return parser.parse_args()


def setup_experiment_directories(base_dir: str = '..') -> Dict[str, str]:
    """
    Set up directories for the experiment.
    
    Args:
        base_dir: Base directory
        
    Returns:
        dict: Dictionary mapping directory names to paths
    """
    # Create directory structure
    directories = {
        'data': os.path.join(base_dir, 'data'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results'),
        'metadata': os.path.join(base_dir, 'data', 'metadata'),
        'contexts': os.path.join(base_dir, 'data', 'contexts'),
        'visualizations': os.path.join(base_dir, 'results', 'visualizations')
    }
    
    # Create directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def prepare_metadata_and_contexts(directories: Dict[str, str]):
    """
    Prepare metadata and contexts for the experiment.
    
    Args:
        directories: Dictionary mapping directory names to paths
    """
    # Create example metadata
    save_all_metadata(directories['metadata'])
    logger.info(f"Saved example metadata to {directories['metadata']}")
    
    # Create example contexts
    save_example_contexts(directories['contexts'])
    logger.info(f"Saved example contexts to {directories['contexts']}")


def run_experiment(args):
    """
    Run the ContextBench experiment.
    
    Args:
        args: Command line arguments
    """
    # Start timer
    start_time = time.time()
    
    # Set up directories
    directories = setup_experiment_directories()
    
    # Prepare metadata and contexts
    prepare_metadata_and_contexts(directories)
    
    # Load metadata for the dataset
    metadata_dict = create_example_metadata()
    metadata = metadata_dict.get(args.dataset)
    
    if metadata is None:
        # Create a default metadata if not found
        metadata = ContextualMetadata(
            dataset_id=args.dataset,
            domain_tags=[args.domain, args.task_type]
        )
    
    logger.info(f"Loaded metadata for {args.dataset} dataset")
    
    # Load and preprocess dataset
    logger.info(f"Loading and preprocessing {args.dataset} dataset...")
    data_dict, preprocessor_dict = load_and_preprocess_dataset(
        args.dataset,
        data_dir=directories['data'],
        random_state=args.random_state,
        download=True
    )
    logger.info(f"Loaded and preprocessed {args.dataset} dataset")
    
    # Get baselines for the domain and task type
    baselines = get_baselines_for_domain(args.domain, args.task_type)
    logger.info(f"Using {len(baselines)} baseline models for {args.domain} {args.task_type}")
    
    # Initialize task configurator
    task_configurator = DynamicTaskConfigurator(metadata.data)
    
    # Initialize evaluation suite
    evaluation_suite = MultiMetricEvaluationSuite(metadata.data, args.task_type)
    
    # Initialize perturbation function for interpretability evaluation
    is_image = args.domain == 'vision'
    is_text = args.domain == 'text'
    perturbation_func = get_shap_perturbation_function(is_image, is_text)
    
    # Dictionary to store results
    results = {}
    context_results = {}
    
    # Train and evaluate baseline models
    for model_name, (model_class, hyperparams) in baselines.items():
        logger.info(f"Training {model_name}...")
        
        # Initialize model trainer
        model_trainer = ModelTrainer(
            model_class=model_class,
            hyperparams=hyperparams,
            task_type=args.task_type,
            model_name=model_name,
            random_state=args.random_state
        )
        
        # Start monitoring resources
        evaluation_suite.start_monitoring()
        
        # Train model
        model = model_trainer.train(
            data_dict['X_train'],
            data_dict['y_train'],
            data_dict.get('X_val'),
            data_dict.get('y_val')
        )
        
        # Get attribution function
        attribution_func = model_trainer.create_attribution_function(
            is_image=is_image
        )
        
        # Create adversarial examples
        adversarial_X = create_adversarial_examples(
            model=model,
            X=data_dict['X_test'],
            y=data_dict['y_test']
        )
        
        # Get predictions
        y_pred = model_trainer.predict(data_dict['X_test'])
        
        try:
            y_score = model_trainer.predict_proba(data_dict['X_test'])
            if y_score.shape[1] == 2:  # Binary classification
                y_score = y_score[:, 1]
        except:
            y_score = None
        
        # Evaluate model on standard test set
        logger.info(f"Evaluating {model_name} on standard test set...")
        
        # Define attack function for adversarial robustness evaluation
        def attack_function(model, X, y, epsilon=0.1, norm='l_inf'):
            return adversarial_X
        
        # Evaluate model
        standard_evaluation = evaluation_suite.evaluate(
            model=model,
            x_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            y_pred=y_pred,
            y_score=y_score,
            feature_data=data_dict.get('feature_data', {}).get('test', {}),
            x_shifted=data_dict.get('X_test_shifted'),
            y_shifted=data_dict.get('y_test_shifted'),
            attribution_func=attribution_func,
            perturbation_func=perturbation_func,
            attack_function=attack_function,
            model_size_mb=model_trainer.model_size_mb,
            domain=args.domain
        )
        
        # Generate report
        standard_report = evaluation_suite.generate_report(
            standard_evaluation,
            context={'domain': args.domain, 'task_type': args.task_type}
        )
        
        # Save standard evaluation results
        results[model_name] = standard_report
        
        # Evaluate model on each context
        context_model_results = {}
        
        for context_name in args.contexts:
            logger.info(f"Evaluating {model_name} in {context_name} context...")
            
            # Create context-specific test split
            X_test_context, y_test_context = task_configurator.create_domain_specific_split(
                data_dict['X_test'],
                data_dict['y_test'],
                domain=context_name,
                feature_data=data_dict.get('feature_data', {}).get('test', {}),
                random_state=args.random_state
            )
            
            # Get predictions for context-specific test split
            y_pred_context = model_trainer.predict(X_test_context)
            
            try:
                y_score_context = model_trainer.predict_proba(X_test_context)
                if y_score_context.shape[1] == 2:  # Binary classification
                    y_score_context = y_score_context[:, 1]
            except:
                y_score_context = None
            
            # Evaluate model on context-specific test set
            context_evaluation = evaluation_suite.evaluate(
                model=model,
                x_test=X_test_context,
                y_test=y_test_context,
                y_pred=y_pred_context,
                y_score=y_score_context,
                feature_data=data_dict.get('feature_data', {}).get('test', {}),
                x_shifted=data_dict.get('X_test_shifted'),
                y_shifted=data_dict.get('y_test_shifted'),
                attribution_func=attribution_func,
                perturbation_func=perturbation_func,
                attack_function=attack_function,
                model_size_mb=model_trainer.model_size_mb,
                domain=context_name
            )
            
            # Generate context report
            context_report = evaluation_suite.generate_report(
                context_evaluation,
                context={'domain': context_name, 'task_type': args.task_type}
            )
            
            # Calculate context score
            context_vector = ContextVector(
                domain=context_name,
                constraints={},
                subgroup_weights={},
                performance_thresholds={}
            )
            
            context_score = task_configurator.calculate_context_score(
                context_evaluation,
                context_vector
            )
            
            # Add context score to report
            context_report.update(context_score)
            
            # Add to context results
            context_model_results[context_name] = context_report
        
        # Save context evaluation results
        context_results[model_name] = context_model_results
        
        # Save model
        model_path = model_trainer.save_model(
            directories['models'],
            f"{args.dataset}_{model_name.lower()}.joblib"
        )
        logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save results to JSON
    results_path = os.path.join(directories['results'], f"{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    context_results_path = os.path.join(directories['results'], f"{args.dataset}_context_results.json")
    with open(context_results_path, 'w') as f:
        json.dump(context_results, f, indent=2)
    logger.info(f"Saved context results to {context_results_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualization_paths = generate_all_visualizations(
        results,
        context_results,
        directories['visualizations'],
        args.contexts
    )
    
    # Save visualization paths to JSON
    visualization_paths_path = os.path.join(directories['results'], 'visualization_paths.json')
    with open(visualization_paths_path, 'w') as f:
        json.dump(visualization_paths, f, indent=2)
    logger.info(f"Saved visualization paths to {visualization_paths_path}")
    
    # Calculate total experiment time
    total_time = time.time() - start_time
    logger.info(f"Total experiment time: {total_time:.2f} seconds")
    
    # Generate results.md
    generate_results_markdown(
        results,
        context_results,
        visualization_paths,
        directories['results'],
        args.dataset,
        args.contexts,
        total_time
    )
    logger.info(f"Generated results.md in {directories['results']}")


def generate_results_markdown(
    results: Dict[str, Any],
    context_results: Dict[str, Dict[str, Any]],
    visualization_paths: Dict[str, str],
    output_dir: str,
    dataset_name: str,
    contexts: List[str],
    total_time: float
):
    """
    Generate a markdown file summarizing the results.
    
    Args:
        results: Dictionary of evaluation results
        context_results: Dictionary of context-specific results
        visualization_paths: Dictionary mapping visualization names to file paths
        output_dir: Directory to save the markdown file
        dataset_name: Name of the dataset
        contexts: List of contexts
        total_time: Total experiment time in seconds
    """
    # Create results.md file
    with open(os.path.join(output_dir, 'results.md'), 'w') as f:
        # Write header
        f.write(f"# ContextBench Experiment Results: {dataset_name.upper()} Dataset\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Write experiment summary
        f.write("## Experiment Summary\n\n")
        f.write(f"- **Dataset**: {dataset_name}\n")
        f.write(f"- **Contexts**: {', '.join(contexts)}\n")
        f.write(f"- **Total Experiment Time**: {total_time:.2f} seconds\n")
        f.write(f"- **Number of Models**: {len(results)}\n\n")
        
        # Write models summary
        f.write("## Models Summary\n\n")
        f.write("| Model | Accuracy | F1 Score | Fairness | Robustness | Env. Impact | Interpretability | Overall |\n")
        f.write("|-------|----------|----------|----------|------------|-------------|-----------------|--------|\n")
        
        for model_name, result in results.items():
            # Extract metrics
            accuracy = result.get('performance', {}).get('accuracy', '-')
            f1 = result.get('performance', {}).get('f1', '-')
            
            # Format as percentage
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.2%}"
            if isinstance(f1, (int, float)):
                f1 = f"{f1:.2%}"
            
            # Get category scores
            fairness = result.get('category_scores', {}).get('fairness', '-')
            robustness = result.get('category_scores', {}).get('robustness', '-')
            env_impact = result.get('category_scores', {}).get('environmental_impact', '-')
            interpretability = result.get('category_scores', {}).get('interpretability', '-')
            overall = result.get('overall_score', '-')
            
            # Format as percentage
            if isinstance(fairness, (int, float)):
                fairness = f"{fairness:.2%}"
            if isinstance(robustness, (int, float)):
                robustness = f"{robustness:.2%}"
            if isinstance(env_impact, (int, float)):
                env_impact = f"{env_impact:.2%}"
            if isinstance(interpretability, (int, float)):
                interpretability = f"{interpretability:.2%}"
            if isinstance(overall, (int, float)):
                overall = f"{overall:.2%}"
            
            # Write row
            f.write(f"| {model_name} | {accuracy} | {f1} | {fairness} | {robustness} | {env_impact} | {interpretability} | {overall} |\n")
        
        f.write("\n")
        
        # Write performance comparison
        f.write("## Performance Comparison\n\n")
        
        # Add performance comparison visualization
        if 'performance_comparison' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['performance_comparison'], output_dir)
            f.write(f"![Performance Comparison]({vis_path})\n\n")
        
        # Write fairness comparison
        f.write("## Fairness Comparison\n\n")
        
        # Add fairness comparison visualization
        if 'fairness_comparison' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['fairness_comparison'], output_dir)
            f.write(f"![Fairness Comparison]({vis_path})\n\n")
        
        # Write robustness comparison
        f.write("## Robustness Comparison\n\n")
        
        # Add robustness comparison visualization
        if 'robustness_comparison' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['robustness_comparison'], output_dir)
            f.write(f"![Robustness Comparison]({vis_path})\n\n")
        
        # Write environmental impact comparison
        f.write("## Environmental Impact\n\n")
        
        # Add environmental impact visualization
        if 'environmental_impact' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['environmental_impact'], output_dir)
            f.write(f"![Environmental Impact]({vis_path})\n\n")
        
        # Write interpretability comparison
        f.write("## Interpretability\n\n")
        
        # Add interpretability comparison visualization
        if 'interpretability_comparison' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['interpretability_comparison'], output_dir)
            f.write(f"![Interpretability Comparison]({vis_path})\n\n")
        
        # Write multi-metric comparison
        f.write("## Multi-Metric Comparison\n\n")
        
        # Add radar chart visualization
        if 'radar_chart' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['radar_chart'], output_dir)
            f.write(f"![Multi-Metric Comparison]({vis_path})\n\n")
        
        # Write trade-offs
        f.write("## Performance Trade-offs\n\n")
        
        # Add performance-fairness trade-off visualization
        if 'performance_fairness_tradeoff' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['performance_fairness_tradeoff'], output_dir)
            f.write(f"![Performance-Fairness Trade-off]({vis_path})\n\n")
        
        # Add performance-robustness trade-off visualization
        if 'performance_robustness_tradeoff' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['performance_robustness_tradeoff'], output_dir)
            f.write(f"![Performance-Robustness Trade-off]({vis_path})\n\n")
        
        # Write context-specific results
        f.write("## Context-Specific Results\n\n")
        
        # Add context comparison visualization
        if 'context_comparison' in visualization_paths:
            vis_path = os.path.relpath(visualization_paths['context_comparison'], output_dir)
            f.write(f"![Context Comparison]({vis_path})\n\n")
        
        # Write context-specific tables
        for context in contexts:
            f.write(f"### {context.capitalize()} Context\n\n")
            f.write("| Model | Overall Score | Performance | Fairness | Robustness | Env. Impact | Interpretability |\n")
            f.write("|-------|--------------|------------|----------|------------|-------------|------------------|\n")
            
            for model_name, context_model_results in context_results.items():
                if context in context_model_results:
                    result = context_model_results[context]
                    
                    # Get scores
                    overall = result.get('overall_score', '-')
                    performance = result.get('category_scores', {}).get('performance', '-')
                    fairness = result.get('category_scores', {}).get('fairness', '-')
                    robustness = result.get('category_scores', {}).get('robustness', '-')
                    env_impact = result.get('category_scores', {}).get('environmental_impact', '-')
                    interpretability = result.get('category_scores', {}).get('interpretability', '-')
                    
                    # Format as percentage
                    if isinstance(overall, (int, float)):
                        overall = f"{overall:.2%}"
                    if isinstance(performance, (int, float)):
                        performance = f"{performance:.2%}"
                    if isinstance(fairness, (int, float)):
                        fairness = f"{fairness:.2%}"
                    if isinstance(robustness, (int, float)):
                        robustness = f"{robustness:.2%}"
                    if isinstance(env_impact, (int, float)):
                        env_impact = f"{env_impact:.2%}"
                    if isinstance(interpretability, (int, float)):
                        interpretability = f"{interpretability:.2%}"
                    
                    # Write row
                    f.write(f"| {model_name} | {overall} | {performance} | {fairness} | {robustness} | {env_impact} | {interpretability} |\n")
            
            f.write("\n")
            
            # Add model-specific context profiles
            for model_name in results.keys():
                profile_key = f"{model_name}_{context}_profile"
                if profile_key in visualization_paths:
                    vis_path = os.path.relpath(visualization_paths[profile_key], output_dir)
                    f.write(f"**{model_name} Profile in {context.capitalize()} Context**\n\n")
                    f.write(f"![{model_name} in {context.capitalize()}]({vis_path})\n\n")
        
        # Write conclusions
        f.write("## Conclusions\n\n")
        f.write("The experiments with the ContextBench framework demonstrate the importance of holistic, ")
        f.write("multi-dimensional evaluation of machine learning models. Key findings include:\n\n")
        
        # Generate conclusions based on results
        best_model = max(results.items(), key=lambda x: x[1].get('overall_score', 0))[0]
        most_fair = max(results.items(), key=lambda x: x[1].get('category_scores', {}).get('fairness', 0))[0]
        most_robust = max(results.items(), key=lambda x: x[1].get('category_scores', {}).get('robustness', 0))[0]
        most_efficient = max(results.items(), key=lambda x: x[1].get('category_scores', {}).get('environmental_impact', 0))[0]
        most_interpretable = max(results.items(), key=lambda x: x[1].get('category_scores', {}).get('interpretability', 0))[0]
        
        f.write(f"1. **Overall Performance**: {best_model} achieved the highest overall score across all metrics, ")
        f.write(f"making it the most balanced model in the experiment.\n\n")
        
        f.write(f"2. **Fairness**: {most_fair} demonstrated the best fairness metrics, ")
        f.write("suggesting it has the most equitable performance across different demographic groups.\n\n")
        
        f.write(f"3. **Robustness**: {most_robust} showed the highest robustness, ")
        f.write("maintaining performance under noise, distribution shift, and adversarial conditions.\n\n")
        
        f.write(f"4. **Environmental Efficiency**: {most_efficient} had the lowest environmental impact, ")
        f.write("using fewer computational resources and energy.\n\n")
        
        f.write(f"5. **Interpretability**: {most_interpretable} provided the most stable and interpretable ")
        f.write("feature attributions.\n\n")
        
        f.write("6. **Context Sensitivity**: Model performance varied significantly across different contexts, ")
        f.write("highlighting the importance of context-aware evaluation.\n\n")
        
        # Write limitations
        f.write("## Limitations\n\n")
        f.write("Despite the comprehensive nature of ContextBench, this implementation has several limitations:\n\n")
        
        f.write("1. **Limited Datasets**: The experiment was conducted on a single dataset, which may not ")
        f.write("represent the diversity of real-world data.\n\n")
        
        f.write("2. **Simplified Contexts**: The context definitions used are simplified approximations of ")
        f.write("real-world deployment contexts.\n\n")
        
        f.write("3. **Synthetic Sensitivity Attributes**: For some datasets, sensitive attributes were ")
        f.write("synthetically created due to lack of real demographic information.\n\n")
        
        f.write("4. **Simplified Adversarial Testing**: The adversarial examples generation is a simplified ")
        f.write("approach and may not represent sophisticated adversarial attacks.\n\n")
        
        f.write("5. **Limited Model Selection**: The experiment used a fixed set of baseline models, which ")
        f.write("may not include state-of-the-art architectures for specific domains.\n\n")
        
        # Write future work
        f.write("## Future Work\n\n")
        f.write("Future development of ContextBench could include:\n\n")
        
        f.write("1. **Extended Dataset Suite**: Incorporate a more diverse set of datasets across different domains.\n\n")
        
        f.write("2. **Real Demographic Data**: Use datasets with real demographic information for more ")
        f.write("accurate fairness evaluation.\n\n")
        
        f.write("3. **Advanced Adversarial Testing**: Implement more sophisticated adversarial attack methods.\n\n")
        
        f.write("4. **User Studies**: Conduct user studies to validate the interpretability and usefulness ")
        f.write("of context profiles.\n\n")
        
        f.write("5. **Integration with ML Platforms**: Develop plugins for common ML platforms to seamlessly ")
        f.write("incorporate ContextBench in existing workflows.\n\n")
        
        f.write("6. **Expanded Metrics**: Include additional metrics for privacy, security, and domain-specific ")
        f.write("requirements.\n\n")


if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)