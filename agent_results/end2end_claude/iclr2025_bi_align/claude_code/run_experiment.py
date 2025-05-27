"""
Main experiment runner for the CEVA framework evaluation.

This script runs experiments to evaluate the Co-Evolutionary Value Alignment (CEVA) framework
against baseline alignment methods across various scenarios.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from claude_code.config import (
    ROOT_DIR, RESULTS_DIR, FIGURES_DIR, DATA_DIR,
    SIMULATION_CONFIG, VALUE_DIMENSIONS, VALUE_EVOLUTION_PARAMS,
    VALUE_ADAPTATION_RATES, SCENARIOS, MODELS, METRICS, VISUALIZATION_CONFIG
)
from claude_code.value_evolution import (
    ValueSystem, ValueEvolutionModel, HumanAgent, ExternalEvent, ValueEvolutionSimulation
)
from claude_code.alignment_models import (
    BaseAlignmentModel, StaticAlignmentModel, AdaptiveAlignmentModel, CEVAModel
)
from claude_code.evaluation import (
    AlignmentScenario, EvaluationManager
)
from claude_code.visualization import (
    visualize_experiment_results, generate_results_tables
)


def setup_logging(log_file: Path) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('ceva_experiment')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_results(results: Dict, results_dir: Path) -> None:
    """
    Save experiment results to disk.
    
    Args:
        results: Dictionary with experiment results
        results_dir: Directory to save results in
    """
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save raw results as JSON
    with open(results_dir / 'results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_copy = results.copy()
        
        # Process raw data
        for scenario_name, scenario_data in results_copy['scenarios'].items():
            raw_data = scenario_data['raw_data']
            
            # Convert human values
            raw_data['human_values'] = [
                [[float(v) for v in agent_values] for agent_values in step_values]
                for step_values in raw_data['human_values']
            ]
            
            # Convert model values
            for model_name in raw_data['model_values']:
                raw_data['model_values'][model_name] = [
                    [float(v) for v in step_values]
                    for step_values in raw_data['model_values'][model_name]
                ]
        
        # Serialize to JSON
        json.dump(results_copy, f, indent=2)
    
    # Generate and save tables
    tables = generate_results_tables(results)
    for table_name, df in tables.items():
        df.to_csv(results_dir / f'{table_name}.csv', index=False)


def print_experiment_summary(results: Dict) -> None:
    """
    Print a summary of experiment results.
    
    Args:
        results: Dictionary with experiment results
    """
    print("\n=== EXPERIMENT SUMMARY ===\n")
    
    # Print overall metrics
    print("=== Overall Performance ===")
    metrics_to_show = ['avg_adaptation_accuracy', 'avg_stability', 
                     'avg_user_satisfaction', 'avg_agency_preservation']
    
    # Get model names
    models = list(results['aggregate_metrics'].keys())
    
    # Print header
    print(f"{'Metric':<30} " + " ".join([f"{model:<15}" for model in models]))
    
    # Print metrics
    for metric in metrics_to_show:
        metric_name = metric[4:].replace('_', ' ').title()
        metric_values = [results['aggregate_metrics'][model].get(metric, 0) for model in models]
        print(f"{metric_name:<30} " + " ".join([f"{value:.3f}        " for value in metric_values]))
    
    print("\n=== Best Performing Model per Scenario ===")
    for scenario_name, scenario_data in results['scenarios'].items():
        print(f"\nScenario: {scenario_name}")
        print(f"Description: {scenario_data['scenario_description']}")
        
        # Find best model for adaptation accuracy
        best_model = None
        best_score = -float('inf')
        
        for model_name, model_data in scenario_data['models'].items():
            score = model_data['metrics']['adaptation_accuracy']
            if score > best_score:
                best_score = score
                best_model = model_name
                
        print(f"Best model for adaptation accuracy: {best_model} ({best_score:.3f})")


def run_experiment(random_seed: Optional[int] = None) -> Dict:
    """
    Run the CEVA framework evaluation experiment.
    
    Args:
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with experiment results
    """
    # Set up evaluation manager
    evaluation_manager = EvaluationManager(
        value_dimensions=VALUE_DIMENSIONS,
        scenarios=SCENARIOS,
        models=MODELS,
        value_adaptation_rates=VALUE_ADAPTATION_RATES,
        random_seed=random_seed
    )
    
    # Run evaluation
    results = evaluation_manager.run_evaluation()
    
    return results


def generate_results_markdown(
    results: Dict, 
    figure_paths: Dict[str, str], 
    tables: Dict[str, pd.DataFrame],
    output_file: Path
) -> None:
    """
    Generate a markdown file with experiment results.
    
    Args:
        results: Dictionary with experiment results
        figure_paths: Dictionary mapping figure types to file paths
        tables: Dictionary mapping table names to DataFrames
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write("# Co-Evolutionary Value Alignment (CEVA) Framework Experiment Results\n\n")
        
        # Write experiment summary
        f.write("## 1. Experiment Summary\n\n")
        f.write(f"This report presents the results of evaluating the Co-Evolutionary Value Alignment (CEVA) ")
        f.write(f"framework against baseline alignment methods across various scenarios.\n\n")
        
        # Write experimental setup
        f.write("### 1.1 Experimental Setup\n\n")
        
        setup_table = tables['experimental_setup']
        f.write("| Parameter | Value |\n")
        f.write("| --- | --- |\n")
        for _, row in setup_table.iterrows():
            f.write(f"| {row['Parameter']} | {row['Value']} |\n")
        f.write("\n")
        
        # Write overall performance results
        f.write("## 2. Overall Performance\n\n")
        f.write("The following table summarizes the performance of each model across all scenarios:\n\n")
        
        performance_table = tables['overall_performance']
        
        # Create markdown table header from column names
        f.write("| " + " | ".join(performance_table.columns) + " |\n")
        f.write("| " + " | ".join(["---" for _ in performance_table.columns]) + " |\n")
        
        # Write table rows
        for _, row in performance_table.iterrows():
            f.write("| " + " | ".join([str(row[col]) for col in performance_table.columns]) + " |\n")
        f.write("\n")
        
        # Add figure if available
        if 'aggregate_metrics' in figure_paths:
            f.write(f"![Aggregate Metrics Comparison](../{os.path.basename(figure_paths['aggregate_metrics'])})\n\n")
            f.write("*Figure 1: Comparison of key performance metrics across models*\n\n")
        
        if 'metric_radar' in figure_paths:
            f.write(f"![Model Performance Radar](../{os.path.basename(figure_paths['metric_radar'])})\n\n")
            f.write("*Figure 2: Radar chart of model performance metrics*\n\n")
        
        # Write scenario-specific results
        f.write("## 3. Scenario-Specific Results\n\n")
        
        for i, scenario_name in enumerate(results['scenarios'].keys()):
            scenario_data = results['scenarios'][scenario_name]
            f.write(f"### 3.{i+1} {scenario_name}\n\n")
            f.write(f"**Description**: {scenario_data['scenario_description']}\n\n")
            
            # Write scenario-specific table
            if f'scenario_{scenario_name}' in tables:
                scenario_table = tables[f'scenario_{scenario_name}']
                f.write("| " + " | ".join(scenario_table.columns) + " |\n")
                f.write("| " + " | ".join(["---" for _ in scenario_table.columns]) + " |\n")
                
                for _, row in scenario_table.iterrows():
                    f.write("| " + " | ".join([str(row[col]) for col in scenario_table.columns]) + " |\n")
                f.write("\n")
            
            # Add alignment comparison figure
            if f'alignment_{scenario_name}' in figure_paths:
                f.write(f"![Alignment Comparison](../{os.path.basename(figure_paths[f'alignment_{scenario_name}'])})\n\n")
                f.write(f"*Figure 3.{i+1}a: Comparison of alignment scores for {scenario_name}*\n\n")
            
            # Add human value evolution figure
            if f'human_values_{scenario_name}' in figure_paths:
                f.write(f"![Human Value Evolution](../{os.path.basename(figure_paths[f'human_values_{scenario_name}'])})\n\n")
                f.write(f"*Figure 3.{i+1}b: Evolution of human values in {scenario_name}*\n\n")
            
            # Add model value evolution figures
            for j, model_name in enumerate(scenario_data['models'].keys()):
                if f'model_values_{model_name}_{scenario_name}' in figure_paths:
                    f.write(f"![{model_name} Value Evolution](../{os.path.basename(figure_paths[f'model_values_{model_name}_{scenario_name}'])})\n\n")
                    f.write(f"*Figure 3.{i+1}{chr(99+j)}: Evolution of {model_name} values in {scenario_name}*\n\n")
        
        # Write comparative analysis
        f.write("## 4. Comparative Analysis\n\n")
        
        f.write("### 4.1 Adaptation Accuracy\n\n")
        f.write("Adaptation accuracy measures how well the AI model's values match human values over time.\n\n")
        
        if 'metric_adaptation_accuracy' in figure_paths:
            f.write(f"![Adaptation Accuracy](../{os.path.basename(figure_paths['metric_adaptation_accuracy'])})\n\n")
            f.write("*Figure 4.1: Adaptation accuracy across scenarios*\n\n")
        
        f.write("### 4.2 Stability\n\n")
        f.write("Stability measures the model's resistance to spurious adaptation, maintaining consistency when appropriate.\n\n")
        
        if 'metric_stability' in figure_paths:
            f.write(f"![Stability](../{os.path.basename(figure_paths['metric_stability'])})\n\n")
            f.write("*Figure 4.2: Stability across scenarios*\n\n")
        
        f.write("### 4.3 User Satisfaction\n\n")
        f.write("User satisfaction evaluates the perceived quality of responses based on value alignment.\n\n")
        
        if 'metric_user_satisfaction' in figure_paths:
            f.write(f"![User Satisfaction](../{os.path.basename(figure_paths['metric_user_satisfaction'])})\n\n")
            f.write("*Figure 4.3: User satisfaction across scenarios*\n\n")
        
        f.write("### 4.4 Agency Preservation\n\n")
        f.write("Agency preservation measures how well human agency is maintained in the alignment process.\n\n")
        
        if 'metric_agency_preservation' in figure_paths:
            f.write(f"![Agency Preservation](../{os.path.basename(figure_paths['metric_agency_preservation'])})\n\n")
            f.write("*Figure 4.4: Agency preservation across scenarios*\n\n")
        
        # Write conclusion
        f.write("## 5. Conclusion\n\n")
        
        # Find the best overall model
        best_model = None
        best_score = -float('inf')
        
        for model_name, metrics in results['aggregate_metrics'].items():
            # Calculate an overall score as the average of key metrics
            score = np.mean([
                metrics.get('avg_adaptation_accuracy', 0),
                metrics.get('avg_stability', 0),
                metrics.get('avg_user_satisfaction', 0),
                metrics.get('avg_agency_preservation', 0)
            ])
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        f.write(f"The experimental evaluation demonstrates that the {best_model} model achieves the best overall performance ")
        f.write(f"across the tested scenarios. This supports the hypothesis that ")
        
        if best_model.startswith('ceva'):
            f.write(f"the Co-Evolutionary Value Alignment (CEVA) framework provides advantages over traditional ")
            f.write(f"static and simple adaptive alignment approaches. In particular, the {best_model} excels in:\n\n")
        else:
            f.write(f"the {best_model} approach provides effective alignment capabilities. In particular, it excels in:\n\n")
        
        # List top strengths of the best model
        strengths = []
        metrics_to_check = [
            ('avg_adaptation_accuracy', 'adaptation accuracy'),
            ('avg_stability', 'stability'),
            ('avg_user_satisfaction', 'user satisfaction'),
            ('avg_agency_preservation', 'human agency preservation')
        ]
        
        for metric_key, metric_name in metrics_to_check:
            # Check if this metric is among the top 2 for the best model
            metric_value = results['aggregate_metrics'][best_model].get(metric_key, 0)
            
            # Count how many other models have better scores for this metric
            better_count = 0
            for other_model, other_metrics in results['aggregate_metrics'].items():
                if other_model != best_model and other_metrics.get(metric_key, 0) > metric_value:
                    better_count += 1
            
            # If it's among the top 2, add it as a strength
            if better_count <= 1:
                strengths.append(metric_name)
        
        # Write strengths
        for strength in strengths:
            f.write(f"- **{strength.title()}**: The model demonstrates excellent {strength}.\n")
        f.write("\n")
        
        # Add limitations and future work
        f.write("### 5.1 Limitations\n\n")
        f.write("While the experiments demonstrate the effectiveness of the CEVA framework, several limitations should be acknowledged:\n\n")
        f.write("- The simulations use a simplified representation of human values and their evolution\n")
        f.write("- The evaluation relies on synthetic scenarios rather than real-world human-AI interactions\n")
        f.write("- The baseline models are theoretical approximations of existing alignment approaches\n")
        f.write("- The experiments do not account for the full complexity of language model responses\n\n")
        
        f.write("### 5.2 Future Work\n\n")
        f.write("Future research directions for the CEVA framework include:\n\n")
        f.write("- Conducting longitudinal studies with real human participants to validate the value evolution models\n")
        f.write("- Implementing and testing the full CEVA framework with production-grade language models\n")
        f.write("- Exploring more sophisticated bidirectional feedback mechanisms\n")
        f.write("- Investigating culture-specific value trajectories and their implications for alignment\n")
        f.write("- Developing more nuanced metrics for measuring alignment quality and human satisfaction\n")


def main() -> None:
    """
    Main function to run the CEVA framework evaluation experiment.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run CEVA framework evaluation experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    if output_dir is None:
        # Default to project root directory
        output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        output_dir = Path(output_dir)
        
    # Set up paths
    results_dir = output_dir / 'results'
    figures_dir = results_dir / 'figures'
    log_file = results_dir / 'log.txt'
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_file)
    logger.info("Starting CEVA framework evaluation experiment")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Figures directory: {figures_dir}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Record experiment start time
        start_time = time.time()
        
        # Run experiment
        logger.info("Running experiment...")
        results = run_experiment(random_seed=args.seed)
        
        # Record experiment end time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Experiment completed in {execution_time:.2f} seconds")
        
        # Visualize results
        logger.info("Generating visualizations...")
        figure_paths = visualize_experiment_results(results, figures_dir, VISUALIZATION_CONFIG)
        
        # Generate tables
        logger.info("Generating results tables...")
        tables = generate_results_tables(results)
        
        # Save results
        logger.info("Saving results...")
        save_results(results, results_dir)
        
        # Generate markdown report
        logger.info("Generating results markdown...")
        generate_results_markdown(
            results,
            figure_paths,
            tables,
            results_dir / 'results.md'
        )
        
        # Print summary
        logger.info("Experiment summary:")
        print_experiment_summary(results)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        raise
        

if __name__ == "__main__":
    main()