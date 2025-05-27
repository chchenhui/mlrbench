"""
Visualize VERIL experiment results.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

from config import RESULTS_DIR
from evaluation import EvaluationResult
from utils import load_json, plot_bar_comparison, plot_learning_curve

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_results():
    """Load evaluation results."""
    results_file = RESULTS_DIR / "all_results.json"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return {}
    
    # Load raw results
    raw_results = load_json(results_file)
    
    # Convert to EvaluationResult objects
    results = {}
    for model_name, result_dict in raw_results.items():
        result = EvaluationResult(model_name=model_name)
        result.total_samples = result_dict["total_samples"]
        result.correct_samples = result_dict["correct_samples"]
        result.verification_results = result_dict["verification_results"]
        result.error_counts = defaultdict(int, result_dict["error_counts"])
        result.execution_times = result_dict["execution_times"]
        result.learning_metrics = result_dict.get("learning_metrics", {})
        
        results[model_name] = result
    
    return results

def visualize_model_comparison(results):
    """Visualize model comparison."""
    model_names = list(results.keys())
    
    # Calculate metrics
    metrics = {
        "pass_rate": [result.correct_samples / result.total_samples for result in results.values()],
        "error_rate": [
            sum(result.error_counts.values()) / result.total_samples 
            for result in results.values()
        ],
        "veri_pass_rate": [
            sum(result.verification_results) / result.total_samples 
            for result in results.values()
        ],
    }
    
    # Plot bar comparison
    plot_bar_comparison(
        model_names=model_names,
        metrics=metrics,
        title="Model Performance Comparison",
        xlabel="Models",
        ylabel="Performance",
        output_path=RESULTS_DIR / "model_comparison.png",
    )
    
    print(f"Generated model comparison plot: {RESULTS_DIR}/model_comparison.png")

def visualize_error_types(results):
    """Visualize error type distribution."""
    model_names = list(results.keys())
    error_types = ["syntax", "type", "logic", "semantic", "security"]
    
    # Calculate error rates by type
    error_metrics = {}
    for error_type in error_types:
        error_metrics[f"{error_type}_error"] = [
            result.error_counts.get(error_type, 0) / result.total_samples 
            for result in results.values()
        ]
    
    # Plot bar comparison
    plot_bar_comparison(
        model_names=model_names,
        metrics=error_metrics,
        title="Error Types Distribution",
        xlabel="Models",
        ylabel="Error Rate",
        output_path=RESULTS_DIR / "error_types.png",
    )
    
    print(f"Generated error types plot: {RESULTS_DIR}/error_types.png")

def visualize_learning_curves(results):
    """Visualize learning curves for VERIL models."""
    for model_name, result in results.items():
        if "veril" in model_name.lower() and result.learning_metrics:
            metrics = {k: v for k, v in result.learning_metrics.items() if k != "iterations"}
            
            plot_learning_curve(
                iterations=result.learning_metrics["iterations"],
                metrics=metrics,
                title=f"Learning Progress - {model_name}",
                xlabel="Iterations",
                ylabel="Performance",
                output_path=RESULTS_DIR / f"learning_curve_{model_name}.png",
            )
            
            print(f"Generated learning curve plot: {RESULTS_DIR}/learning_curve_{model_name}.png")

def enhance_results_markdown():
    """Enhance the results.md file with visualization references."""
    results_md_path = RESULTS_DIR / "results.md"
    
    if not os.path.exists(results_md_path):
        print(f"Results markdown file not found: {results_md_path}")
        return
    
    with open(results_md_path, "r") as f:
        content = f.read()
    
    # Check if visualizations are already referenced
    if "model_comparison.png" in content:
        print("Visualizations already referenced in results.md")
        return
    
    # Add visualization references
    enhanced_content = content.replace(
        "## Performance Comparison",
        "## Performance Comparison\n\n![Model Comparison](model_comparison.png)\n"
    )
    
    enhanced_content = enhanced_content.replace(
        "## Learning Progress",
        "## Learning Progress\n\n![Learning Curve - VERIL Static](learning_curve_veril_static.png)\n\n![Learning Curve - VERIL Dynamic](learning_curve_veril_dynamic.png)\n\n## Error Analysis\n\n![Error Types](error_types.png)\n"
    )
    
    with open(results_md_path, "w") as f:
        f.write(enhanced_content)
    
    print(f"Enhanced results markdown file: {results_md_path}")

def main():
    """Main entry point."""
    # Load results
    results = load_results()
    
    if not results:
        print("No results available. Please run the experiment first.")
        return
    
    print(f"Loaded results for {len(results)} models: {', '.join(results.keys())}")
    
    # Visualize results
    visualize_model_comparison(results)
    visualize_error_types(results)
    visualize_learning_curves(results)
    
    # Enhance results markdown
    enhance_results_markdown()
    
    print("Visualization completed successfully.")

if __name__ == "__main__":
    main()