#!/usr/bin/env python3
"""
Generate benchmark cards for selected datasets.
This script creates benchmark card templates that can be edited and refined.
"""

import os
import sys
import json
import argparse
import logging
from sklearn.datasets import fetch_openml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_benchmark_card(dataset_name, output_dir, version=1):
    """Generate a benchmark card for a dataset."""
    logger.info(f"Generating benchmark card for {dataset_name}")
    
    try:
        # Fetch dataset info from OpenML
        data = fetch_openml(name=dataset_name, version=version, as_frame=True)
        
        # Get dataset information
        X, y = data.data, data.target
        feature_names = X.columns.tolist()
        target_names = y.unique().tolist()
        
        # Create benchmark card content
        card = {
            "name": f"{dataset_name.capitalize()} Benchmark",
            "description": f"A benchmark for evaluating classification models on the {dataset_name} dataset",
            "intended_use_cases": {
                "general_performance": "Evaluate overall predictive performance across all population groups",
                "fairness_focused": "Evaluate models with emphasis on fairness across demographic groups",
                "resource_constrained": "Evaluate models for deployment in resource-constrained environments",
                "interpretability_needed": "Evaluate models where interpretability is critical for domain experts",
                "robustness_required": "Evaluate models for robustness to distribution shifts and outliers"
            },
            "dataset_composition": {
                "name": dataset_name,
                "num_samples": len(X) + len(y),
                "num_features": len(feature_names),
                "feature_names": feature_names,
                "target_names": target_names,
                "description": f"The {dataset_name} dataset contains {len(X)} samples with {len(feature_names)} features."
            },
            "evaluation_metrics": {
                "accuracy": "Proportion of correctly classified instances",
                "balanced_accuracy": "Accuracy adjusted for class imbalance",
                "precision": "Precision score (weighted average across classes)",
                "recall": "Recall score (weighted average across classes)",
                "f1_score": "F1 score (weighted average across classes)",
                "roc_auc": "Area under the ROC curve",
                "fairness_disparity": "Maximum accuracy difference between demographic subgroups",
                "inference_time": "Time required for model inference (seconds)",
                "model_complexity": "Complexity of the model (proxy for interpretability)"
            },
            "robustness_metrics": {
                "fairness_disparity": "Maximum accuracy difference between demographic subgroups"
            },
            "limitations": [
                "The benchmark may not fully capture all aspects of model performance",
                "Subgroup fairness is limited to predefined demographic attributes",
                "Inference time measurements are platform-dependent",
                "Model complexity is a simplified proxy for interpretability"
            ],
            "version": "1.0",
            "use_case_weights": {
                "general_performance": {
                    "accuracy": 0.3,
                    "balanced_accuracy": 0.2,
                    "precision": 0.2,
                    "recall": 0.2,
                    "f1_score": 0.1
                },
                "fairness_focused": {
                    "accuracy": 0.2,
                    "fairness_disparity": 0.5,
                    "balanced_accuracy": 0.2,
                    "f1_score": 0.1
                },
                "resource_constrained": {
                    "accuracy": 0.4,
                    "inference_time": 0.4,
                    "model_complexity": 0.2
                },
                "interpretability_needed": {
                    "accuracy": 0.3,
                    "model_complexity": 0.5,
                    "precision": 0.1,
                    "recall": 0.1
                },
                "robustness_required": {
                    "accuracy": 0.2,
                    "balanced_accuracy": 0.3,
                    "fairness_disparity": 0.3,
                    "precision": 0.1,
                    "recall": 0.1
                }
            }
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save benchmark card to JSON file
        output_file = os.path.join(output_dir, f"{dataset_name}_benchmark_card.json")
        with open(output_file, 'w') as f:
            json.dump(card, f, indent=2)
        
        logger.info(f"Benchmark card saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating benchmark card for {dataset_name}: {e}")
        return False


def main():
    """Main function to generate benchmark cards."""
    parser = argparse.ArgumentParser(description="Generate benchmark cards for datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["adult", "diabetes", "credit-g"],
                        help="Datasets to generate cards for")
    parser.add_argument("--output-dir", type=str, default="benchmark_cards",
                        help="Directory to save benchmark cards")
    args = parser.parse_args()
    
    # Get full paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    
    # Generate benchmark cards for each dataset
    for dataset in args.datasets:
        generate_benchmark_card(dataset, output_dir)


if __name__ == "__main__":
    main()