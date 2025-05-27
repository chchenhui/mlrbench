"""
Minimal version of the experiment runner for testing

This script runs a smaller version of the ContextBench experiment to test if everything works.
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_mini_experiment(output_dir='../results'):
    """
    Run a minimal version of the ContextBench experiment.
    
    Args:
        output_dir: Directory to save results
    """
    # Start timer
    start_time = time.time()
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    logger.info("Running minimal ContextBench experiment...")
    
    # Load a small dataset (Adult UCI)
    logger.info("Loading Adult dataset...")
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    X = adult.data
    y = (adult.target == '>50K').astype(int)
    
    # Use only a small subset for testing
    X = X.iloc[:2000]
    y = y.iloc[:2000]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess data
    logger.info("Preprocessing data...")
    
    # Convert categorical features to one-hot encoding
    X_train_processed = pd.get_dummies(X_train, drop_first=True)
    X_test_processed = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure the same columns in train and test
    common_cols = list(set(X_train_processed.columns) & set(X_test_processed.columns))
    X_train_processed = X_train_processed[common_cols]
    X_test_processed = X_test_processed[common_cols]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train_processed)
    X_test_processed = scaler.transform(X_test_processed)
    
    # Create simple models
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=100),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=10)
    }
    
    # Train and evaluate models
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_processed, y_train)
        test_score = model.score(X_test_processed, y_test)
        
        logger.info(f"{model_name} - Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        # Store results
        results[model_name] = {
            'train_score': train_score,
            'test_score': test_score
        }
    
    # Generate a simple visualization
    plt.figure(figsize=(10, 6))
    
    x = list(results.keys())
    train_scores = [results[model]['train_score'] for model in x]
    test_scores = [results[model]['test_score'] for model in x]
    
    bar_width = 0.35
    x_pos = np.arange(len(x))
    
    plt.bar(x_pos - bar_width/2, train_scores, bar_width, label='Train')
    plt.bar(x_pos + bar_width/2, test_scores, bar_width, label='Test')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(x_pos, x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the visualization
    vis_path = os.path.join(output_dir, 'visualizations', 'model_comparison.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {vis_path}")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'mini_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Generate results.md
    with open(os.path.join(output_dir, 'results.md'), 'w') as f:
        # Write header
        f.write("# ContextBench Mini Experiment Results\n\n")
        
        # Write model comparison table
        f.write("## Model Comparison\n\n")
        f.write("| Model | Train Accuracy | Test Accuracy |\n")
        f.write("|-------|---------------|---------------|\n")
        
        for model_name, scores in results.items():
            train_score = scores['train_score']
            test_score = scores['test_score']
            f.write(f"| {model_name} | {train_score:.4f} | {test_score:.4f} |\n")
        
        f.write("\n")
        
        # Add visualization
        f.write("## Visualization\n\n")
        vis_rel_path = os.path.relpath(vis_path, output_dir)
        f.write(f"![Model Comparison]({vis_rel_path})\n\n")
        
        # Write conclusions
        f.write("## Conclusions\n\n")
        best_model = max(results.items(), key=lambda x: x[1]['test_score'])[0]
        f.write(f"The {best_model} model achieved the best test accuracy in this experiment.\n")
    
    logger.info(f"Generated results.md in {output_dir}")
    
    # Calculate total experiment time
    total_time = time.time() - start_time
    logger.info(f"Total experiment time: {total_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    run_mini_experiment()