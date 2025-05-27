#!/usr/bin/env python3
"""
Model training and evaluation utilities for Benchmark Cards experiments.
This script handles training and evaluating various ML models.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

# Import our own modules
from data_processing import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

def get_model_list():
    """
    Get a list of models to train.
    
    Returns:
        dict: Dictionary of model name to model constructor
    """
    return {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "svm": SVC(random_state=42, probability=True),
        "mlp": MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50)),
        "gradient_boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
    }


def train_model(model, X_train, y_train, model_name, use_cv=False, n_folds=5):
    """
    Train a model on the given data.
    
    Args:
        model: The model to train
        X_train: Training features
        y_train: Training labels
        model_name (str): Name of the model (for logging)
        use_cv (bool): Whether to use cross-validation
        n_folds (int): Number of folds for cross-validation
        
    Returns:
        tuple: (trained_model, training_time, cv_scores if use_cv else None)
    """
    logger.info(f"Training {model_name}...")
    
    # Record training time
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Compute training time
    training_time = time.time() - start_time
    
    logger.info(f"{model_name} trained in {training_time:.2f} seconds")
    
    # Perform cross-validation if requested
    cv_scores = None
    if use_cv:
        logger.info(f"Performing {n_folds}-fold cross-validation for {model_name}")
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model, training_time, cv_scores


def evaluate_model(model, X_test, y_test, model_name, sensitive_feature=None, sensitive_values=None):
    """
    Evaluate a model on test data across multiple metrics.
    
    Args:
        model: The trained model to evaluate
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model (for logging)
        sensitive_feature (str, optional): Name of a sensitive feature for fairness evaluation
        sensitive_values (list, optional): List of values in X_test to use for subgroup evaluation
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Prepare results dictionary
    results = {}
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Get probability predictions if available
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    
    # Compute overall metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred, average='weighted')
    results['recall'] = recall_score(y_test, y_pred, average='weighted')
    results['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    
    # Compute ROC AUC if probability predictions are available
    if y_pred_proba is not None:
        try:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            results['roc_auc'] = np.nan
    
    # Compute confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    # Compute inference time
    results['inference_time'] = inference_time
    
    # Compute model complexity
    if hasattr(model, "n_features_in_"):
        results['model_complexity'] = int(model.n_features_in_)
    elif hasattr(model, "feature_importances_"):
        # Count non-zero feature importances
        results['model_complexity'] = int(np.sum(model.feature_importances_ > 0.01))
    else:
        results['model_complexity'] = np.nan
    
    # Evaluate on subgroups if sensitive feature is provided
    if sensitive_feature is not None and sensitive_values is not None:
        subgroup_results = {}
        subgroup_accuracies = []
        
        for value in sensitive_values:
            # Filter test data for this subgroup
            mask = X_test[sensitive_feature] == value
            X_subgroup = X_test[mask]
            y_subgroup = y_test[mask]
            
            if len(X_subgroup) > 0:
                # Make predictions for this subgroup
                y_pred_subgroup = model.predict(X_subgroup)
                
                # Compute metrics for this subgroup
                subgroup_accuracy = accuracy_score(y_subgroup, y_pred_subgroup)
                subgroup_results[f'accuracy_subgroup_{value}'] = subgroup_accuracy
                subgroup_accuracies.append(subgroup_accuracy)
        
        # Compute fairness disparity (max difference in accuracy between subgroups)
        if len(subgroup_accuracies) > 1:
            results['fairness_disparity'] = max(subgroup_accuracies) - min(subgroup_accuracies)
            
        # Add subgroup results to overall results
        results['subgroup_performance'] = subgroup_results
    
    # Log results
    logger.info(f"{model_name} evaluation results:")
    for metric, value in results.items():
        if metric not in ['confusion_matrix', 'subgroup_performance']:
            logger.info(f"  {metric}: {value}")
    
    return results


def train_and_evaluate_all_models(dataset_name, output_dir, version=1, use_cv=False, n_folds=5, 
                                sensitive_feature=None):
    """
    Train and evaluate all models on a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save results
        version (int): Version of the dataset
        use_cv (bool): Whether to use cross-validation
        n_folds (int): Number of folds for cross-validation
        sensitive_feature (str, optional): Name of a sensitive feature for fairness evaluation
        
    Returns:
        tuple: (all_results, training_history)
    """
    logger.info(f"Training and evaluating models on {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset_info = load_dataset(dataset_name, version)
    X_train = dataset_info['X_train']
    X_test = dataset_info['X_test']
    y_train = dataset_info['y_train']
    y_test = dataset_info['y_test']
    raw_X_test = dataset_info['raw_X_test']
    
    # Get sensitive values if sensitive feature is provided
    sensitive_values = None
    if sensitive_feature is not None and sensitive_feature in raw_X_test.columns:
        sensitive_values = raw_X_test[sensitive_feature].unique().tolist()
    
    # Get models to train
    models = get_model_list()
    
    # Results dictionary
    all_results = {}
    training_history = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        # Train model
        trained_model, training_time, cv_scores = train_model(
            model, X_train, y_train, model_name, use_cv, n_folds
        )
        
        # Record training information
        training_history[model_name] = {
            'training_time': training_time,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'cv_mean': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_std': float(cv_scores.std()) if cv_scores is not None else None
        }
        
        # Evaluate model
        results = evaluate_model(
            trained_model, X_test, y_test, model_name, 
            sensitive_feature, sensitive_values
        )
        
        # Add training information to results
        results['training_time'] = training_time
        
        # Add results to overall results
        all_results[model_name] = results
    
    # Save results to file
    with open(os.path.join(output_dir, f"{dataset_name}_model_results.json"), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for model_name, results in all_results.items():
            serializable_results[model_name] = {}
            for metric, value in results.items():
                if isinstance(value, (np.number, np.float_, np.int_)):
                    serializable_results[model_name][metric] = float(value)
                else:
                    serializable_results[model_name][metric] = value
        
        json.dump(serializable_results, f, indent=2)
    
    # Save training history to file
    with open(os.path.join(output_dir, f"{dataset_name}_training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Results saved to {os.path.join(output_dir, f'{dataset_name}_model_results.json')}")
    
    return all_results, training_history


def visualize_results(all_results, dataset_name, output_dir):
    """
    Create visualizations of model evaluation results.
    
    Args:
        all_results (dict): Dictionary of model evaluation results
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save visualizations
    """
    logger.info(f"Creating visualizations for {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and metrics
    model_names = list(all_results.keys())
    
    # Select metrics to visualize
    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
    if 'roc_auc' in all_results[model_names[0]]:
        metrics.append('roc_auc')
    if 'fairness_disparity' in all_results[model_names[0]]:
        metrics.append('fairness_disparity')
    if 'inference_time' in all_results[model_names[0]]:
        metrics.append('inference_time')
    if 'model_complexity' in all_results[model_names[0]]:
        metrics.append('model_complexity')
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(index=model_names, columns=metrics)
    for model in model_names:
        for metric in metrics:
            if metric in all_results[model]:
                df.loc[model, metric] = all_results[model][metric]
    
    # Plot each metric
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=df.index, y=df[metric])
            ax.set_title(f"{metric.replace('_', ' ').title()} by Model")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Add value labels on top of bars
            for i, v in enumerate(df[metric]):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_{metric}.png"), dpi=300)
            plt.close()
    
    # Create heatmap of all metrics
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"Model Performance Metrics: {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_metrics_heatmap.png"), dpi=300)
    plt.close()
    
    # Plot radar chart for model comparison across multiple metrics
    # (similar to the one in visualize_results.py)
    radar_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']
    if 'roc_auc' in df.columns:
        radar_metrics.append('roc_auc')
    
    # Only include metrics that are present
    radar_metrics = [m for m in radar_metrics if m in df.columns]
    
    if len(radar_metrics) > 2:  # Need at least 3 metrics for a meaningful radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Add labels for metrics
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
        
        # Normalize data for better visualization
        min_values = df[radar_metrics].min()
        max_values = df[radar_metrics].max()
        norm_df = (df[radar_metrics] - min_values) / (max_values - min_values)
        
        # Plot each model
        for i, model in enumerate(model_names):
            values = norm_df.loc[model].tolist()
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_title(f"Model Comparison: {dataset_name}", size=15)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_radar_chart.png"), dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to train and evaluate models on a dataset."""
    parser = argparse.ArgumentParser(description="Train and evaluate models on a dataset")
    parser.add_argument("--dataset", type=str, default="adult",
                        help="Name of the dataset on OpenML")
    parser.add_argument("--version", type=int, default=1,
                        help="Version of the dataset")
    parser.add_argument("--output-dir", type=str, default="model_results",
                        help="Directory to save results")
    parser.add_argument("--use-cv", action="store_true",
                        help="Use cross-validation during training")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--sensitive-feature", type=str, default=None,
                        help="Name of a sensitive feature for fairness evaluation")
    args = parser.parse_args()
    
    try:
        # Train and evaluate models
        all_results, training_history = train_and_evaluate_all_models(
            args.dataset, args.output_dir, args.version, args.use_cv, args.n_folds, args.sensitive_feature
        )
        
        # Create visualizations
        visualize_results(all_results, args.dataset, args.output_dir)
        
        logger.info("Model training and evaluation completed successfully")
    
    except Exception as e:
        logger.error(f"Error training and evaluating models: {e}")
        logger.exception(e)


if __name__ == "__main__":
    main()