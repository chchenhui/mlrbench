#!/usr/bin/env python3
"""
Benchmark Cards Experiment: Testing the effectiveness of Benchmark Cards
for holistic evaluation of machine learning models.

This script implements an experimental validation of the Benchmark Cards
methodology proposed in the paper, focusing on how incorporating context-specific
evaluation metrics affects model selection for different use cases.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'experiment.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

class BenchmarkCard:
    """
    Implementation of the Benchmark Card concept proposed in the paper.
    
    Benchmark Cards standardize contextual reporting of benchmark intentions,
    promote multi-metric evaluation, and document limitations.
    """
    
    def __init__(self, name, description, intended_use_cases, dataset_composition, 
                 evaluation_metrics, robustness_metrics, limitations, version="1.0"):
        """
        Initialize a Benchmark Card with its core components.
        
        Args:
            name (str): Name of the benchmark
            description (str): Brief description of the benchmark
            intended_use_cases (dict): Mapping of use case names to descriptions
            dataset_composition (dict): Information about dataset characteristics
            evaluation_metrics (dict): Metrics for evaluation with descriptions
            robustness_metrics (dict): Metrics for robustness evaluation
            limitations (list): Known limitations of the benchmark
            version (str): Version of the benchmark card
        """
        self.name = name
        self.description = description
        self.intended_use_cases = intended_use_cases
        self.dataset_composition = dataset_composition
        self.evaluation_metrics = evaluation_metrics
        self.robustness_metrics = robustness_metrics
        self.limitations = limitations
        self.version = version
        
        # Use case specific metric weights
        self.use_case_weights = {}
        
    def add_use_case_weights(self, use_case, metric_weights):
        """
        Add metric weights for a specific use case.
        
        Args:
            use_case (str): Name of the use case
            metric_weights (dict): Mapping of metric names to weights (should sum to 1)
        """
        # Check if all metrics exist
        for metric in metric_weights:
            if metric not in self.evaluation_metrics and metric not in self.robustness_metrics:
                raise ValueError(f"Metric '{metric}' not found in evaluation or robustness metrics")
        
        # Check if weights sum to 1
        if abs(sum(metric_weights.values()) - 1.0) > 1e-6:
            logger.warning(f"Weights for use case '{use_case}' do not sum to 1.0 (sum={sum(metric_weights.values())})")
            
        self.use_case_weights[use_case] = metric_weights
        
    def compute_composite_score(self, metric_values, use_case, thresholds=None):
        """
        Compute a composite score for a model based on multiple metrics and use case weights.
        
        Args:
            metric_values (dict): Mapping of metric names to values
            use_case (str): The specific use case to compute composite score for
            thresholds (dict, optional): Minimum acceptable values for each metric
            
        Returns:
            float: Composite score
        """
        if use_case not in self.use_case_weights:
            raise ValueError(f"Use case '{use_case}' not found in use case weights")
            
        weights = self.use_case_weights[use_case]
        
        # If no thresholds provided, default to 0 (no penalty)
        if thresholds is None:
            thresholds = {metric: 0 for metric in weights}
            
        composite_score = 0.0
        for metric, weight in weights.items():
            if metric not in metric_values:
                raise ValueError(f"Metric '{metric}' not found in provided metric values")
                
            metric_value = metric_values[metric]
            threshold = thresholds.get(metric, 0)
            
            # Apply normalization by threshold if it's not zero
            if threshold > 0:
                normalized_value = metric_value / threshold
            else:
                normalized_value = metric_value
                
            composite_score += weight * normalized_value
            
        return composite_score
    
    def to_dict(self):
        """Convert Benchmark Card to dictionary format for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "intended_use_cases": self.intended_use_cases,
            "dataset_composition": self.dataset_composition,
            "evaluation_metrics": self.evaluation_metrics,
            "robustness_metrics": self.robustness_metrics,
            "limitations": self.limitations,
            "version": self.version,
            "use_case_weights": self.use_case_weights
        }
        
    def to_json(self, indent=2):
        """Convert Benchmark Card to JSON format"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, file_path):
        """Save Benchmark Card to a JSON file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
            
    @classmethod
    def load(cls, file_path):
        """Load Benchmark Card from a JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        card = cls(
            name=data["name"],
            description=data["description"],
            intended_use_cases=data["intended_use_cases"],
            dataset_composition=data["dataset_composition"],
            evaluation_metrics=data["evaluation_metrics"],
            robustness_metrics=data["robustness_metrics"],
            limitations=data["limitations"],
            version=data.get("version", "1.0")
        )
        
        # Load use case weights
        for use_case, weights in data.get("use_case_weights", {}).items():
            card.use_case_weights[use_case] = weights
            
        return card


def load_dataset(dataset_name="adult", version=2):
    """
    Load a dataset for the experiment.
    
    Args:
        dataset_name (str): Name of the dataset on OpenML
        version (int): Version of the dataset
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, target_names
    """
    logger.info(f"Loading dataset: {dataset_name}, version {version}")
    
    try:
        # Fetch dataset from OpenML
        data = fetch_openml(name=dataset_name, version=version, as_frame=True)
        X, y = data.data, data.target
        feature_names = X.columns.tolist()
        
        # Get unique target classes
        target_names = y.unique().tolist()
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Identify categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for name, dtype in zip(feature_names, X.dtypes):
            # Check if feature is categorical
            if dtype == 'object' or dtype == 'category':
                categorical_features.append(name)
            # Check if feature looks categorical (few unique values)
            elif X[name].nunique() < 10:
                categorical_features.append(name)
            # Otherwise, assume numerical
            else:
                numerical_features.append(name)
        
        logger.info(f"Feature types: {len(categorical_features)} categorical, {len(numerical_features)} numerical")
        
        # Process categorical features
        for feature in categorical_features:
            # One-hot encode categorical features
            dummies = pd.get_dummies(X_train[feature], prefix=feature, drop_first=True)
            X_train = pd.concat([X_train.drop(feature, axis=1), dummies], axis=1)
            
            # Apply same transformation to test set
            dummies = pd.get_dummies(X_test[feature], prefix=feature, drop_first=True)
            X_test = pd.concat([X_test.drop(feature, axis=1), dummies], axis=1)
        
        # Update feature names after one-hot encoding
        feature_names = X_train.columns.tolist()
        
        # Scale numerical features
        if numerical_features:
            scaler = StandardScaler()
            X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
            X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        
        logger.info(f"Dataset loaded successfully: {len(X_train)} training samples, "
                   f"{len(X_test)} test samples, {len(feature_names)} features, "
                   f"{len(target_names)} classes")
        
        return X_train, X_test, y_train, y_test, feature_names, target_names
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def train_models(X_train, y_train):
    """
    Train a set of different models on the dataset.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Training models...")
    
    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "svm": SVC(random_state=42, probability=True),
        "mlp": MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"{name} trained successfully")
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    return trained_models


def evaluate_models(models, X_test, y_test, use_subgroups=True, sensitive_feature=None):
    """
    Evaluate models using multiple metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        use_subgroups (bool): Whether to evaluate on subgroups
        sensitive_feature (str): Feature used to define subgroups
        
    Returns:
        dict: Dictionary of model evaluation results
    """
    logger.info("Evaluating models...")
    
    # Define evaluation metrics
    metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
        "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        "roc_auc": lambda y_true, y_pred_proba: roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        )
    }
    
    # Dictionary to store evaluation results
    results = {}
    
    # Evaluate each model
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        model_results = {}
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        
        # Evaluate overall performance
        for metric_name, metric_fn in metrics.items():
            try:
                if metric_name == "roc_auc" and y_pred_proba is not None:
                    score = metric_fn(y_test, y_pred_proba)
                else:
                    score = metric_fn(y_test, y_pred)
                model_results[metric_name] = score
            except Exception as e:
                logger.error(f"Error computing {metric_name} for {name}: {e}")
                model_results[metric_name] = float("nan")
        
        # Evaluate subgroup performance if requested
        if use_subgroups and sensitive_feature is not None and sensitive_feature in X_test.columns:
            subgroup_results = {}
            
            # Get unique subgroups
            subgroups = X_test[sensitive_feature].unique()
            
            for subgroup in subgroups:
                # Filter test data for this subgroup
                mask = X_test[sensitive_feature] == subgroup
                X_subgroup = X_test[mask]
                y_subgroup = y_test[mask]
                
                if len(X_subgroup) == 0:
                    continue
                
                # Make predictions for this subgroup
                y_pred_subgroup = model.predict(X_subgroup)
                
                # Evaluate metrics for this subgroup
                subgroup_metrics = {}
                for metric_name, metric_fn in metrics.items():
                    # Skip ROC-AUC for subgroups for simplicity
                    if metric_name == "roc_auc":
                        continue
                    
                    try:
                        score = metric_fn(y_subgroup, y_pred_subgroup)
                        subgroup_metrics[metric_name] = score
                    except Exception as e:
                        logger.error(f"Error computing {metric_name} for {name} on subgroup {subgroup}: {e}")
                        subgroup_metrics[metric_name] = float("nan")
                
                subgroup_results[f"subgroup_{subgroup}"] = subgroup_metrics
            
            # Add fairness metrics
            if len(subgroup_results) > 1:
                # Calculate max disparity in accuracy between subgroups
                accuracies = [metrics["accuracy"] for subgroup, metrics in subgroup_results.items()]
                max_disparity = max(accuracies) - min(accuracies)
                model_results["fairness_disparity"] = max_disparity
            
            # Add subgroup results to overall results
            model_results["subgroup_performance"] = subgroup_results
        
        # Compute inference time (efficiency metric)
        try:
            start_time = datetime.now()
            _ = model.predict(X_test.iloc[:100])  # Use a smaller batch for timing
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            model_results["inference_time"] = inference_time
        except Exception as e:
            logger.error(f"Error computing inference time for {name}: {e}")
            model_results["inference_time"] = float("nan")
        
        # Compute model complexity (proxy for interpretability)
        if hasattr(model, "n_features_in_"):
            model_results["model_complexity"] = model.n_features_in_
        elif hasattr(model, "feature_importances_"):
            # Count non-zero feature importances
            model_results["model_complexity"] = sum(model.feature_importances_ > 0.01)
        else:
            model_results["model_complexity"] = float("nan")
        
        # Add model evaluation results to overall results
        results[name] = model_results
    
    return results


def create_benchmark_card(dataset_name, feature_names, target_names):
    """
    Create a Benchmark Card for the selected dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        feature_names (list): Names of features in the dataset
        target_names (list): Names of target classes
        
    Returns:
        BenchmarkCard: A benchmark card instance
    """
    logger.info(f"Creating Benchmark Card for {dataset_name}")
    
    # Define intended use cases based on the Adult dataset
    intended_use_cases = {
        "general_performance": "Evaluate overall predictive performance across all population groups",
        "fairness_focused": "Evaluate models with emphasis on fairness across demographic groups",
        "resource_constrained": "Evaluate models for deployment in resource-constrained environments",
        "interpretability_needed": "Evaluate models where interpretability is critical for domain experts",
        "robustness_required": "Evaluate models for robustness to distribution shifts and outliers"
    }
    
    # Define dataset composition
    dataset_composition = {
        "name": dataset_name,
        "num_samples": None,  # To be filled during experiment
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "target_names": target_names,
        "description": f"The {dataset_name} dataset is used for classification tasks."
    }
    
    # Define evaluation metrics
    evaluation_metrics = {
        "accuracy": "Proportion of correctly classified instances",
        "balanced_accuracy": "Accuracy adjusted for class imbalance",
        "precision": "Precision score (weighted average across classes)",
        "recall": "Recall score (weighted average across classes)",
        "f1_score": "F1 score (weighted average across classes)",
        "roc_auc": "Area under the ROC curve (weighted average across classes)",
        "fairness_disparity": "Maximum accuracy difference between demographic subgroups",
        "inference_time": "Time required for model inference (seconds)",
        "model_complexity": "Complexity of the model (feature count or non-zero importances)"
    }
    
    # Define robustness metrics
    robustness_metrics = {
        "fairness_disparity": "Maximum accuracy difference between demographic subgroups"
    }
    
    # Define limitations
    limitations = [
        "The benchmark may not fully capture all aspects of model performance",
        "Subgroup fairness is limited to predefined demographic attributes",
        "Inference time measurements are platform-dependent",
        "Model complexity is a simplified proxy for interpretability"
    ]
    
    # Create the benchmark card
    card = BenchmarkCard(
        name=f"{dataset_name} Benchmark Card",
        description=f"A benchmark card for evaluating classification models on the {dataset_name} dataset",
        intended_use_cases=intended_use_cases,
        dataset_composition=dataset_composition,
        evaluation_metrics=evaluation_metrics,
        robustness_metrics=robustness_metrics,
        limitations=limitations
    )
    
    # Define use case specific weights for the adult dataset
    # General performance
    card.add_use_case_weights("general_performance", {
        "accuracy": 0.3,
        "balanced_accuracy": 0.2,
        "precision": 0.2,
        "recall": 0.2,
        "f1_score": 0.1
    })
    
    # Fairness focused
    card.add_use_case_weights("fairness_focused", {
        "accuracy": 0.2,
        "fairness_disparity": 0.5,  # Higher weight for fairness
        "balanced_accuracy": 0.2,
        "f1_score": 0.1
    })
    
    # Resource constrained
    card.add_use_case_weights("resource_constrained", {
        "accuracy": 0.4,
        "inference_time": 0.4,  # Higher weight for efficiency
        "model_complexity": 0.2
    })
    
    # Interpretability needed
    card.add_use_case_weights("interpretability_needed", {
        "accuracy": 0.3,
        "model_complexity": 0.5,  # Higher weight for interpretability
        "precision": 0.1,
        "recall": 0.1
    })
    
    # Robustness required
    card.add_use_case_weights("robustness_required", {
        "accuracy": 0.2,
        "balanced_accuracy": 0.3,
        "fairness_disparity": 0.3,  # Higher weight for robustness
        "precision": 0.1,
        "recall": 0.1
    })
    
    return card


def simulate_users(model_results, benchmark_card, thresholds=None):
    """
    Simulate user model selection with and without benchmark cards.
    
    Args:
        model_results (dict): Model evaluation results
        benchmark_card (BenchmarkCard): Benchmark card for the dataset
        thresholds (dict, optional): Thresholds for metrics
        
    Returns:
        dict: User model selection results
    """
    logger.info("Simulating user model selection...")
    
    # Default selection (using only accuracy)
    default_selections = {}
    for use_case in benchmark_card.intended_use_cases:
        # For default, just select based on accuracy
        best_model = max(model_results.keys(), 
                          key=lambda model: model_results[model]["accuracy"])
        default_selections[use_case] = best_model
    
    # Benchmark card based selection
    card_selections = {}
    scores = {}
    
    for use_case in benchmark_card.intended_use_cases:
        use_case_scores = {}
        
        for model_name, results in model_results.items():
            # Get subset of metrics that are actually used in this use case
            metrics_used = benchmark_card.use_case_weights[use_case].keys()
            metric_values = {metric: results[metric] for metric in metrics_used if metric in results}
            
            # For metrics like inference_time and fairness_disparity, lower is better
            # So invert these values for scoring
            if "inference_time" in metric_values:
                metric_values["inference_time"] = 1.0 / max(metric_values["inference_time"], 0.001)
            
            if "fairness_disparity" in metric_values:
                metric_values["fairness_disparity"] = 1.0 - min(metric_values["fairness_disparity"], 1.0)
            
            # Compute composite score
            try:
                score = benchmark_card.compute_composite_score(
                    metric_values, use_case, thresholds
                )
                use_case_scores[model_name] = score
            except Exception as e:
                logger.error(f"Error computing score for {model_name} on use case {use_case}: {e}")
                use_case_scores[model_name] = float("-inf")
        
        # Select best model for this use case
        if use_case_scores:
            best_model = max(use_case_scores.keys(), key=lambda model: use_case_scores[model])
            card_selections[use_case] = best_model
            scores[use_case] = use_case_scores
        else:
            card_selections[use_case] = "no_selection"
            scores[use_case] = {}
    
    return {
        "default_selections": default_selections,
        "card_selections": card_selections,
        "scores": scores
    }


def visualize_results(model_results, simulation_results, output_dir, dataset_name):
    """
    Create visualizations of the experimental results.
    
    Args:
        model_results (dict): Model evaluation results
        simulation_results (dict): User model selection results
        output_dir (str): Directory to save visualizations
        dataset_name (str): Name of the dataset
    """
    logger.info("Creating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Extract model names and metrics for plotting
    model_names = list(model_results.keys())
    metrics = [m for m in model_results[model_names[0]].keys() if not isinstance(model_results[model_names[0]][m], dict)]
    
    # 1. Model performance comparison
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    
    for i, metric in enumerate(metrics):
        metric_values = []
        for model in model_names:
            metric_values.append(model_results[model][metric])
        
        ax = axes[i] if len(metrics) > 1 else axes
        bars = ax.bar(model_names, metric_values)
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_ylabel("Value")
        ax.set_xlabel("Model")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_model_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Radar charts for model comparisons across multiple metrics
    # Select a subset of metrics for radar chart
    radar_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
    if "fairness_disparity" in metrics:
        radar_metrics.append("fairness_disparity")
    
    radar_metrics = [m for m in radar_metrics if m in metrics]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Add labels for metrics
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
    
    # Scale data to [0, 1] for visualization
    model_data = {}
    for model in model_names:
        values = []
        for metric in radar_metrics:
            value = model_results[model][metric]
            
            # For fairness_disparity, lower is better, so invert
            if metric == "fairness_disparity":
                value = 1.0 - value
                
            values.append(value)
            
        values += values[:1]  # Close the polygon
        model_data[model] = values
    
    # Plot each model
    for i, (model, values) in enumerate(model_data.items()):
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_title(f"Model Comparison: {dataset_name}", size=15)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_radar_chart.png"), dpi=300)
    plt.close()
    
    # 3. Model selection comparison
    # Extract use cases and selections
    use_cases = list(simulation_results["default_selections"].keys())
    default_selections = [simulation_results["default_selections"][uc] for uc in use_cases]
    card_selections = [simulation_results["card_selections"][uc] for uc in use_cases]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(use_cases))
    width = 0.35
    
    ax.bar(x - width/2, [model_names.index(m) for m in default_selections], width, label='Default Selection')
    ax.bar(x + width/2, [model_names.index(m) for m in card_selections], width, label='Benchmark Card Selection')
    
    ax.set_xticks(x)
    ax.set_xticklabels([uc.replace('_', ' ').title() for uc in use_cases])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    
    ax.set_title('Model Selection Comparison')
    ax.set_xlabel('Use Case')
    ax.set_ylabel('Selected Model')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_selection_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Heatmap of scores for each model per use case
    for use_case in simulation_results["scores"]:
        scores = simulation_results["scores"][use_case]
        
        if not scores:
            continue
            
        # Convert to list for heatmap
        score_values = [scores[model] for model in model_names]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 2))
        sns.heatmap([score_values], annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=model_names, yticklabels=[use_case.replace('_', ' ').title()],
                    cbar_kws={'label': 'Composite Score'})
        
        ax.set_title(f'Model Scores for {use_case.replace("_", " ").title()} Use Case')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{use_case}_scores.png"), dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def generate_results_summary(model_results, simulation_results, dataset_name, benchmark_card):
    """
    Generate a summary of the experimental results.
    
    Args:
        model_results (dict): Model evaluation results
        simulation_results (dict): User model selection results
        dataset_name (str): Name of the dataset
        benchmark_card (BenchmarkCard): Benchmark card used for evaluation
        
    Returns:
        str: Markdown-formatted summary of results
    """
    logger.info("Generating results summary...")
    
    # Extract model names and metrics
    model_names = list(model_results.keys())
    metrics = [m for m in model_results[model_names[0]].keys() if not isinstance(model_results[model_names[0]][m], dict)]
    
    # Start building the markdown output
    md = f"# Benchmark Cards Experiment Results: {dataset_name}\n\n"
    
    # Dataset information
    md += "## Dataset Information\n\n"
    md += f"- **Dataset**: {dataset_name}\n"
    md += f"- **Features**: {len(benchmark_card.dataset_composition['feature_names'])}\n"
    md += f"- **Target Classes**: {benchmark_card.dataset_composition['target_names']}\n\n"
    
    # Model performance table
    md += "## Model Performance\n\n"
    md += "The following table shows the performance of different models on the test set:\n\n"
    
    # Create table header
    md += "| Model | " + " | ".join([m.replace('_', ' ').title() for m in metrics]) + " |\n"
    md += "| --- | " + " | ".join(["---" for _ in metrics]) + " |\n"
    
    # Add model results to table
    for model in model_names:
        row = f"| {model} |"
        for metric in metrics:
            value = model_results[model][metric]
            if isinstance(value, float):
                row += f" {value:.4f} |"
            else:
                row += f" {value} |"
        md += row + "\n"
    
    md += "\n"
    
    # Model selection comparison
    md += "## Model Selection Comparison\n\n"
    md += "This table compares model selection with and without Benchmark Cards for different use cases:\n\n"
    
    # Create table header
    md += "| Use Case | Default Selection (Accuracy Only) | Benchmark Card Selection | Different? |\n"
    md += "| --- | --- | --- | --- |\n"
    
    # Add selection results to table
    different_count = 0
    use_cases = list(simulation_results["default_selections"].keys())
    
    for use_case in use_cases:
        default = simulation_results["default_selections"][use_case]
        card = simulation_results["card_selections"][use_case]
        different = "Yes" if default != card else "No"
        if different == "Yes":
            different_count += 1
            
        md += f"| {use_case.replace('_', ' ').title()} | {default} | {card} | {different} |\n"
    
    # Summary statistics
    percentage_different = (different_count / len(use_cases)) * 100
    md += f"\n**Summary**: Benchmark Cards resulted in different model selections in "
    md += f"{different_count} out of {len(use_cases)} use cases ({percentage_different:.1f}%).\n\n"
    
    # Use case specific weights
    md += "## Use Case Specific Metric Weights\n\n"
    md += "The Benchmark Card defined the following weights for each use case:\n\n"
    
    for use_case, weights in benchmark_card.use_case_weights.items():
        md += f"### {use_case.replace('_', ' ').title()}\n\n"
        
        # Create table header
        md += "| Metric | Weight |\n"
        md += "| --- | --- |\n"
        
        # Add weights to table
        for metric, weight in weights.items():
            md += f"| {metric.replace('_', ' ').title()} | {weight:.2f} |\n"
        
        md += "\n"
    
    # Conclusions
    md += "## Conclusions\n\n"
    
    if percentage_different > 50:
        md += "The experiment showed that using Benchmark Cards significantly changes model selection "
        md += "compared to using accuracy as the only metric. This demonstrates the value of holistic, "
        md += "context-aware evaluation in machine learning.\n\n"
    else:
        md += "The experiment showed that using Benchmark Cards sometimes leads to different model selections "
        md += "compared to using accuracy as the only metric. This highlights the potential value of holistic, "
        md += "context-aware evaluation, especially in specific use cases where non-accuracy metrics are important.\n\n"
    
    # Add specific insights based on the results
    # Find which use cases had different selections
    different_use_cases = [use_case for use_case in use_cases 
                           if simulation_results["default_selections"][use_case] != 
                              simulation_results["card_selections"][use_case]]
    
    if different_use_cases:
        md += "### Key Insights\n\n"
        
        for use_case in different_use_cases:
            default_model = simulation_results["default_selections"][use_case]
            card_model = simulation_results["card_selections"][use_case]
            
            md += f"- For the **{use_case.replace('_', ' ').title()}** use case, the Benchmark Card selected "
            md += f"**{card_model}** instead of **{default_model}** (highest accuracy).\n"
            
            # Find key differences in metrics
            default_metrics = model_results[default_model]
            card_metrics = model_results[card_model]
            
            # Get metrics important for this use case (high weight)
            important_metrics = [m for m, w in benchmark_card.use_case_weights[use_case].items() 
                                if w >= 0.2 and m in default_metrics and m in card_metrics]
            
            if important_metrics:
                md += "  - Key metric differences:\n"
                for metric in important_metrics:
                    default_value = default_metrics[metric]
                    card_value = card_metrics[metric]
                    
                    # For some metrics, lower is better
                    better_model = None
                    if metric in ["inference_time", "fairness_disparity"]:
                        better_model = "selected" if card_value < default_value else "default"
                    else:
                        better_model = "selected" if card_value > default_value else "default"
                    
                    md += f"    - **{metric.replace('_', ' ').title()}**: {card_value:.4f} vs {default_value:.4f} "
                    md += f"(better in {better_model} model)\n"
            
            md += "\n"
    
    # Limitations and future work
    md += "## Limitations and Future Work\n\n"
    md += "This experiment demonstrates the concept of Benchmark Cards, but has several limitations:\n\n"
    md += "1. The experiment used a single dataset and a small set of models.\n"
    md += "2. The use cases and metric weights were defined artificially rather than by domain experts.\n"
    md += "3. The simulation doesn't fully capture the complexity of real-world model selection decisions.\n\n"
    
    md += "Future work could address these limitations by:\n\n"
    md += "1. Expanding to multiple datasets across different domains.\n"
    md += "2. Conducting surveys with domain experts to define realistic use cases and metric weights.\n"
    md += "3. Implementing a more sophisticated composite scoring formula that better handles trade-offs.\n"
    md += "4. Developing an interactive tool that allows users to explore model performance with custom weights.\n"
    
    return md


def run_experiment(dataset_name="adult", output_dir="results", sensitive_feature=None, version=1):
    """
    Run the complete experiment.
    
    Args:
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save results
        sensitive_feature (str, optional): Sensitive feature for fairness evaluation
        version (int): Version of the dataset
        
    Returns:
        dict: Complete experiment results
    """
    start_time = datetime.now()
    logger.info(f"Starting experiment with dataset: {dataset_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load dataset
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(dataset_name, version)
    
    # Step 2: Create benchmark card
    benchmark_card = create_benchmark_card(dataset_name, feature_names, target_names)
    
    # Update dataset composition with actual numbers
    benchmark_card.dataset_composition["num_samples"] = len(X_train) + len(X_test)
    
    # Save benchmark card
    benchmark_card.save(os.path.join(output_dir, f"{dataset_name}_benchmark_card.json"))
    logger.info(f"Benchmark card saved to {os.path.join(output_dir, f'{dataset_name}_benchmark_card.json')}")
    
    # Step 3: Train models
    models = train_models(X_train, y_train)
    
    # Step 4: Evaluate models
    model_results = evaluate_models(models, X_test, y_test, 
                                   use_subgroups=sensitive_feature is not None,
                                   sensitive_feature=sensitive_feature)
    
    # Save model results
    with open(os.path.join(output_dir, f"{dataset_name}_model_results.json"), "w") as f:
        # Convert numpy values to float for JSON serialization
        serializable_results = {}
        for model_name, results in model_results.items():
            serializable_results[model_name] = {}
            for metric, value in results.items():
                if isinstance(value, dict):
                    serializable_results[model_name][metric] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            serializable_results[model_name][metric][k] = {
                                mk: float(mv) if isinstance(mv, (np.number, np.floating, np.integer)) else mv
                                for mk, mv in v.items()
                            }
                        else:
                            serializable_results[model_name][metric][k] = float(v) if isinstance(v, (np.number, np.floating, np.integer)) else v
                else:
                    serializable_results[model_name][metric] = float(value) if isinstance(value, (np.number, np.floating, np.integer)) else value
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Model results saved to {os.path.join(output_dir, f'{dataset_name}_model_results.json')}")
    
    # Step 5: Simulate user model selection
    simulation_results = simulate_users(model_results, benchmark_card)
    
    # Save simulation results
    with open(os.path.join(output_dir, f"{dataset_name}_simulation_results.json"), "w") as f:
        json.dump(simulation_results, f, indent=2)
    
    logger.info(f"Simulation results saved to {os.path.join(output_dir, f'{dataset_name}_simulation_results.json')}")
    
    # Step 6: Create visualizations
    visualize_results(model_results, simulation_results, output_dir, dataset_name)
    
    # Step 7: Generate results summary
    results_summary = generate_results_summary(model_results, simulation_results, dataset_name, benchmark_card)
    
    # Save results summary
    with open(os.path.join(output_dir, "results.md"), "w") as f:
        f.write(results_summary)
    
    logger.info(f"Results summary saved to {os.path.join(output_dir, 'results.md')}")
    
    # Log experiment completion time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Experiment completed in {duration:.2f} seconds")
    
    return {
        "dataset_name": dataset_name,
        "benchmark_card": benchmark_card.to_dict(),
        "model_results": model_results,
        "simulation_results": simulation_results,
        "duration": duration
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Cards Experiment")
    parser.add_argument("--dataset", type=str, default="adult",
                        help="Name of the dataset to use")
    parser.add_argument("--version", type=int, default=1,
                        help="Version of the dataset")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--sensitive-feature", type=str, default=None,
                        help="Sensitive feature for fairness evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create log file for experiment
    log_file = os.path.join(os.path.dirname(__file__), "log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Print start message
    logger.info("Starting Benchmark Cards Experiment")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Dataset version: {args.version}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sensitive feature: {args.sensitive_feature}")
    
    # Run the experiment
    results = run_experiment(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        sensitive_feature=args.sensitive_feature,
        version=args.version
    )
    
    # Print completion message
    logger.info("Experiment completed successfully")