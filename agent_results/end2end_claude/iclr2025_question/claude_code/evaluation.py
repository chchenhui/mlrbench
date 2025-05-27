"""
Evaluation module for Reasoning Uncertainty Networks (RUNs) experiment.
"""
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    confusion_matrix,
    brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import OUTPUTS_DIR, EVAL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluator for hallucination detection models.
    """
    
    def __init__(self, model_name: str, model_type: str = "runs"):
        """
        Initialize the model evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            model_type: Type of model ("runs", "baseline", etc.)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.results = {}
        self.predictions = []
        self.ground_truth = []
        self.hallucination_scores = []
    
    def reset(self) -> None:
        """Reset evaluation results."""
        self.results = {}
        self.predictions = []
        self.ground_truth = []
        self.hallucination_scores = []
    
    def add_prediction(self, prediction: bool, ground_truth: bool, score: float = None) -> None:
        """
        Add a prediction to the evaluation.
        
        Args:
            prediction: Predicted hallucination (True/False)
            ground_truth: Actual hallucination (True/False)
            score: Hallucination score (0-1)
        """
        self.predictions.append(int(prediction))
        self.ground_truth.append(int(ground_truth))
        if score is not None:
            self.hallucination_scores.append(score)
    
    def add_batch(self, predictions: List[bool], ground_truth: List[bool], scores: List[float] = None) -> None:
        """
        Add a batch of predictions to the evaluation.
        
        Args:
            predictions: List of predicted hallucinations (True/False)
            ground_truth: List of actual hallucinations (True/False)
            scores: List of hallucination scores (0-1)
        """
        self.predictions.extend([int(p) for p in predictions])
        self.ground_truth.extend([int(gt) for gt in ground_truth])
        if scores is not None:
            self.hallucination_scores.extend(scores)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions or not self.ground_truth:
            logger.warning("No predictions to evaluate")
            return {}
        
        # Ensure equal lengths
        assert len(self.predictions) == len(self.ground_truth), "Predictions and ground truth must have the same length"
        
        # Basic classification metrics
        metrics = {}
        metrics["precision"] = precision_score(self.ground_truth, self.predictions, zero_division=0)
        metrics["recall"] = recall_score(self.ground_truth, self.predictions, zero_division=0)
        metrics["f1"] = f1_score(self.ground_truth, self.predictions, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.ground_truth, self.predictions, labels=[0, 1]).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        
        # Compute false positive rate and false negative rate
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Advanced metrics if scores are available
        if self.hallucination_scores:
            # ROC AUC
            try:
                metrics["auroc"] = roc_auc_score(self.ground_truth, self.hallucination_scores)
            except:
                metrics["auroc"] = 0.5  # Default value if computation fails
            
            # Precision-Recall AUC
            try:
                precision, recall, _ = precision_recall_curve(self.ground_truth, self.hallucination_scores)
                metrics["auprc"] = auc(recall, precision)
            except:
                metrics["auprc"] = 0.0  # Default value if computation fails
            
            # Brier score
            try:
                metrics["brier"] = brier_score_loss(self.ground_truth, self.hallucination_scores)
            except:
                metrics["brier"] = 1.0  # Default value if computation fails
        
        # Expected Calibration Error (ECE)
        # This is a simplified version, a more complex implementation would use binning
        if self.hallucination_scores:
            metrics["ece"] = self._compute_ece()
        
        self.results = metrics
        return metrics
    
    def _compute_ece(self, num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            num_bins: Number of bins for calibration
            
        Returns:
            ECE value
        """
        # Bin the predictions
        bin_indices = np.digitize(self.hallucination_scores, np.linspace(0, 1, num_bins+1)[:-1])
        
        ece = 0.0
        for bin_idx in range(1, num_bins+1):
            bin_mask = (bin_indices == bin_idx)
            if np.sum(bin_mask) > 0:
                bin_confidence = np.mean(np.array(self.hallucination_scores)[bin_mask])
                bin_accuracy = np.mean(np.array(self.ground_truth)[bin_mask])
                bin_count = np.sum(bin_mask)
                
                # Add weighted absolute difference
                ece += (bin_count / len(self.hallucination_scores)) * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def visualize_calibration(self, output_path: Optional[str] = None) -> None:
        """
        Create a calibration plot.
        
        Args:
            output_path: Optional path to save the visualization
        """
        if not self.hallucination_scores:
            logger.warning("No scores available for calibration plot")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Bin the predictions for visualization
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins+1)
        bin_indices = np.digitize(self.hallucination_scores, bin_edges[:-1])
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_idx in range(1, num_bins+1):
            bin_mask = (bin_indices == bin_idx)
            if np.sum(bin_mask) > 0:
                bin_confidence = np.mean(np.array(self.hallucination_scores)[bin_mask])
                bin_accuracy = np.mean(np.array(self.ground_truth)[bin_mask])
                bin_count = np.sum(bin_mask)
                
                bin_confidences.append(bin_confidence)
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(bin_count)
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
        
        # Actual calibration
        plt.scatter(bin_confidences, bin_accuracies, s=[100 * c / max(bin_counts) for c in bin_counts], 
                   alpha=0.7, label='Model calibration')
        
        plt.xlabel('Predicted probability')
        plt.ylabel('True probability')
        plt.title(f'Calibration Plot - {self.model_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved calibration plot to {output_path}")
        
        plt.close()
    
    def visualize_confusion_matrix(self, output_path: Optional[str] = None) -> None:
        """
        Create a confusion matrix visualization.
        
        Args:
            output_path: Optional path to save the visualization
        """
        if not self.predictions or not self.ground_truth:
            logger.warning("No predictions available for confusion matrix")
            return
        
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(self.ground_truth, self.predictions, labels=[0, 1])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Hallucination', 'Hallucination'],
                   yticklabels=['Not Hallucination', 'Hallucination'])
        
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {self.model_name}')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {output_path}")
        
        plt.close()
    
    def visualize_roc_curve(self, output_path: Optional[str] = None) -> None:
        """
        Create a ROC curve visualization.
        
        Args:
            output_path: Optional path to save the visualization
        """
        if not self.hallucination_scores:
            logger.warning("No scores available for ROC curve")
            return
        
        from sklearn.metrics import roc_curve
        
        plt.figure(figsize=(10, 8))
        
        fpr, tpr, _ = roc_curve(self.ground_truth, self.hallucination_scores)
        auroc = self.results.get("auroc", 0.5)
        
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {auroc:.3f})')
        
        # Add reference line
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curve to {output_path}")
        
        plt.close()
    
    def visualize_pr_curve(self, output_path: Optional[str] = None) -> None:
        """
        Create a Precision-Recall curve visualization.
        
        Args:
            output_path: Optional path to save the visualization
        """
        if not self.hallucination_scores:
            logger.warning("No scores available for PR curve")
            return
        
        plt.figure(figsize=(10, 8))
        
        precision, recall, _ = precision_recall_curve(self.ground_truth, self.hallucination_scores)
        auprc = self.results.get("auprc", 0.0)
        
        plt.plot(recall, precision, label=f'{self.model_name} (AUC = {auprc:.3f})')
        
        # Add baseline
        baseline = sum(self.ground_truth) / len(self.ground_truth)
        plt.axhline(y=baseline, linestyle='--', color='gray', label=f'Baseline (y = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved PR curve to {output_path}")
        
        plt.close()
    
    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        # Ensure metrics are computed
        if not self.results:
            self.compute_metrics()
        
        # Create results dictionary
        results_dict = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "metrics": self.results,
            "num_samples": len(self.predictions)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")


class ComparisonEvaluator:
    """
    Evaluator for comparing different hallucination detection models.
    """
    
    def __init__(self, models: List[str], model_types: List[str] = None):
        """
        Initialize the comparison evaluator.
        
        Args:
            models: List of model names to compare
            model_types: List of model types corresponding to the models
        """
        self.models = models
        self.model_types = model_types or ["baseline"] * len(models)
        self.results = {}
        self.metrics = EVAL_CONFIG["metrics"]
    
    def load_results(self, results_dir: str) -> Dict:
        """
        Load evaluation results for all models.
        
        Args:
            results_dir: Directory containing results files
            
        Returns:
            Dictionary of results by model
        """
        results = {}
        
        for model_name in self.models:
            results_file = os.path.join(results_dir, f"{model_name}_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    model_results = json.load(f)
                results[model_name] = model_results
                logger.info(f"Loaded results for {model_name}")
            else:
                logger.warning(f"Results file not found for {model_name}")
        
        self.results = results
        return results
    
    def compare_metrics(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare metrics across all models.
        
        Args:
            output_path: Optional path to save the comparison table
            
        Returns:
            DataFrame with metric comparisons
        """
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison_data = {}
        
        for model_name, result in self.results.items():
            metrics = result.get("metrics", {})
            comparison_data[model_name] = {}
            
            for metric in self.metrics:
                if metric in metrics:
                    comparison_data[model_name][metric] = metrics[metric]
                else:
                    comparison_data[model_name][metric] = None
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Add model type column
        model_types = {model: mtype for model, mtype in zip(self.models, self.model_types) 
                       if model in comparison_df.index}
        comparison_df["model_type"] = pd.Series(model_types)
        
        # Reorder columns
        cols = ["model_type"] + self.metrics
        comparison_df = comparison_df[cols]
        
        # Save to CSV if output path is provided
        if output_path:
            comparison_df.to_csv(output_path)
            logger.info(f"Saved comparison table to {output_path}")
        
        return comparison_df
    
    def visualize_metric_comparison(self, metric: str, output_path: Optional[str] = None) -> None:
        """
        Create a bar chart to compare a specific metric across models.
        
        Args:
            metric: Name of the metric to compare
            output_path: Optional path to save the visualization
        """
        if not self.results:
            logger.warning("No results to compare")
            return
        
        comparison_df = self.compare_metrics()
        
        if metric not in comparison_df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Sort by metric value
        sorted_df = comparison_df.sort_values(by=metric, ascending=False)
        
        # Get model types for coloring
        model_types = sorted_df["model_type"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
        color_map = {mtype: color for mtype, color in zip(model_types, colors)}
        
        # Create bar chart
        bars = plt.bar(
            range(len(sorted_df)), 
            sorted_df[metric],
            color=[color_map[mtype] for mtype in sorted_df["model_type"]]
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
        for mtype, color in color_map.items():
            plt.bar(0, 0, color=color, label=mtype)
        
        plt.legend()
        plt.xticks(range(len(sorted_df)), sorted_df.index, rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.title(f"Comparison of {metric.upper()} across models")
        plt.ylim(0, min(1.0, max(sorted_df[metric]) * 1.2))  # Set reasonable y limit
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved {metric} comparison to {output_path}")
        
        plt.close()
    
    def visualize_all_metrics(self, output_dir: str) -> None:
        """
        Create visualizations for all metrics.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual metric comparisons
        for metric in self.metrics:
            if metric in ["brier", "ece"]:  # Lower is better for these metrics
                continue  # Skip for now, we'll create special visualizations for these
            
            self.visualize_metric_comparison(
                metric,
                os.path.join(output_dir, f"comparison_{metric}.png")
            )
        
        # Create combined visualization for precision, recall, F1
        self._visualize_combined_metrics(
            ["precision", "recall", "f1"],
            os.path.join(output_dir, "comparison_combined.png")
        )
    
    def _visualize_combined_metrics(self, metrics: List[str], output_path: Optional[str] = None) -> None:
        """
        Create a grouped bar chart for multiple metrics.
        
        Args:
            metrics: List of metrics to include
            output_path: Optional path to save the visualization
        """
        if not self.results:
            logger.warning("No results to compare")
            return
        
        comparison_df = self.compare_metrics()
        
        # Check if all metrics are available
        for metric in metrics:
            if metric not in comparison_df.columns:
                logger.warning(f"Metric {metric} not found in results")
                return
        
        plt.figure(figsize=(14, 8))
        
        # Sort models by average of the specified metrics
        metric_avg = comparison_df[metrics].mean(axis=1)
        sorted_indices = metric_avg.sort_values(ascending=False).index
        
        # Number of models and metrics
        n_models = len(sorted_indices)
        n_metrics = len(metrics)
        
        # Width of bars
        bar_width = 0.8 / n_metrics
        
        # Create grouped bar chart
        for i, metric in enumerate(metrics):
            positions = np.arange(n_models) + i * bar_width - (n_metrics - 1) * bar_width / 2
            bars = plt.bar(
                positions,
                comparison_df.loc[sorted_indices, metric],
                width=bar_width,
                label=metric.upper()
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=8
                )
        
        plt.legend()
        plt.xticks(np.arange(n_models), sorted_indices, rotation=45, ha="right")
        plt.xlabel("Model")
        plt.title(f"Comparison of {', '.join(m.upper() for m in metrics)} across models")
        plt.ylim(0, 1.1)  # Set y limit from 0 to 1.1 for better visualization
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved combined metrics comparison to {output_path}")
        
        plt.close()


class StatisticalAnalyzer:
    """
    Performs statistical analysis to compare model performances.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the statistical analyzer.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.significance_level = significance_level
        self.model_predictions = {}
        self.ground_truth = []
    
    def add_model_predictions(self, model_name: str, predictions: List[bool], scores: List[float] = None) -> None:
        """
        Add model predictions for statistical comparison.
        
        Args:
            model_name: Name of the model
            predictions: List of predicted hallucinations (True/False)
            scores: Optional list of hallucination scores
        """
        self.model_predictions[model_name] = {
            "predictions": [int(p) for p in predictions],
            "scores": scores if scores is not None else []
        }
    
    def set_ground_truth(self, ground_truth: List[bool]) -> None:
        """
        Set the ground truth labels.
        
        Args:
            ground_truth: List of actual hallucinations (True/False)
        """
        self.ground_truth = [int(gt) for gt in ground_truth]
    
    def compare_f1_scores(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare F1 scores using statistical tests.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.model_predictions or not self.ground_truth:
            logger.warning("No data for statistical comparison")
            return {}
        
        # Compute F1 scores for each model
        f1_scores = {}
        for model_name, data in self.model_predictions.items():
            predictions = data["predictions"]
            
            if len(predictions) != len(self.ground_truth):
                logger.warning(f"Prediction length mismatch for {model_name}")
                continue
            
            f1 = f1_score(self.ground_truth, predictions, zero_division=0)
            f1_scores[model_name] = f1
        
        # Perform pairwise comparisons
        results = {}
        model_names = list(f1_scores.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # Perform McNemar's test for comparing classifiers
                predictions1 = self.model_predictions[model1]["predictions"]
                predictions2 = self.model_predictions[model2]["predictions"]
                
                # Create contingency table
                # [both correct, model1 correct & model2 wrong,
                #  model1 wrong & model2 correct, both wrong]
                contingency = [0, 0, 0, 0]
                
                for p1, p2, gt in zip(predictions1, predictions2, self.ground_truth):
                    if p1 == gt and p2 == gt:
                        contingency[0] += 1
                    elif p1 == gt and p2 != gt:
                        contingency[1] += 1
                    elif p1 != gt and p2 == gt:
                        contingency[2] += 1
                    else:  # p1 != gt and p2 != gt
                        contingency[3] += 1
                
                # McNemar's test
                try:
                    chi2 = (abs(contingency[1] - contingency[2]) - 1)**2 / (contingency[1] + contingency[2])
                    p_value = 1 - stats.chi2.cdf(chi2, df=1)
                    
                    results[f"{model1} vs {model2}"] = {
                        "f1_1": f1_scores[model1],
                        "f1_2": f1_scores[model2],
                        "diff": f1_scores[model1] - f1_scores[model2],
                        "better_model": model1 if f1_scores[model1] > f1_scores[model2] else model2,
                        "chi2": chi2,
                        "p_value": p_value,
                        "significant": p_value < self.significance_level
                    }
                except:
                    logger.warning(f"Failed to compute McNemar's test for {model1} vs {model2}")
        
        return results
    
    def compare_roc_auc(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare ROC AUC scores using statistical tests.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.model_predictions or not self.ground_truth:
            logger.warning("No data for statistical comparison")
            return {}
        
        # Check which models have scores available
        models_with_scores = []
        for model_name, data in self.model_predictions.items():
            if data["scores"] and len(data["scores"]) == len(self.ground_truth):
                models_with_scores.append(model_name)
        
        if len(models_with_scores) < 2:
            logger.warning("Not enough models with scores for AUC comparison")
            return {}
        
        # Compute AUC for each model
        auc_scores = {}
        for model_name in models_with_scores:
            scores = self.model_predictions[model_name]["scores"]
            auc = roc_auc_score(self.ground_truth, scores)
            auc_scores[model_name] = auc
        
        # Perform pairwise comparisons
        # For a proper implementation, we would use DeLong's test for AUC comparison
        # This is a simplified version
        results = {}
        for i, model1 in enumerate(models_with_scores):
            for model2 in models_with_scores[i+1:]:
                results[f"{model1} vs {model2}"] = {
                    "auc_1": auc_scores[model1],
                    "auc_2": auc_scores[model2],
                    "diff": auc_scores[model1] - auc_scores[model2],
                    "better_model": model1 if auc_scores[model1] > auc_scores[model2] else model2,
                }
        
        return results
    
    def save_results(self, output_path: str) -> None:
        """
        Save statistical analysis results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        # Perform analyses
        f1_comparison = self.compare_f1_scores()
        auc_comparison = self.compare_roc_auc()
        
        # Create results dictionary
        results_dict = {
            "f1_comparison": f1_comparison,
            "auc_comparison": auc_comparison,
            "significance_level": self.significance_level,
            "num_samples": len(self.ground_truth)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved statistical analysis results to {output_path}")


# Example usage
if __name__ == "__main__":
    # Test the evaluation module
    print("Testing evaluation module...")
    
    # Create output directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Create a synthetic test scenario
    np.random.seed(42)
    
    num_samples = 1000
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
    print("\nRUNs metrics:", runs_metrics)
    
    evaluator_baseline1 = ModelEvaluator("baseline1", "baseline")
    evaluator_baseline1.add_batch(baseline1_preds, ground_truth, baseline1_scores)
    baseline1_metrics = evaluator_baseline1.compute_metrics()
    print("\nBaseline 1 metrics:", baseline1_metrics)
    
    evaluator_baseline2 = ModelEvaluator("baseline2", "baseline")
    evaluator_baseline2.add_batch(baseline2_preds, ground_truth, baseline2_scores)
    baseline2_metrics = evaluator_baseline2.compute_metrics()
    print("\nBaseline 2 metrics:", baseline2_metrics)
    
    # Create and save visualizations
    evaluator_runs.visualize_calibration(os.path.join(OUTPUTS_DIR, "runs_calibration.png"))
    evaluator_runs.visualize_confusion_matrix(os.path.join(OUTPUTS_DIR, "runs_confusion.png"))
    evaluator_runs.visualize_roc_curve(os.path.join(OUTPUTS_DIR, "runs_roc.png"))
    evaluator_runs.visualize_pr_curve(os.path.join(OUTPUTS_DIR, "runs_pr.png"))
    
    # Save evaluation results
    evaluator_runs.save_results(os.path.join(OUTPUTS_DIR, "runs_results.json"))
    evaluator_baseline1.save_results(os.path.join(OUTPUTS_DIR, "baseline1_results.json"))
    evaluator_baseline2.save_results(os.path.join(OUTPUTS_DIR, "baseline2_results.json"))
    
    # Compare models
    comparison = ComparisonEvaluator(["runs", "baseline1", "baseline2"], ["runs", "baseline", "baseline"])
    comparison.load_results(OUTPUTS_DIR)
    
    # Create comparison visualizations
    comparison.visualize_all_metrics(OUTPUTS_DIR)
    
    # Perform statistical analysis
    analyzer = StatisticalAnalyzer()
    analyzer.set_ground_truth(ground_truth)
    analyzer.add_model_predictions("runs", runs_preds, runs_scores)
    analyzer.add_model_predictions("baseline1", baseline1_preds, baseline1_scores)
    analyzer.add_model_predictions("baseline2", baseline2_preds, baseline2_scores)
    
    f1_comparison = analyzer.compare_f1_scores()
    print("\nF1 comparison:", f1_comparison)
    
    auc_comparison = analyzer.compare_roc_auc()
    print("\nAUC comparison:", auc_comparison)
    
    analyzer.save_results(os.path.join(OUTPUTS_DIR, "statistical_analysis.json"))
    
    print("\nEvaluation module test complete.")