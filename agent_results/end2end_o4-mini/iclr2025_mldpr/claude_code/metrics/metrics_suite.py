"""
Multi-Metric Evaluation Suite (MES) Aggregator

This module provides the main interface for the Multi-Metric Evaluation Suite (MES),
combining all metrics into a unified evaluation report.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import time
import json
import os
import warnings

from .performance import calculate_performance_metrics
from .fairness import calculate_fairness_metrics
from .robustness import calculate_robustness_metrics
from .environmental_impact import (
    ResourceMonitor, 
    calculate_environmental_impact
)
from .interpretability import calculate_interpretability_metrics


class MultiMetricEvaluationSuite:
    """
    Main class for the Multi-Metric Evaluation Suite (MES).
    """
    
    def __init__(self, metadata: Dict[str, Any], task_type: str = 'classification'):
        """
        Initialize the Multi-Metric Evaluation Suite.
        
        Args:
            metadata: Dataset metadata
            task_type: Type of task ('classification' or 'regression')
        """
        self.metadata = metadata
        self.task_type = task_type
        self.resource_monitor = ResourceMonitor(gpu_available=self._check_gpu_available())
    
    def _check_gpu_available(self) -> bool:
        """
        Check if GPU is available.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start_monitoring(self):
        """
        Start monitoring resources.
        """
        self.resource_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """
        Stop monitoring resources.
        
        Returns:
            dict: Resource usage metrics
        """
        return self.resource_monitor.stop_monitoring()
    
    def evaluate(
        self,
        model: Any,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        feature_data: Optional[Dict[str, np.ndarray]] = None,
        x_shifted: Optional[np.ndarray] = None,
        y_shifted: Optional[np.ndarray] = None,
        attribution_func: Optional[Callable] = None,
        perturbation_func: Optional[Callable] = None,
        attack_function: Optional[Callable] = None,
        model_size_mb: Optional[float] = None,
        gpu_power_draw: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a model across multiple metrics.
        
        Args:
            model: ML model
            x_test: Test data
            y_test: True labels
            y_pred: Predicted labels (optional, will be computed if not provided)
            y_score: Prediction scores/probabilities (optional)
            feature_data: Dictionary mapping feature names to feature values (optional)
            x_shifted: Shifted test data (optional)
            y_shifted: Shifted labels (optional)
            attribution_func: Function for feature attribution (optional)
            perturbation_func: Function for input perturbation (optional)
            attack_function: Function for adversarial attacks (optional)
            model_size_mb: Model size in MB (optional)
            gpu_power_draw: Estimated GPU power draw in watts (optional)
            gpu_utilization: Estimated GPU utilization (0-1) (optional)
            domain: Domain of the task (optional)
            
        Returns:
            dict: Dictionary of metrics across all dimensions
        """
        results = {}
        
        # Get predictions if not provided
        if y_pred is None:
            try:
                y_pred = model.predict(x_test)
            except Exception as e:
                warnings.warn(f"Failed to get predictions: {str(e)}")
                return {"error": str(e)}
        
        # Get prediction scores if not provided and model has predict_proba
        if y_score is None and hasattr(model, 'predict_proba'):
            try:
                y_score = model.predict_proba(x_test)
                # For binary classification, use the positive class probability
                if y_score.shape[1] == 2:
                    y_score = y_score[:, 1]
            except Exception:
                pass
        
        # 1. Performance metrics
        try:
            performance_metrics = calculate_performance_metrics(
                self.task_type, y_test, y_pred, y_score, domain
            )
            results['performance'] = performance_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate performance metrics: {str(e)}")
            results['performance'] = {"error": str(e)}
        
        # 2. Fairness metrics
        if feature_data is not None:
            try:
                fairness_metrics = calculate_fairness_metrics(
                    y_test, y_pred, self.metadata, self.task_type, feature_data
                )
                results['fairness'] = fairness_metrics
            except Exception as e:
                warnings.warn(f"Failed to calculate fairness metrics: {str(e)}")
                results['fairness'] = {"error": str(e)}
        else:
            results['fairness'] = {"no_feature_data": True}
        
        # 3. Robustness metrics
        try:
            robustness_metrics = calculate_robustness_metrics(
                model, x_test, y_test, x_shifted, y_shifted, attack_function, domain=domain
            )
            results['robustness'] = robustness_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate robustness metrics: {str(e)}")
            results['robustness'] = {"error": str(e)}
        
        # 4. Environmental impact metrics
        try:
            monitor_results = self.stop_monitoring()
            environmental_metrics = calculate_environmental_impact(
                monitor_results, len(x_test), model_size_mb, gpu_power_draw, gpu_utilization
            )
            results['environmental_impact'] = environmental_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate environmental impact metrics: {str(e)}")
            results['environmental_impact'] = {"error": str(e)}
        
        # 5. Interpretability metrics
        try:
            # Get feature importances if model has them
            feature_importances = None
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            
            interpretability_metrics = calculate_interpretability_metrics(
                model, x_test, attribution_func, perturbation_func, feature_importances
            )
            results['interpretability'] = interpretability_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate interpretability metrics: {str(e)}")
            results['interpretability'] = {"error": str(e)}
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            context: Context information (optional)
            
        Returns:
            dict: Evaluation report
        """
        report = {
            "dataset": self.metadata.get("dataset_id", "unknown"),
            "task_type": self.task_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        if context is not None:
            report["context"] = context
        
        return report
    
    def save_report(self, report: Dict[str, Any], directory: str, filename: Optional[str] = None) -> str:
        """
        Save the evaluation report to a JSON file.
        
        Args:
            report: Evaluation report
            directory: Directory to save the report
            filename: Filename for the report (optional)
            
        Returns:
            str: Path to the saved report
        """
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            dataset_id = self.metadata.get("dataset_id", "unknown")
            timestamp = time.strftime("%Y%m%d%H%M%S")
            filename = f"{dataset_id}_{timestamp}_report.json"
        
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath