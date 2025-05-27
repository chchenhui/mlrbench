"""
Performance metrics module

This module implements standard performance metrics for classification, regression,
and other machine learning tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Any, Union, Optional, Tuple, Callable


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate standard classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction probabilities or scores (optional)
        average: Method for averaging in multiclass settings ('micro', 'macro', 'weighted')
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    # Add AUC-ROC if scores are provided
    if y_score is not None:
        # For binary classification
        if len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_score))
            except Exception:
                # Handle cases where AUC is not well-defined
                metrics['auc_roc'] = float('nan')
        else:
            # For multiclass, calculate one-vs-rest ROC AUC
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_score, average=average, multi_class='ovr'))
            except Exception:
                metrics['auc_roc'] = float('nan')
    
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['r2'] = float(r2_score(y_true, y_pred))
    
    return metrics


def domain_specific_metrics(
    domain: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate domain-specific performance metrics.
    
    Args:
        domain: Domain of the task (e.g., "healthcare", "finance")
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values
        y_score: Prediction probabilities or scores (optional)
        
    Returns:
        dict: Dictionary of domain-specific metrics
    """
    metrics = {}
    
    if domain == "healthcare":
        # For healthcare, we might prioritize sensitivity (recall) and specificity
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specificity (true negative rate)
        if cm.shape[0] == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # Calculate balanced accuracy
            metrics['balanced_accuracy'] = float((metrics['recall'] + metrics['specificity']) / 2)
            
            # Calculate positive predictive value (PPV) and negative predictive value (NPV)
            metrics['ppv'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    elif domain == "finance":
        # For finance, we might track costs of false positives vs false negatives
        # For example, in fraud detection, missing a fraud (false negative) might be costlier
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            
            # Custom cost metric (example: FN costs 5x more than FP)
            fn_cost_multiplier = 5.0
            fp_cost = 1.0
            fn_cost = fn_cost_multiplier * fp_cost
            
            total_cost = (fp * fp_cost) + (fn * fn_cost)
            metrics['cost_metric'] = float(total_cost)
            
            # Profit metric (if available)
            # This would require additional information on profit/loss from TP, TN, FP, FN
    
    return metrics


def calculate_performance_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    domain: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate performance metrics based on task type and domain.
    
    Args:
        task_type: Type of task ('classification' or 'regression')
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values
        y_score: Prediction probabilities or scores (optional)
        domain: Domain of the task (optional)
        **kwargs: Additional arguments
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics based on task type
    if task_type == 'classification':
        metrics.update(classification_metrics(y_true, y_pred, y_score, **kwargs))
    elif task_type == 'regression':
        metrics.update(regression_metrics(y_true, y_pred))
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Add domain-specific metrics if domain is provided
    if domain is not None:
        domain_metrics = domain_specific_metrics(domain, y_true, y_pred, y_score)
        metrics.update(domain_metrics)
    
    return metrics