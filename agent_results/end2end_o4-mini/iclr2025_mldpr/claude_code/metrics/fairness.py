"""
Fairness metrics module

This module implements fairness metrics for evaluating machine learning models
across sensitive demographic groups. It focuses on group fairness metrics
like demographic parity, equalized odds, and equal opportunity.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import confusion_matrix, accuracy_score


def demographic_parity_difference(
    y_pred: np.ndarray,
    group_membership: np.ndarray
) -> float:
    """
    Calculate the maximum demographic parity difference across groups.
    
    Demographic parity requires that prediction rates are equal across groups.
    
    Args:
        y_pred: Predicted labels (binary)
        group_membership: Group membership for each sample
        
    Returns:
        float: Maximum difference in prediction rates between any two groups
    """
    unique_groups = np.unique(group_membership)
    
    if len(unique_groups) <= 1:
        return 0.0
    
    # Calculate prediction rate for each group
    group_pred_rates = []
    for group in unique_groups:
        group_mask = (group_membership == group)
        if np.sum(group_mask) == 0:
            continue
        pred_rate = np.mean(y_pred[group_mask])
        group_pred_rates.append(pred_rate)
    
    # Calculate maximum difference
    max_diff = max(group_pred_rates) - min(group_pred_rates)
    return float(max_diff)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_membership: np.ndarray
) -> Dict[str, float]:
    """
    Calculate the equalized odds differences across groups.
    
    Equalized odds requires equal true positive rates and false positive rates across groups.
    
    Args:
        y_true: Ground truth labels (binary)
        y_pred: Predicted labels (binary)
        group_membership: Group membership for each sample
        
    Returns:
        dict: Maximum difference in TPR and FPR between any two groups
    """
    unique_groups = np.unique(group_membership)
    
    if len(unique_groups) <= 1:
        return {"max_tpr_diff": 0.0, "max_fpr_diff": 0.0}
    
    # Calculate TPR and FPR for each group
    group_tpr = []
    group_fpr = []
    
    for group in unique_groups:
        group_mask = (group_membership == group)
        if np.sum(group_mask) == 0:
            continue
            
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred[group_mask]
        
        # Skip groups with no positive or negative samples
        if np.sum(y_true_group) == 0 or np.sum(y_true_group) == len(y_true_group):
            continue
            
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        group_tpr.append(tpr)
        group_fpr.append(fpr)
    
    # Calculate maximum differences
    max_tpr_diff = max(group_tpr) - min(group_tpr) if group_tpr else 0.0
    max_fpr_diff = max(group_fpr) - min(group_fpr) if group_fpr else 0.0
    
    return {
        "max_tpr_diff": float(max_tpr_diff),
        "max_fpr_diff": float(max_fpr_diff)
    }


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_membership: np.ndarray
) -> float:
    """
    Calculate the equal opportunity difference across groups.
    
    Equal opportunity requires equal true positive rates across groups.
    
    Args:
        y_true: Ground truth labels (binary)
        y_pred: Predicted labels (binary)
        group_membership: Group membership for each sample
        
    Returns:
        float: Maximum difference in TPR between any two groups
    """
    # The TPR difference is already calculated in equalized_odds_difference
    return equalized_odds_difference(y_true, y_pred, group_membership)["max_tpr_diff"]


def disparate_impact(
    y_pred: np.ndarray,
    group_membership: np.ndarray,
    favorable_outcome: int = 1
) -> float:
    """
    Calculate the disparate impact ratio.
    
    Disparate impact is the ratio of the probability of favorable outcome for the unprivileged group
    to the probability of favorable outcome for the privileged group.
    
    Args:
        y_pred: Predicted labels
        group_membership: Group membership for each sample
        favorable_outcome: Label considered as favorable outcome (default: 1)
        
    Returns:
        float: Disparate impact ratio (values closer to 1 indicate more fairness)
    """
    unique_groups = np.unique(group_membership)
    
    if len(unique_groups) <= 1:
        return 1.0
    
    # Calculate prediction rate for each group
    group_pred_rates = {}
    for group in unique_groups:
        group_mask = (group_membership == group)
        if np.sum(group_mask) == 0:
            continue
        pred_rate = np.mean(y_pred[group_mask] == favorable_outcome)
        group_pred_rates[group] = pred_rate
    
    # Find group with highest prediction rate (privileged group)
    # and group with lowest prediction rate (unprivileged group)
    if not group_pred_rates:
        return 1.0
        
    privileged_group = max(group_pred_rates, key=group_pred_rates.get)
    unprivileged_group = min(group_pred_rates, key=group_pred_rates.get)
    
    # Calculate disparate impact
    privileged_rate = group_pred_rates[privileged_group]
    unprivileged_rate = group_pred_rates[unprivileged_group]
    
    # Avoid division by zero
    if privileged_rate == 0:
        return 1.0 if unprivileged_rate == 0 else float('inf')
    
    di = unprivileged_rate / privileged_rate
    return float(di)


def group_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: Dict[str, np.ndarray],
    task_type: str = 'classification'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate various group fairness metrics for each sensitive feature.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        sensitive_features: Dictionary mapping feature names to group membership arrays
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        dict: Dictionary of fairness metrics for each sensitive feature
    """
    if task_type != 'classification':
        raise ValueError("Group fairness metrics are currently only supported for classification tasks")
    
    # Convert predictions to binary if needed
    y_true_binary = (y_true > 0).astype(int) if not np.array_equal(y_true, y_true.astype(bool)) else y_true
    y_pred_binary = (y_pred > 0).astype(int) if not np.array_equal(y_pred, y_pred.astype(bool)) else y_pred
    
    fairness_metrics = {}
    
    for feature_name, group_membership in sensitive_features.items():
        metrics = {}
        
        # Demographic parity
        metrics['demographic_parity_diff'] = demographic_parity_difference(y_pred_binary, group_membership)
        
        # Equalized odds
        eq_odds = equalized_odds_difference(y_true_binary, y_pred_binary, group_membership)
        metrics['equalized_odds_tpr_diff'] = eq_odds['max_tpr_diff']
        metrics['equalized_odds_fpr_diff'] = eq_odds['max_fpr_diff']
        
        # Equal opportunity
        metrics['equal_opportunity_diff'] = equal_opportunity_difference(
            y_true_binary, y_pred_binary, group_membership
        )
        
        # Disparate impact
        metrics['disparate_impact'] = disparate_impact(y_pred_binary, group_membership)
        
        # Group-specific accuracy
        unique_groups = np.unique(group_membership)
        accuracy_disparity = []
        
        for group in unique_groups:
            group_mask = (group_membership == group)
            if np.sum(group_mask) == 0:
                continue
                
            group_acc = accuracy_score(y_true[group_mask], y_pred[group_mask])
            accuracy_disparity.append(group_acc)
        
        # Max accuracy difference between groups
        metrics['max_accuracy_diff'] = max(accuracy_disparity) - min(accuracy_disparity) if accuracy_disparity else 0.0
        
        fairness_metrics[feature_name] = metrics
    
    return fairness_metrics


def calculate_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: Dict[str, Any],
    task_type: str = 'classification',
    feature_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate fairness metrics based on the provided metadata and feature data.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metadata: Dataset metadata containing demographic information
        task_type: Type of task ('classification' or 'regression')
        feature_data: Dictionary of feature data for sensitive attributes
        
    Returns:
        dict: Dictionary of fairness metrics
    """
    if feature_data is None or len(feature_data) == 0:
        return {"overall": {"no_sensitive_features": 0.0}}
    
    sensitive_features = {}
    
    # Extract sensitive features from the feature data
    for feature_name, feature_array in feature_data.items():
        # Check if this feature is marked as sensitive in metadata
        is_sensitive = False
        
        if 'features' in metadata and feature_name in metadata['features']:
            is_sensitive = metadata['features'][feature_name].get('sensitive', False)
        
        if is_sensitive:
            sensitive_features[feature_name] = feature_array
    
    # If no sensitive features were found, return empty result
    if not sensitive_features:
        return {"overall": {"no_sensitive_features": 0.0}}
    
    # Calculate fairness metrics for each sensitive feature
    return group_fairness_metrics(y_true, y_pred, sensitive_features, task_type)