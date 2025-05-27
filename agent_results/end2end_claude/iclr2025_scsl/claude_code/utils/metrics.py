#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics computation utilities for CIMRL experiments.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def compute_metrics(targets, preds, probs=None):
    """
    Compute classification metrics.
    
    Args:
        targets: Ground truth labels
        preds: Model predictions
        probs: Prediction probabilities (optional)
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, preds)
    
    # Precision, recall, F1 score (macro)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # AUC-ROC (if probabilities are provided)
    if probs is not None:
        try:
            # For binary classification
            if probs.shape[1] == 2:
                metrics['auc'] = roc_auc_score(targets, probs[:, 1])
                metrics['average_auc'] = metrics['auc']
            # For multi-class classification
            else:
                n_classes = probs.shape[1]
                metrics['auc'] = {}
                auc_sum = 0
                
                for i in range(n_classes):
                    # One-vs-rest ROC AUC
                    class_targets = (targets == i).astype(int)
                    class_probs = probs[:, i]
                    
                    # Skip if only one class is present
                    if len(np.unique(class_targets)) == 1:
                        metrics['auc'][f'class_{i}'] = np.nan
                    else:
                        metrics['auc'][f'class_{i}'] = roc_auc_score(class_targets, class_probs)
                        auc_sum += metrics['auc'][f'class_{i}']
                
                metrics['average_auc'] = auc_sum / n_classes
        except Exception as e:
            # Fallback if AUC calculation fails
            metrics['auc'] = np.nan
            metrics['average_auc'] = np.nan
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(targets, preds).tolist()
    
    return metrics


def compute_group_metrics(targets, preds, group_labels):
    """
    Compute metrics for each group.
    
    Args:
        targets: Ground truth labels
        preds: Model predictions
        group_labels: Group labels for each sample
        
    Returns:
        group_metrics: Dictionary of metrics for each group
    """
    unique_groups = np.unique(group_labels)
    group_metrics = {}
    
    for group in unique_groups:
        # Get indices for this group
        group_idx = (group_labels == group)
        
        # Skip if no samples in this group
        if np.sum(group_idx) == 0:
            continue
        
        # Get targets and predictions for this group
        group_targets = targets[group_idx]
        group_preds = preds[group_idx]
        
        # Compute metrics
        group_metrics[f'group_{group}'] = {}
        group_metrics[f'group_{group}']['size'] = int(np.sum(group_idx))
        group_metrics[f'group_{group}']['accuracy'] = accuracy_score(group_targets, group_preds)
        
        # Precision, recall, F1 score
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(group_targets, group_preds, average='macro')
            group_metrics[f'group_{group}']['precision'] = precision
            group_metrics[f'group_{group}']['recall'] = recall
            group_metrics[f'group_{group}']['f1'] = f1
        except:
            # Fallback if computation fails (e.g., one class)
            group_metrics[f'group_{group}']['precision'] = np.nan
            group_metrics[f'group_{group}']['recall'] = np.nan
            group_metrics[f'group_{group}']['f1'] = np.nan
    
    return group_metrics


def compute_worst_group_metrics(targets, preds, group_labels):
    """
    Compute worst-group performance metrics.
    
    Args:
        targets: Ground truth labels
        preds: Model predictions
        group_labels: Group labels for each sample
        
    Returns:
        worst_group_metrics: Dictionary of worst-group metrics
    """
    group_metrics = compute_group_metrics(targets, preds, group_labels)
    worst_group_metrics = {}
    
    # Find worst group accuracy
    group_accuracies = [metrics['accuracy'] for group, metrics in group_metrics.items()]
    worst_group_metrics['worst_group_accuracy'] = min(group_accuracies) if group_accuracies else np.nan
    
    # Average group accuracy
    worst_group_metrics['average_group_accuracy'] = np.mean(group_accuracies) if group_accuracies else np.nan
    
    # Group accuracy gap (between best and worst)
    if group_accuracies:
        worst_group_metrics['group_accuracy_gap'] = max(group_accuracies) - min(group_accuracies)
    else:
        worst_group_metrics['group_accuracy_gap'] = np.nan
    
    return worst_group_metrics


def compute_feature_importance(model, dataloader, device):
    """
    Compute feature importance based on gradient attribution.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to use
        
    Returns:
        feature_importance: Dictionary of feature importance scores
    """
    # This is a placeholder for actual implementation
    # Would need to be customized based on the specific model architecture
    feature_importance = {
        'vision': {},
        'text': {}
    }
    
    return feature_importance