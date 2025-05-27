"""
Evaluation utilities for Neural Weight Archeology experiments.

This module provides functions for evaluating models and computing metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from typing import Dict, List, Tuple, Any

def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: str) -> Dict:
    """
    Evaluate a model on a dataset
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to use for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Store predictions and targets
    all_predictions = {
        'classification': {},
        'regression': []
    }
    
    all_targets = {
        'classification': {},
        'regression': []
    }
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Collect classification predictions
            if 'classification' in outputs:
                for class_name, logits in outputs['classification'].items():
                    # Initialize target and prediction lists if not already done
                    if class_name not in all_predictions['classification']:
                        all_predictions['classification'][class_name] = []
                        all_targets['classification'][class_name] = []
                    
                    # Get predictions
                    predictions = torch.argmax(logits, dim=1).cpu().numpy()
                    targets = batch[f'class_{class_name}'].cpu().numpy()
                    
                    all_predictions['classification'][class_name].extend(predictions)
                    all_targets['classification'][class_name].extend(targets)
            
            # Collect regression predictions
            if 'regression' in outputs:
                predictions = outputs['regression'].cpu().numpy()
                targets = batch['regression_targets'].cpu().numpy()
                
                all_predictions['regression'].extend(predictions)
                all_targets['regression'].extend(targets)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    return metrics

def compute_metrics(predictions: Dict, targets: Dict) -> Dict:
    """
    Compute evaluation metrics for classification and regression
    
    Args:
        predictions: Dictionary of predictions
        targets: Dictionary of targets
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Classification metrics
    if predictions['classification'] and targets['classification']:
        classification_metrics = {}
        
        for class_name in predictions['classification']:
            class_predictions = np.array(predictions['classification'][class_name])
            class_targets = np.array(targets['classification'][class_name])
            
            # Compute metrics
            accuracy = accuracy_score(class_targets, class_predictions)
            
            # For multi-class, use weighted average
            precision = precision_score(class_targets, class_predictions, average='weighted', zero_division=0)
            recall = recall_score(class_targets, class_predictions, average='weighted', zero_division=0)
            f1 = f1_score(class_targets, class_predictions, average='weighted', zero_division=0)
            
            classification_metrics[class_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
            }
        
        # Compute average metrics across all classes
        avg_accuracy = np.mean([m['accuracy'] for m in classification_metrics.values()])
        avg_precision = np.mean([m['precision'] for m in classification_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in classification_metrics.values()])
        avg_f1 = np.mean([m['f1_score'] for m in classification_metrics.values()])
        
        classification_metrics['average'] = {
            'accuracy': float(avg_accuracy),
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1_score': float(avg_f1),
        }
        
        metrics['classification'] = classification_metrics
    
    # Regression metrics
    if predictions['regression'] and targets['regression']:
        pred_array = np.array(predictions['regression'])
        target_array = np.array(targets['regression'])
        
        # If there's a single regression target, add a dimension for consistency
        if len(pred_array.shape) == 1:
            pred_array = pred_array.reshape(-1, 1)
            target_array = target_array.reshape(-1, 1)
        
        # Compute metrics for each target dimension
        r2_values = []
        mse_values = []
        mae_values = []
        
        for i in range(pred_array.shape[1]):
            r2 = r2_score(target_array[:, i], pred_array[:, i])
            mse = mean_squared_error(target_array[:, i], pred_array[:, i])
            mae = mean_absolute_error(target_array[:, i], pred_array[:, i])
            
            r2_values.append(float(r2))
            mse_values.append(float(mse))
            mae_values.append(float(mae))
        
        # Compute overall metrics
        overall_r2 = np.mean(r2_values)
        overall_mse = np.mean(mse_values)
        overall_mae = np.mean(mae_values)
        
        metrics['regression'] = {
            'r2_score': float(overall_r2),
            'mse': float(overall_mse),
            'mae': float(overall_mae),
            'per_target': {
                'r2_score': r2_values,
                'mse': mse_values,
                'mae': mae_values,
            }
        }
    
    return metrics