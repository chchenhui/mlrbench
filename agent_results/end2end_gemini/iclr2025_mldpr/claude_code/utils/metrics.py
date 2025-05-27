"""
Evaluation metrics for the AEB project.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def calculate_accuracy(outputs, targets):
    """Calculate accuracy from model outputs and targets."""
    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0) * 100
    return accuracy

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation dataset
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        Tuple of (loss, accuracy, predictions, ground_truth)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate loss and accuracy
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Store predictions and targets for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)

def calculate_detailed_metrics(predictions, ground_truth, num_classes=10):
    """
    Calculate detailed classification metrics.
    
    Args:
        predictions: Numpy array of predictions
        ground_truth: Numpy array of ground truth labels
        num_classes: Number of classes
    
    Returns:
        Dictionary containing various metrics
    """
    # Convert inputs to numpy arrays if they are tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # Class labels
    labels = range(num_classes)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'precision_macro': precision_score(ground_truth, predictions, average='macro', labels=labels, zero_division=0),
        'recall_macro': recall_score(ground_truth, predictions, average='macro', labels=labels, zero_division=0),
        'f1_macro': f1_score(ground_truth, predictions, average='macro', labels=labels, zero_division=0),
        'precision_weighted': precision_score(ground_truth, predictions, average='weighted', labels=labels, zero_division=0),
        'recall_weighted': recall_score(ground_truth, predictions, average='weighted', labels=labels, zero_division=0),
        'f1_weighted': f1_score(ground_truth, predictions, average='weighted', labels=labels, zero_division=0),
        'confusion_matrix': confusion_matrix(ground_truth, predictions, labels=labels)
    }
    
    # Calculate per-class metrics
    per_class_precision = precision_score(ground_truth, predictions, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(ground_truth, predictions, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(ground_truth, predictions, average=None, labels=labels, zero_division=0)
    
    for i, label in enumerate(labels):
        metrics[f'precision_class_{label}'] = per_class_precision[i]
        metrics[f'recall_class_{label}'] = per_class_recall[i]
        metrics[f'f1_class_{label}'] = per_class_f1[i]
    
    return metrics

def calculate_robustness_metrics(standard_metrics, adversarial_metrics):
    """
    Calculate robustness metrics comparing standard and adversarial performance.
    
    Args:
        standard_metrics: Metrics on standard data
        adversarial_metrics: Metrics on adversarially transformed data
    
    Returns:
        Dictionary of robustness metrics
    """
    robustness_metrics = {}
    
    # Accuracy degradation
    robustness_metrics['accuracy_degradation'] = standard_metrics['accuracy'] - adversarial_metrics['accuracy']
    robustness_metrics['accuracy_degradation_percentage'] = (
        (standard_metrics['accuracy'] - adversarial_metrics['accuracy']) / standard_metrics['accuracy'] * 100
    ) if standard_metrics['accuracy'] > 0 else float('inf')
    
    # F1 score degradation
    robustness_metrics['f1_degradation'] = standard_metrics['f1_weighted'] - adversarial_metrics['f1_weighted']
    robustness_metrics['f1_degradation_percentage'] = (
        (standard_metrics['f1_weighted'] - adversarial_metrics['f1_weighted']) / standard_metrics['f1_weighted'] * 100
    ) if standard_metrics['f1_weighted'] > 0 else float('inf')
    
    # Per-class robustness
    robustness_metrics['per_class_degradation'] = {}
    
    # Find common metrics that are per-class
    per_class_keys = [key for key in standard_metrics.keys() if key.startswith('f1_class_')]
    
    for key in per_class_keys:
        degradation = standard_metrics[key] - adversarial_metrics[key]
        degradation_percentage = (
            (standard_metrics[key] - adversarial_metrics[key]) / standard_metrics[key] * 100
        ) if standard_metrics[key] > 0 else float('inf')
        
        robustness_metrics['per_class_degradation'][key] = {
            'absolute': degradation,
            'percentage': degradation_percentage
        }
    
    # Calculate weighted robustness score (lower is better)
    # This is a weighted combination of accuracy and F1 degradation
    robustness_metrics['robustness_score'] = 0.7 * robustness_metrics['accuracy_degradation_percentage'] + \
                                             0.3 * robustness_metrics['f1_degradation_percentage']
    
    return robustness_metrics

def comparative_analysis(model_performances):
    """
    Perform comparative analysis of different models.
    
    Args:
        model_performances: Dictionary with model names as keys and dictionaries of metrics as values
    
    Returns:
        Dictionary with comparative metrics and rankings
    """
    comparison = {
        'accuracy': {},
        'robustness': {},
        'rankings': {}
    }
    
    # Extract metrics for comparison
    for model_name, metrics in model_performances.items():
        comparison['accuracy'][model_name] = metrics['standard']['accuracy']
        
        # Use robustness score if available, otherwise set a high value (poor robustness)
        if 'robustness' in metrics:
            comparison['robustness'][model_name] = metrics['robustness']['robustness_score']
        else:
            comparison['robustness'][model_name] = float('inf')
    
    # Rank models by accuracy (higher is better)
    accuracy_ranking = sorted(comparison['accuracy'].items(), key=lambda x: x[1], reverse=True)
    comparison['rankings']['accuracy'] = {model: rank+1 for rank, (model, _) in enumerate(accuracy_ranking)}
    
    # Rank models by robustness (lower robustness score is better)
    robustness_ranking = sorted(comparison['robustness'].items(), key=lambda x: x[1])
    comparison['rankings']['robustness'] = {model: rank+1 for rank, (model, _) in enumerate(robustness_ranking)}
    
    # Calculate overall ranking (weighted sum of ranks)
    # Lower rank is better (1st place = 1)
    overall_scores = {}
    for model in model_performances.keys():
        # Weight accuracy and robustness equally
        acc_rank = comparison['rankings']['accuracy'][model]
        rob_rank = comparison['rankings']['robustness'][model]
        overall_scores[model] = 0.5 * acc_rank + 0.5 * rob_rank
    
    # Sort by overall score (lower is better)
    overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
    comparison['rankings']['overall'] = {model: rank+1 for rank, (model, _) in enumerate(overall_ranking)}
    comparison['overall_scores'] = overall_scores
    
    return comparison