"""
Evaluation metrics for measuring model robustness to spurious correlations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from torch.utils.data import DataLoader
import torch.nn.functional as F


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        
    Returns:
        Classification accuracy
    """
    _, preds = torch.max(outputs, dim=1)
    return (preds == targets).float().mean().item()


def compute_worst_group_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    spurious_channels: Dict,
    modality: str = "multimodal"
) -> Dict:
    """
    Compute accuracy for each group defined by joint spurious attribute assignments.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with return_attributes=True
        device: Device to run the model on
        num_classes: Number of classes
        spurious_channels: Dictionary of spurious channels and their attributes
        modality: Data modality ("image", "text", or "multimodal")
        
    Returns:
        Dictionary containing overall accuracy and worst-group accuracy
    """
    model.eval()
    
    # Initialize counts and correct predictions for each group
    groups = {}
    
    with torch.no_grad():
        for batch in dataloader:
            if modality == "image":
                images, labels, attributes = batch
                inputs = images.to(device)
            elif modality == "text":
                texts, labels, attributes = batch
                inputs = texts.to(device)
            else:  # multimodal
                images, texts, labels, attributes = batch
                inputs = (images.to(device), texts.to(device))
                
            labels = labels.to(device)
            
            # Forward pass
            if modality == "multimodal":
                outputs = model(inputs[0], inputs[1])
            else:
                outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Process each example in the batch
            for i in range(labels.size(0)):
                # Create group key based on class and spurious attributes
                class_idx = labels[i].item()
                
                # Create tuple of attribute indices for each channel
                attr_tuple = tuple(attr[i].item() for attr in attributes.values())
                
                # Create group key
                group_key = (class_idx, attr_tuple)
                
                # Initialize group stats if needed
                if group_key not in groups:
                    groups[group_key] = {"count": 0, "correct": 0}
                    
                # Update group stats
                groups[group_key]["count"] += 1
                groups[group_key]["correct"] += (preds[i] == labels[i]).item()
    
    # Compute accuracy for each group
    group_accuracies = {}
    for group_key, stats in groups.items():
        if stats["count"] > 0:
            group_accuracies[group_key] = stats["correct"] / stats["count"]
    
    # Compute overall and worst-group accuracy
    overall_correct = sum(stats["correct"] for stats in groups.values())
    overall_count = sum(stats["count"] for stats in groups.values())
    overall_accuracy = overall_correct / overall_count if overall_count > 0 else 0
    
    worst_group_accuracy = min(group_accuracies.values()) if group_accuracies else 0
    worst_group = min(group_accuracies, key=group_accuracies.get) if group_accuracies else None
    
    return {
        "overall_accuracy": overall_accuracy,
        "worst_group_accuracy": worst_group_accuracy,
        "worst_group": worst_group,
        "group_accuracies": group_accuracies
    }


def compute_spurious_sensitivity_score(
    model: torch.nn.Module,
    shuffled_loader: DataLoader,
    device: torch.device,
    modality: str = "multimodal"
) -> float:
    """
    Compute Spurious Sensitivity Score (SSS) for a model.
    
    SSS measures the expected absolute difference in predicted probability
    when a spurious attribute is shuffled.
    
    Args:
        model: Model to evaluate
        shuffled_loader: DataLoader with original and shuffled samples
        device: Device to run the model on
        modality: Data modality ("image", "text", or "multimodal")
        
    Returns:
        Spurious Sensitivity Score
    """
    model.eval()
    
    total_diff = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in shuffled_loader:
            if modality == "image":
                orig_images, shuffled_images, labels = batch
                orig_inputs = orig_images.to(device)
                shuffled_inputs = shuffled_images.to(device)
            elif modality == "text":
                orig_texts, shuffled_texts, labels = batch
                orig_inputs = orig_texts.to(device)
                shuffled_inputs = shuffled_texts.to(device)
            else:  # multimodal
                orig_images, orig_texts, shuffled_images, shuffled_texts, labels = batch
                orig_inputs = (orig_images.to(device), orig_texts.to(device))
                shuffled_inputs = (shuffled_images.to(device), shuffled_texts.to(device))
                
            labels = labels.to(device)
            
            # Forward pass for original inputs
            if modality == "multimodal":
                orig_outputs = model(orig_inputs[0], orig_inputs[1])
            else:
                orig_outputs = model(orig_inputs)
                
            # Forward pass for shuffled inputs
            if modality == "multimodal":
                shuffled_outputs = model(shuffled_inputs[0], shuffled_inputs[1])
            else:
                shuffled_outputs = model(shuffled_inputs)
                
            # Apply softmax to get probabilities
            orig_probs = F.softmax(orig_outputs, dim=1)
            shuffled_probs = F.softmax(shuffled_outputs, dim=1)
            
            # Get probability for true class
            batch_size = labels.size(0)
            true_class_orig_probs = orig_probs[torch.arange(batch_size), labels]
            true_class_shuffled_probs = shuffled_probs[torch.arange(batch_size), labels]
            
            # Compute absolute difference
            diffs = torch.abs(true_class_orig_probs - true_class_shuffled_probs)
            
            # Update running average
            total_diff += diffs.sum().item()
            count += batch_size
    
    # Compute final SSS
    sss = total_diff / count if count > 0 else 0
    
    return sss


def compute_invariance_gap(
    model: torch.nn.Module,
    ctrl_loader: DataLoader,
    uncontrolled_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    modality: str = "multimodal"
) -> float:
    """
    Compute Invariance Gap (IG) for a model.
    
    IG measures the difference in loss between controlled and uncontrolled spurious settings.
    
    Args:
        model: Model to evaluate
        ctrl_loader: DataLoader with controlled spurious attributes
        uncontrolled_loader: DataLoader with uncontrolled (random) spurious attributes
        device: Device to run the model on
        criterion: Loss function
        modality: Data modality ("image", "text", or "multimodal")
        
    Returns:
        Invariance Gap
    """
    model.eval()
    
    # Compute loss for controlled setting
    ctrl_loss_total = 0.0
    ctrl_count = 0
    
    with torch.no_grad():
        for batch in ctrl_loader:
            if modality == "image":
                images, labels = batch
                inputs = images.to(device)
            elif modality == "text":
                texts, labels = batch
                inputs = texts.to(device)
            else:  # multimodal
                images, texts, labels = batch
                inputs = (images.to(device), texts.to(device))
                
            labels = labels.to(device)
            
            # Forward pass
            if modality == "multimodal":
                outputs = model(inputs[0], inputs[1])
            else:
                outputs = model(inputs)
                
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Update running average
            ctrl_loss_total += loss.item() * labels.size(0)
            ctrl_count += labels.size(0)
    
    # Compute loss for uncontrolled setting
    unc_loss_total = 0.0
    unc_count = 0
    
    with torch.no_grad():
        for batch in uncontrolled_loader:
            if modality == "image":
                # Shuffled loader returns original, shuffled, and labels
                orig_images, shuffled_images, labels = batch
                inputs = shuffled_images.to(device)  # Use the shuffled version
            elif modality == "text":
                orig_texts, shuffled_texts, labels = batch
                inputs = shuffled_texts.to(device)  # Use the shuffled version
            else:  # multimodal
                orig_images, orig_texts, shuffled_images, shuffled_texts, labels = batch
                inputs = (shuffled_images.to(device), shuffled_texts.to(device))  # Use the shuffled version

            labels = labels.to(device)

            # Forward pass
            if modality == "multimodal":
                outputs = model(inputs[0], inputs[1])
            else:
                outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Update running average
            unc_loss_total += loss.item() * labels.size(0)
            unc_count += labels.size(0)
    
    # Compute final Invariance Gap
    ctrl_loss = ctrl_loss_total / ctrl_count if ctrl_count > 0 else 0
    unc_loss = unc_loss_total / unc_count if unc_count > 0 else 0
    
    ig = ctrl_loss - unc_loss
    
    return ig