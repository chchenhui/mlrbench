"""
Training and evaluation functions for the LASS framework.
"""

import os
import time
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from tqdm import tqdm

from models import BaseModel, ImageClassifier, TextClassifier, GroupDRO, ULEModel, LLMAugmentedModel
from utils import plot_learning_curves, plot_confusion_matrix, compute_worst_group_accuracy

logger = logging.getLogger("LASS.train")

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
              criterion: Callable, device: torch.device, epoch: int,
              scheduler: Optional[Any] = None, group_dro: bool = False,
              aux_loss: bool = False) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to run on.
        epoch: Current epoch number.
        scheduler: Learning rate scheduler.
        group_dro: Whether to use Group-DRO loss.
        aux_loss: Whether to use auxiliary loss.
        
    Returns:
        metrics: Training metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    group_losses = torch.zeros(4).to(device)  # Assuming 4 groups max
    group_counts = torch.zeros(4).to(device)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch
        if len(batch) == 3:
            inputs, labels, groups = batch
            spurious_labels = None
        elif len(batch) == 4:
            inputs, labels, groups, spurious_labels = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        
        inputs, labels = inputs.to(device), labels.to(device)
        groups = groups.to(device)
        if spurious_labels is not None:
            spurious_labels = spurious_labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if aux_loss:
            outputs = model(inputs)
            loss_dict = model.compute_loss(outputs, labels, groups, spurious_labels)
            loss = loss_dict['total_loss']
        elif group_dro:
            outputs = model(inputs)
            loss, batch_group_losses = model.compute_group_dro_loss(outputs, labels, groups)
            
            # Update group losses
            for g in range(len(batch_group_losses)):
                g_mask = (groups == g)
                if g_mask.sum() > 0:
                    group_losses[g] += batch_group_losses[g].item() * g_mask.sum().item()
                    group_counts[g] += g_mask.sum().item()
        else:
            outputs = model(inputs)
            
            # Handle different model outputs
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    logits = outputs['logits']
                elif 'student' in outputs and 'teacher' in outputs:
                    logits = outputs['teacher']  # For ULE model, use teacher for prediction
                else:
                    logits = list(outputs.values())[0]  # Take first output
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * inputs.size(0)
        
        # Handle different model outputs for prediction
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                pred = outputs['logits'].argmax(dim=1)
            elif 'student' in outputs and 'teacher' in outputs:
                pred = outputs['teacher'].argmax(dim=1)
            else:
                pred = list(outputs.values())[0].argmax(dim=1)
        else:
            pred = outputs.argmax(dim=1)
            
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
        
    # Update learning rate
    if scheduler is not None:
        scheduler.step()
    
    # Compute metrics
    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    # Add per-group metrics if available
    if group_dro and torch.all(group_counts > 0):
        group_metrics = {}
        for g in range(len(group_losses)):
            if group_counts[g] > 0:
                group_metrics[f'group_{g}_loss'] = group_losses[g].item() / group_counts[g].item()
        metrics.update(group_metrics)
    
    return metrics

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: Callable, 
           device: torch.device, groups: bool = True, 
           aux_loss: bool = False) -> Dict[str, float]:
    """
    Evaluate model.
    
    Args:
        model: PyTorch model.
        dataloader: Evaluation data loader.
        criterion: Loss function.
        device: Device to run on.
        groups: Whether to compute group-wise metrics.
        aux_loss: Whether to use auxiliary loss.
        
    Returns:
        metrics: Evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_preds = []
    all_groups = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            if len(batch) == 3:
                inputs, labels, batch_groups = batch
                spurious_labels = None
            elif len(batch) == 4:
                inputs, labels, batch_groups, spurious_labels = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            
            inputs, labels = inputs.to(device), labels.to(device)
            batch_groups = batch_groups.to(device)
            if spurious_labels is not None:
                spurious_labels = spurious_labels.to(device)
            
            # Forward pass
            if aux_loss:
                outputs = model(inputs)
                loss_dict = model.compute_loss(outputs, labels, batch_groups, spurious_labels)
                loss = loss_dict['total_loss']
                logits = outputs['logits']
            else:
                outputs = model(inputs)
                
                # Handle different model outputs
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'student' in outputs and 'teacher' in outputs:
                        logits = outputs['teacher']  # For ULE model, use teacher for prediction
                    else:
                        logits = list(outputs.values())[0]  # Take first output
                else:
                    logits = outputs
                
                loss = criterion(logits, labels)
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Save for later analysis
            all_labels.append(labels.cpu())
            all_preds.append(pred.cpu())
            all_groups.append(batch_groups.cpu())
            all_outputs.append(logits.cpu())
    
    # Concatenate tensors
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_groups = torch.cat(all_groups)
    all_outputs = torch.cat(all_outputs)
    
    # Compute metrics
    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    # Compute group-wise metrics if requested
    if groups:
        unique_groups = torch.unique(all_groups)
        
        # Per-group accuracy
        group_accs = {}
        for g in unique_groups:
            g_mask = (all_groups == g)
            g_correct = (all_preds[g_mask] == all_labels[g_mask]).sum().item()
            g_total = g_mask.sum().item()
            g_acc = g_correct / g_total if g_total > 0 else 0.0
            group_accs[f'group_{g.item()}_acc'] = g_acc
        
        # Worst-group accuracy
        worst_acc, worst_group = compute_worst_group_accuracy(all_outputs, all_labels, all_groups)
        group_accs['worst_group_accuracy'] = worst_acc
        group_accs['worst_group_id'] = worst_group
        
        # Add group metrics
        metrics.update(group_accs)
    
    return metrics

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              optimizer: optim.Optimizer, criterion: Callable, device: torch.device,
              num_epochs: int, save_dir: str, model_name: str,
              scheduler: Optional[Any] = None, group_dro: bool = False,
              aux_loss: bool = False, early_stopping: bool = True,
              patience: int = 5, tensorboard: bool = True) -> Dict[str, Any]:
    """
    Train and evaluate model.
    
    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to run on.
        num_epochs: Number of epochs to train for.
        save_dir: Directory to save model and logs.
        model_name: Name for saved model.
        scheduler: Learning rate scheduler.
        group_dro: Whether to use Group-DRO loss.
        aux_loss: Whether to use auxiliary loss.
        early_stopping: Whether to use early stopping.
        patience: Patience for early stopping.
        tensorboard: Whether to use TensorBoard.
        
    Returns:
        history: Training history.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    if tensorboard:
        tb_dir = os.path.join(save_dir, 'tensorboard', model_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_worst_group_acc': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_worst_group_acc = 0.0
    best_epoch = 0
    no_improve = 0
    
    # Train loop
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scheduler, group_dro, aux_loss
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(
            model, val_loader, criterion, device, groups=True, aux_loss=aux_loss
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_worst_group_acc'].append(val_metrics.get('worst_group_accuracy', 0.0))
        
        # Log metrics
        logger.info(f"Epoch {epoch}/{num_epochs} - "
                   f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}, "
                   f"Val Worst-Group Acc: {val_metrics.get('worst_group_accuracy', 0.0):.4f}")
        
        # Write to TensorBoard
        if tensorboard:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('Worst-Group Accuracy/val', val_metrics.get('worst_group_accuracy', 0.0), epoch)
            
            # Add per-group metrics if available
            for key, value in val_metrics.items():
                if key.startswith('group_') and key.endswith('_acc'):
                    writer.add_scalar(f'Group Accuracy/{key}', value, epoch)
        
        # Save best model (by worst-group accuracy)
        if val_metrics.get('worst_group_accuracy', 0.0) > best_worst_group_acc:
            best_worst_group_acc = val_metrics.get('worst_group_accuracy', 0.0)
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            # Save model
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
            
            # Reset patience counter
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if early_stopping and no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}. "
                       f"Best epoch: {best_epoch}, "
                       f"Best val loss: {best_val_loss:.4f}, "
                       f"Best val acc: {best_val_acc:.4f}, "
                       f"Best worst-group acc: {best_worst_group_acc:.4f}")
            break
    
    # Close TensorBoard writer
    if tensorboard:
        writer.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_final.pth"))
    
    # Save training history
    with open(os.path.join(save_dir, f"{model_name}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_best.pth")))
    
    # Return history and best metrics
    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_worst_group_acc': best_worst_group_acc
    }

def train_model_with_lass(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                        optimizer: optim.Optimizer, criterion: Callable, device: torch.device,
                        num_epochs: int, save_dir: str, model_name: str,
                        hypotheses: List, intervention_type: str = 'reweighting',
                        scheduler: Optional[Any] = None, early_stopping: bool = True,
                        patience: int = 5, tensorboard: bool = True) -> Dict[str, Any]:
    """
    Train model with LASS framework, incorporating LLM-generated hypotheses.
    
    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to run on.
        num_epochs: Number of epochs to train for.
        save_dir: Directory to save model and logs.
        model_name: Name for saved model.
        hypotheses: List of LLM-generated hypotheses.
        intervention_type: Type of intervention ('reweighting', 'aux_loss', or 'both').
        scheduler: Learning rate scheduler.
        early_stopping: Whether to use early stopping.
        patience: Patience for early stopping.
        tensorboard: Whether to use TensorBoard.
        
    Returns:
        history: Training history.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    if tensorboard:
        tb_dir = os.path.join(save_dir, 'tensorboard', model_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
    
    # Set up LASS-specific parameters
    use_reweighting = intervention_type in ['reweighting', 'both']
    use_aux_loss = intervention_type in ['aux_loss', 'both']
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_worst_group_acc': []
    }
    
    # Add hypothesis information to history
    history['hypotheses'] = [h.to_dict() for h in hypotheses]
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_worst_group_acc = 0.0
    best_epoch = 0
    no_improve = 0
    
    # Train loop
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scheduler, group_dro=False, aux_loss=use_aux_loss
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(
            model, val_loader, criterion, device, groups=True, aux_loss=use_aux_loss
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_worst_group_acc'].append(val_metrics.get('worst_group_accuracy', 0.0))
        
        # Log metrics
        logger.info(f"Epoch {epoch}/{num_epochs} - "
                   f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}, "
                   f"Val Worst-Group Acc: {val_metrics.get('worst_group_accuracy', 0.0):.4f}")
        
        # Write to TensorBoard
        if tensorboard:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('Worst-Group Accuracy/val', val_metrics.get('worst_group_accuracy', 0.0), epoch)
            
            # Add per-group metrics if available
            for key, value in val_metrics.items():
                if key.startswith('group_') and key.endswith('_acc'):
                    writer.add_scalar(f'Group Accuracy/{key}', value, epoch)
        
        # Update reweighting parameters if using LLMAugmentedModel with reweighting
        if use_reweighting and hasattr(model, 'update_sample_weights'):
            # Extract per-group losses
            group_losses = torch.zeros(4, device=device)  # Assuming 4 groups max
            for g in range(4):
                if f'group_{g}_acc' in val_metrics:
                    group_losses[g] = 1.0 - val_metrics[f'group_{g}_acc']
            
            # Update sample weights
            model.update_sample_weights(group_losses)
        
        # Save best model (by worst-group accuracy)
        if val_metrics.get('worst_group_accuracy', 0.0) > best_worst_group_acc:
            best_worst_group_acc = val_metrics.get('worst_group_accuracy', 0.0)
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            # Save model
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pth"))
            
            # Reset patience counter
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if early_stopping and no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}. "
                       f"Best epoch: {best_epoch}, "
                       f"Best val loss: {best_val_loss:.4f}, "
                       f"Best val acc: {best_val_acc:.4f}, "
                       f"Best worst-group acc: {best_worst_group_acc:.4f}")
            break
    
    # Close TensorBoard writer
    if tensorboard:
        writer.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_final.pth"))
    
    # Save training history
    with open(os.path.join(save_dir, f"{model_name}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_best.pth")))
    
    # Return history and best metrics
    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_worst_group_acc': best_worst_group_acc
    }

def extract_embeddings(model: nn.Module, dataloader: DataLoader, 
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract embeddings, predictions, and confidences from a trained model.
    
    Args:
        model: PyTorch model.
        dataloader: Data loader.
        device: Device to run on.
        
    Returns:
        embeddings: Feature embeddings.
        labels: True labels.
        predictions: Model predictions.
        confidences: Prediction confidences.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_preds = []
    all_confs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, groups = batch[:3]  # Ignore additional items if any
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get embeddings
            embeddings = model.get_embeddings(inputs)
            
            # Get predictions
            outputs = model(inputs)
            
            # Handle different model outputs
            if isinstance(outputs, dict):
                if 'logits' in outputs:
                    logits = outputs['logits']
                elif 'student' in outputs and 'teacher' in outputs:
                    logits = outputs['teacher']  # For ULE model, use teacher for prediction
                else:
                    logits = list(outputs.values())[0]  # Take first output
            else:
                logits = outputs
            
            # Get predictions and confidences
            preds = logits.argmax(dim=1)
            confs = F.softmax(logits, dim=1).max(dim=1)[0]
            
            # Save for later analysis
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_confs.append(confs.cpu())
    
    # Concatenate tensors
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_confs = torch.cat(all_confs, dim=0)
    
    return all_embeddings, all_labels, all_preds, all_confs