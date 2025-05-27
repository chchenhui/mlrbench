#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training and evaluation utilities for CIMRL experiments.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from .metrics import compute_metrics, compute_worst_group_metrics


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config, logger, model_name):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to use for training
        config: Training configuration
        logger: Logger for logging training progress
        model_name: Name of the model
        
    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
    """
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('results', 'tensorboard', model_name))
    
    # Training parameters
    num_epochs = config.get('num_epochs', 50)
    patience = config.get('patience', 5)
    gradient_clip = config.get('gradient_clip', 1.0)
    
    # Lists to store losses and metrics
    train_losses = []
    val_losses = []
    train_metrics = {'accuracy': [], 'worst_group_accuracy': [], 'average_auc': []}
    val_metrics = {'accuracy': [], 'worst_group_accuracy': [], 'average_auc': []}
    
    # Best validation metrics for early stopping
    best_val_loss = float('inf')
    best_val_metric = 0.0
    patience_counter = 0
    
    # Save best model
    best_model_path = os.path.join('results', 'checkpoints', f'{model_name}_best.pt')
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        epoch_train_loss, epoch_train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip=gradient_clip
        )
        
        # Validate
        epoch_val_loss, epoch_val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config
        )
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
        
        # Log results
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                   f"Train Loss: {epoch_train_loss:.4f} | "
                   f"Val Loss: {epoch_val_loss:.4f} | "
                   f"Train Acc: {epoch_train_metrics['accuracy']:.4f} | "
                   f"Val Acc: {epoch_val_metrics['accuracy']:.4f} | "
                   f"Train WG Acc: {epoch_train_metrics['worst_group_accuracy']:.4f} | "
                   f"Val WG Acc: {epoch_val_metrics['worst_group_accuracy']:.4f} | "
                   f"Time: {epoch_time:.2f}s")
        
        # Store losses and metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        for metric in train_metrics:
            train_metrics[metric].append(epoch_train_metrics[metric])
            val_metrics[metric].append(epoch_val_metrics[metric])
        
        # Write to TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        
        for metric in train_metrics:
            writer.add_scalar(f'Metrics/{metric}/train', epoch_train_metrics[metric], epoch)
            writer.add_scalar(f'Metrics/{metric}/val', epoch_val_metrics[metric], epoch)
        
        # Check for early stopping (monitor worst group accuracy)
        monitor_metric = 'worst_group_accuracy'
        current_metric = epoch_val_metrics[monitor_metric]
        
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_metrics': epoch_val_metrics
            }, best_model_path)
            
            logger.info(f"Saved new best model with {monitor_metric} = {current_metric:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} with "
               f"{monitor_metric} = {checkpoint['val_metrics'][monitor_metric]:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    
    return train_losses, val_losses, train_metrics, val_metrics


def train_epoch(model, dataloader, optimizer, device, gradient_clip=1.0):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to use for training
        gradient_clip: Gradient clipping threshold
        
    Returns:
        epoch_loss: Average loss for the epoch
        metrics: Dictionary of metrics for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    # Store predictions and targets for metrics computation
    all_targets = []
    all_preds = []
    all_probs = []
    all_group_labels = []
    
    # Iterate over batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], dict):
                for subkey in batch[key]:
                    if isinstance(batch[key][subkey], torch.Tensor):
                        batch[key][subkey] = batch[key][subkey].to(device)
        
        # Forward pass
        outputs = model(batch)
        loss = outputs['losses']['total_loss'] if 'losses' in outputs else outputs['loss']
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
        
        # Store predictions and targets
        targets = batch['labels'].cpu().numpy()
        logits = outputs['pred'].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        probs = torch.softmax(outputs['pred'].detach(), dim=1).cpu().numpy()
        
        all_targets.extend(targets)
        all_preds.extend(preds)
        all_probs.append(probs)
        
        if 'group_labels' in batch:
            all_group_labels.extend(batch['group_labels'].cpu().numpy())
    
    # Compute average loss
    epoch_loss /= len(dataloader)
    
    # Concatenate predictions and probabilities
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_preds, all_probs)
    
    # Compute worst group metrics if group labels are available
    if all_group_labels:
        worst_group_metrics = compute_worst_group_metrics(all_targets, all_preds, np.array(all_group_labels))
        metrics.update(worst_group_metrics)
    
    return epoch_loss, metrics


def evaluate_model(model, dataloader, device, config):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation
        config: Evaluation configuration
        
    Returns:
        epoch_loss: Average loss for the evaluation
        metrics: Dictionary of metrics for the evaluation
    """
    model.eval()
    epoch_loss = 0.0
    
    # Store predictions and targets for metrics computation
    all_targets = []
    all_preds = []
    all_probs = []
    all_group_labels = []
    
    # Iterate over batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(device)
            
            # Forward pass
            outputs = model(batch)
            loss = outputs['losses']['total_loss'] if 'losses' in outputs else outputs['loss']
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Store predictions and targets
            targets = batch['labels'].cpu().numpy()
            logits = outputs['pred'].detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            probs = torch.softmax(outputs['pred'].detach(), dim=1).cpu().numpy()
            
            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.append(probs)
            
            if 'group_labels' in batch:
                all_group_labels.extend(batch['group_labels'].cpu().numpy())
    
    # Compute average loss
    epoch_loss /= len(dataloader)
    
    # Concatenate predictions and probabilities
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_preds, all_probs)
    
    # Compute worst group metrics if group labels are available
    if all_group_labels:
        worst_group_metrics = compute_worst_group_metrics(all_targets, all_preds, np.array(all_group_labels))
        metrics.update(worst_group_metrics)
    
    return epoch_loss, metrics


def predict(model, dataloader, device):
    """
    Get model predictions on a dataset.
    
    Args:
        model: Model to use for prediction
        dataloader: DataLoader for the dataset
        device: Device to use for prediction
        
    Returns:
        targets: Ground truth labels
        preds: Model predictions
        probs: Prediction probabilities
        group_labels: Group labels (if available)
    """
    model.eval()
    
    # Store predictions and targets
    all_targets = []
    all_preds = []
    all_probs = []
    all_group_labels = []
    
    # Iterate over batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting", leave=False)):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(device)
            
            # Forward pass
            outputs = model(batch, compute_loss=False)
            
            # Store predictions and targets
            targets = batch['labels'].cpu().numpy()
            logits = outputs['pred'].detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            probs = torch.softmax(outputs['pred'].detach(), dim=1).cpu().numpy()
            
            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.append(probs)
            
            if 'group_labels' in batch:
                all_group_labels.extend(batch['group_labels'].cpu().numpy())
    
    # Concatenate predictions and probabilities
    all_probs = np.concatenate(all_probs, axis=0)
    
    return np.array(all_targets), np.array(all_preds), all_probs, np.array(all_group_labels) if all_group_labels else None