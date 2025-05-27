"""
Training utilities for target models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import numpy as np
import time
import logging
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import calculate_accuracy

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=50, scheduler=None, early_stopping=None,
                checkpoint_path=None, verbose=True):
    """
    Train a model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        scheduler: Optional learning rate scheduler
        early_stopping: Optional early stopping criteria
        checkpoint_path: Optional path to save model checkpoints
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained model, training history)
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_epoch': 0
    }
    
    # Initialize best metrics
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         disable=not verbose)
        
        for inputs, targets in train_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate epoch training metrics
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                          disable=not verbose)
            
            for inputs, targets in val_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate epoch validation metrics
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            
            # Save checkpoint if path is provided
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Model checkpoint saved to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        if verbose:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                       f"Time={epoch_time:.2f}s")
        
        # Early stopping
        if early_stopping is not None and patience_counter >= early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    # If checkpoint was saved, load the best model
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded best model from epoch {history['best_epoch']+1}")
    
    return model, history


def train_adversarial_model(base_model, train_loader, adversarial_loader, val_loader, 
                           criterion, optimizer, device, num_epochs=50, 
                           alpha=0.5, scheduler=None, checkpoint_path=None, verbose=True):
    """
    Train a model on a mix of original and adversarial examples.
    
    Args:
        base_model: The base model to finetune
        train_loader: Original training data loader
        adversarial_loader: Adversarial data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        alpha: Weight for original data loss (1-alpha for adversarial)
        scheduler: Optional learning rate scheduler
        checkpoint_path: Optional path to save model checkpoints
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained model, training history)
    """
    model = base_model.to(device)
    
    history = {
        'train_loss': [],
        'adv_loss': [],
        'combined_loss': [],
        'val_loss': [],
        'train_acc': [],
        'adv_acc': [],
        'val_acc': [],
        'best_epoch': 0
    }
    
    # Initialize best metrics
    best_val_loss = float('inf')
    
    # Start training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        adv_loss = 0.0
        combined_loss = 0.0
        train_correct = 0
        adv_correct = 0
        train_total = 0
        adv_total = 0
        
        # Get iterators for both loaders
        train_iter = iter(train_loader)
        adv_iter = iter(adversarial_loader)
        
        # Determine number of batches (use the smaller of the two)
        num_batches = min(len(train_loader), len(adversarial_loader))
        
        for _ in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                     disable=not verbose):
            # Get original data batch
            try:
                orig_inputs, orig_targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                orig_inputs, orig_targets = next(train_iter)
            
            # Get adversarial data batch
            try:
                adv_inputs, adv_targets = next(adv_iter)
            except StopIteration:
                adv_iter = iter(adversarial_loader)
                adv_inputs, adv_targets = next(adv_iter)
            
            # Move data to device
            orig_inputs, orig_targets = orig_inputs.to(device), orig_targets.to(device)
            adv_inputs, adv_targets = adv_inputs.to(device), adv_targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass - original data
            orig_outputs = model(orig_inputs)
            loss_orig = criterion(orig_outputs, orig_targets)
            
            # Forward pass - adversarial data
            adv_outputs = model(adv_inputs)
            loss_adv = criterion(adv_outputs, adv_targets)
            
            # Combined loss
            loss = alpha * loss_orig + (1 - alpha) * loss_adv
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate metrics - original data
            train_loss += loss_orig.item() * orig_inputs.size(0)
            _, predicted = torch.max(orig_outputs, 1)
            train_total += orig_targets.size(0)
            train_correct += (predicted == orig_targets).sum().item()
            
            # Calculate metrics - adversarial data
            adv_loss += loss_adv.item() * adv_inputs.size(0)
            _, predicted = torch.max(adv_outputs, 1)
            adv_total += adv_targets.size(0)
            adv_correct += (predicted == adv_targets).sum().item()
            
            # Combined loss for history
            combined_loss += loss.item() * orig_inputs.size(0)
        
        # Calculate epoch training metrics
        train_loss = train_loss / train_total
        adv_loss = adv_loss / adv_total
        combined_loss = combined_loss / train_total
        train_acc = 100 * train_correct / train_total
        adv_acc = 100 * adv_correct / adv_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                          disable=not verbose)
            
            for inputs, targets in val_iter:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate epoch validation metrics
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['adv_loss'].append(adv_loss)
        history['combined_loss'].append(combined_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['adv_acc'].append(adv_acc)
        history['val_acc'].append(val_acc)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            
            # Save checkpoint if path is provided
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        if verbose:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Adv Loss={adv_loss:.4f}, Adv Acc={adv_acc:.2f}%, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                       f"Time={epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    
    # If checkpoint was saved, load the best model
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded best model from epoch {history['best_epoch']+1}")
    
    return model, history