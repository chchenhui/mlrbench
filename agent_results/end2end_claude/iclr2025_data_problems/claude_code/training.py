#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training functionality for Attribution-Guided Training experiments.
"""

import os
import json
import time
import logging
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    device: torch.device,
    task_type: str = "attribution",
    lambda_attr: float = 0.1
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to use for training
        task_type: Type of task ("attribution", "mlm", or "mlm_with_attribution")
        lambda_attr: Weight of attribution loss (for mlm_with_attribution)
        
    Returns:
        Dictionary with training metrics for the epoch
    """
    model.train()
    
    total_loss = 0.0
    total_attr_loss = 0.0
    total_task_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        source_idx = batch["source_idx"].to(device)
        
        # Forward pass
        if task_type == "attribution":
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                source_idx=source_idx
            )
            loss = outputs["loss"]
            attr_loss = outputs.get("attribution_loss", loss)
            task_loss = 0.0
            
        elif task_type == "mlm":
            # Create MLM labels by masking 15% of tokens
            labels = input_ids.clone()
            mask_prob = 0.15
            mask = torch.rand(input_ids.shape, device=device) < mask_prob
            # Don't mask special tokens
            special_tokens_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
            for token_id in [0, 1, 2]:  # CLS, SEP, etc.
                special_tokens_mask |= (input_ids == token_id)
            mask &= ~special_tokens_mask
            
            # Replace masked tokens with [MASK] token (typically token ID 103 for BERT-based models)
            masked_token_id = 103
            input_ids = input_ids.clone()
            input_ids[mask] = masked_token_id
            
            # Forward pass with MLM
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
            attr_loss = 0.0
            task_loss = loss
            
        elif task_type == "mlm_with_attribution":
            # Create MLM labels by masking 15% of tokens
            labels = input_ids.clone()
            mask_prob = 0.15
            mask = torch.rand(input_ids.shape, device=device) < mask_prob
            # Don't mask special tokens
            special_tokens_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
            for token_id in [0, 1, 2]:  # CLS, SEP, etc.
                special_tokens_mask |= (input_ids == token_id)
            mask &= ~special_tokens_mask
            
            # Replace masked tokens with [MASK] token (typically token ID 103 for BERT-based models)
            masked_token_id = 103
            input_ids = input_ids.clone()
            input_ids[mask] = masked_token_id
            
            # Forward pass with both MLM and attribution
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                source_idx=source_idx,
                labels=labels
            )
            loss = outputs["loss"]
            attr_loss = outputs.get("attribution_loss", 0.0)
            task_loss = outputs.get("mlm_loss", 0.0)
            
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        if attr_loss != 0:
            total_attr_loss += attr_loss.item()
            
        if task_loss != 0:
            total_task_loss += task_loss.item()
        
        # Track attribution accuracy
        if task_type in ["attribution", "mlm_with_attribution"] and "source_logits" in outputs:
            source_logits = outputs["source_logits"]
            predictions = torch.argmax(source_logits, dim=-1)
            correct = (predictions == source_idx).sum().item()
            total_correct += correct
            total_samples += source_idx.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })
    
    # Calculate epoch metrics
    metrics = {
        "loss": total_loss / len(train_loader)
    }
    
    if total_attr_loss > 0:
        metrics["attribution_loss"] = total_attr_loss / len(train_loader)
        
    if total_task_loss > 0:
        metrics["task_loss"] = total_task_loss / len(train_loader)
    
    if total_samples > 0:
        metrics["accuracy"] = total_correct / total_samples
    
    return metrics

def validate(
    model,
    val_loader: DataLoader,
    device: torch.device,
    task_type: str = "attribution",
    lambda_attr: float = 0.1
) -> Dict[str, float]:
    """
    Validate model on validation data.
    
    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        device: Device to use for validation
        task_type: Type of task ("attribution", "mlm", or "mlm_with_attribution")
        lambda_attr: Weight of attribution loss (for mlm_with_attribution)
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_attr_loss = 0.0
    total_task_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            source_idx = batch["source_idx"].to(device)
            
            # Forward pass
            if task_type == "attribution":
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    source_idx=source_idx
                )
                loss = outputs["loss"]
                attr_loss = outputs.get("attribution_loss", loss)
                task_loss = 0.0
                
            elif task_type == "mlm":
                # Create MLM labels
                labels = input_ids.clone()
                mask_prob = 0.15
                mask = torch.rand(input_ids.shape, device=device) < mask_prob
                # Don't mask special tokens
                special_tokens_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
                for token_id in [0, 1, 2]:  # CLS, SEP, etc.
                    special_tokens_mask |= (input_ids == token_id)
                mask &= ~special_tokens_mask
                
                # Replace masked tokens with [MASK] token
                masked_token_id = 103
                input_ids = input_ids.clone()
                input_ids[mask] = masked_token_id
                
                # Forward pass with MLM
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                attr_loss = 0.0
                task_loss = loss
                
            elif task_type == "mlm_with_attribution":
                # Create MLM labels
                labels = input_ids.clone()
                mask_prob = 0.15
                mask = torch.rand(input_ids.shape, device=device) < mask_prob
                # Don't mask special tokens
                special_tokens_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)
                for token_id in [0, 1, 2]:  # CLS, SEP, etc.
                    special_tokens_mask |= (input_ids == token_id)
                mask &= ~special_tokens_mask
                
                # Replace masked tokens with [MASK] token
                masked_token_id = 103
                input_ids = input_ids.clone()
                input_ids[mask] = masked_token_id
                
                # Forward pass with both MLM and attribution
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    source_idx=source_idx,
                    labels=labels
                )
                loss = outputs["loss"]
                attr_loss = outputs.get("attribution_loss", 0.0)
                task_loss = outputs.get("mlm_loss", 0.0)
                
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            
            # Track metrics
            total_loss += loss.item()
            
            if attr_loss != 0:
                total_attr_loss += attr_loss.item()
                
            if task_loss != 0:
                total_task_loss += task_loss.item()
            
            # Track attribution accuracy
            if task_type in ["attribution", "mlm_with_attribution"] and "source_logits" in outputs:
                source_logits = outputs["source_logits"]
                predictions = torch.argmax(source_logits, dim=-1)
                correct = (predictions == source_idx).sum().item()
                total_correct += correct
                total_samples += source_idx.size(0)
    
    # Calculate validation metrics
    metrics = {
        "loss": total_loss / len(val_loader)
    }
    
    if total_attr_loss > 0:
        metrics["attribution_loss"] = total_attr_loss / len(val_loader)
        
    if total_task_loss > 0:
        metrics["task_loss"] = total_task_loss / len(val_loader)
    
    if total_samples > 0:
        metrics["accuracy"] = total_correct / total_samples
    
    return metrics

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    name: str = "model"
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        name: Model name for the checkpoint
        
    Returns:
        Path to the saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}_epoch_{epoch}.pt")
    
    # Save model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model separately if this is the best so far
    best_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
    if not os.path.exists(best_path) or metrics["loss"] < torch.load(best_path)["metrics"]["loss"]:
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")
    
    return checkpoint_path

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path: str
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (epoch, metrics)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint["epoch"]
    metrics = checkpoint["metrics"]
    
    logger.info(f"Loaded checkpoint from epoch {epoch} with loss {metrics['loss']:.4f}")
    
    return epoch, metrics

def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_type: str = "attribution",
    lambda_attr: float = 0.1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 10,
    device: torch.device = None,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "model",
    early_stopping_patience: int = 3
) -> Dict[str, Any]:
    """
    Train a model with evaluation.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        task_type: Type of task ("attribution", "mlm", or "mlm_with_attribution")
        lambda_attr: Weight of attribution loss (for mlm_with_attribution)
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        num_epochs: Number of epochs to train
        device: Device to use for training
        checkpoint_dir: Directory to save checkpoints
        model_name: Model name for checkpoints
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        Dictionary with training history and paths to saved checkpoints
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track training progress
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_times": []
    }
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Using device: {device}")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Track epoch time
        start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            task_type=task_type,
            lambda_attr=lambda_attr
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            task_type=task_type,
            lambda_attr=lambda_attr
        )
        
        # Update scheduler
        scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - start_time
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Time: {epoch_time:.2f}s")
        
        if "accuracy" in train_metrics:
            logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
                      f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["learning_rate"].append(scheduler.get_last_lr()[0])
        history["epoch_times"].append(epoch_time)
        
        if "accuracy" in train_metrics:
            history["train_accuracy"].append(train_metrics["accuracy"])
            
        if "accuracy" in val_metrics:
            history["val_accuracy"].append(val_metrics["accuracy"])
        
        # Check for improvement
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_epoch = epoch
            
            # Save checkpoint for best model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_dir=checkpoint_dir,
                name=f"{model_name}_best"
            )
        else:
            patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    checkpoint_dir=checkpoint_dir,
                    name=f"{model_name}_epoch_{epoch+1}"
                )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final checkpoint
    final_checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=val_metrics,
        checkpoint_dir=checkpoint_dir,
        name=f"{model_name}_final"
    )
    
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    
    # Load best model
    load_checkpoint(model, optimizer, scheduler, best_checkpoint_path)
    
    logger.info(f"Training completed. Best epoch: {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    
    return {
        "history": history,
        "final_checkpoint_path": final_checkpoint_path,
        "best_checkpoint_path": best_checkpoint_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: str,
    model_name: str = "model"
) -> List[str]:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
        model_name: Model name for filenames
        
    Returns:
        List of paths to saved plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = []
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(output_dir, f"{model_name}_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    plot_paths.append(loss_plot_path)
    
    # Plot accuracy curves if available
    if "train_accuracy" in history and history["train_accuracy"]:
        plt.figure(figsize=(10, 6))
        plt.plot(history["train_accuracy"], label="Train Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} - Training and Validation Accuracy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        acc_plot_path = os.path.join(output_dir, f"{model_name}_accuracy.png")
        plt.savefig(acc_plot_path, dpi=300)
        plt.close()
        plot_paths.append(acc_plot_path)
    
    # Plot learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(history["learning_rate"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"{model_name} - Learning Rate Schedule")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    lr_plot_path = os.path.join(output_dir, f"{model_name}_lr.png")
    plt.savefig(lr_plot_path, dpi=300)
    plt.close()
    plot_paths.append(lr_plot_path)
    
    # Plot epoch times
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch_times"])
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title(f"{model_name} - Epoch Training Time")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    time_plot_path = os.path.join(output_dir, f"{model_name}_time.png")
    plt.savefig(time_plot_path, dpi=300)
    plt.close()
    plot_paths.append(time_plot_path)
    
    return plot_paths

def save_training_config(
    config: Dict[str, Any],
    output_dir: str,
    filename: str = "training_config.json"
) -> str:
    """
    Save training configuration.
    
    Args:
        config: Training configuration dictionary
        output_dir: Directory to save configuration
        filename: Output filename
        
    Returns:
        Path to the saved configuration file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    logger.info(f"Saved training configuration to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Test training functions
    logging.basicConfig(level=logging.INFO)
    
    # Create mock model and dataloaders
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(100, 128)
            self.lstm = torch.nn.LSTM(128, 64, batch_first=True)
            self.fc = torch.nn.Linear(64, 10)
        
        def forward(self, input_ids, attention_mask, source_idx=None):
            embeds = self.embedding(input_ids)
            lstm_out, _ = self.lstm(embeds)
            logits = self.fc(lstm_out[:, 0, :])
            
            outputs = {"source_logits": logits}
            
            if source_idx is not None:
                loss = torch.nn.functional.cross_entropy(logits, source_idx)
                outputs["loss"] = loss
                outputs["attribution_loss"] = loss
            
            return outputs
    
    model = MockModel()
    
    # Mock batch
    batch_size = 4
    seq_len = 10
    
    class MockDataLoader:
        def __init__(self, num_batches=5):
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield {
                    "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
                    "attention_mask": torch.ones(batch_size, seq_len),
                    "source_idx": torch.randint(0, 10, (batch_size,))
                }
        
        def __len__(self):
            return self.num_batches
    
    train_loader = MockDataLoader()
    val_loader = MockDataLoader(num_batches=2)
    
    # Test training
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    
    # Test one epoch
    metrics = train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device
    )
    
    logger.info(f"Test train_epoch: {metrics}")
    
    # Test validation
    val_metrics = validate(
        model=model,
        val_loader=val_loader,
        device=device
    )
    
    logger.info(f"Test validate: {val_metrics}")
    
    # Test checkpoint saving and loading
    checkpoint_dir = "test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        metrics=val_metrics,
        checkpoint_dir=checkpoint_dir,
        name="test_model"
    )
    
    epoch, loaded_metrics = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path
    )
    
    logger.info(f"Test checkpoint loading: epoch={epoch}, metrics={loaded_metrics}")
    
    logger.info("Training tests passed")