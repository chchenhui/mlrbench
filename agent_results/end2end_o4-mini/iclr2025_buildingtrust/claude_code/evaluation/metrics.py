"""
Evaluation metrics for the Cluster-Driven Certified Unlearning experiment.
"""

import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_perplexity(model, data_loader, device):
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Language model
        data_loader: DataLoader providing examples
        device: Device for computation
        
    Returns:
        perplexity (float): Perplexity score
    """
    model.eval()
    total_loss = 0
    total_length = 0
    
    # Use cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in data_loader:
            # Move inputs to device
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            targets = batch['targets'].to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Compute loss
            if logits.shape[:-1] != targets.shape:
                # Reshape logits and targets
                shifted_logits = logits.contiguous().view(-1, logits.size(-1))
                shifted_targets = targets.contiguous().view(-1)
            else:
                shifted_logits = logits
                shifted_targets = targets
                
            # Compute loss
            loss = loss_fn(shifted_logits, shifted_targets)
            
            # Accumulate loss
            total_loss += loss.item()
            total_length += torch.sum(batch['attention_mask']).item()
    
    # Compute perplexity
    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def compute_knowledge_forgetting_rate(original_model, unlearned_model, deletion_set, device):
    """
    Compute Knowledge Forgetting Rate (KFR) to measure how well the model has forgotten
    the targeted information.
    
    Args:
        original_model: Original language model
        unlearned_model: Unlearned language model
        deletion_set: Set of examples to delete
        device: Device for computation
        
    Returns:
        kfr (float): Knowledge Forgetting Rate
    """
    original_model.eval()
    unlearned_model.eval()
    
    # Initialize metrics
    total_examples = 0
    total_forgotten = 0
    
    with torch.no_grad():
        for batch in deletion_set:
            # Move inputs to device
            if isinstance(batch, dict):
                # Single example
                inputs = {
                    'input_ids': batch['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': batch['attention_mask'].unsqueeze(0).to(device)
                }
                targets = batch['targets'].unsqueeze(0).to(device)
            else:
                # Batch
                inputs = {
                    'input_ids': batch[0]['input_ids'].to(device),
                    'attention_mask': batch[0]['attention_mask'].to(device)
                }
                targets = batch[1].to(device)
            
            # Forward pass with original model
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits
            
            # Forward pass with unlearned model
            unlearned_outputs = unlearned_model(**inputs)
            unlearned_logits = unlearned_outputs.logits
            
            # Compute predictions
            if original_logits.shape[:-1] != targets.shape:
                # Reshape logits and targets
                original_preds = original_logits.contiguous().view(-1, original_logits.size(-1)).argmax(dim=-1)
                unlearned_preds = unlearned_logits.contiguous().view(-1, unlearned_logits.size(-1)).argmax(dim=-1)
                targets = targets.contiguous().view(-1)
            else:
                original_preds = original_logits.argmax(dim=-1)
                unlearned_preds = unlearned_logits.argmax(dim=-1)
                
            # Count examples where original was correct but unlearned is not
            correct_original = (original_preds == targets)
            correct_unlearned = (unlearned_preds == targets)
            forgotten = (correct_original & ~correct_unlearned)
            
            # Update counts
            total_examples += targets.size(0)
            total_forgotten += forgotten.sum().item()
    
    # Compute Knowledge Forgetting Rate
    kfr = total_forgotten / total_examples if total_examples > 0 else 0
    
    return kfr


def compute_knowledge_retention_rate(original_model, unlearned_model, test_loader, device):
    """
    Compute Knowledge Retention Rate (KRR) to measure how well the model has retained
    unrelated information.
    
    Args:
        original_model: Original language model
        unlearned_model: Unlearned language model
        test_loader: DataLoader for test data
        device: Device for computation
        
    Returns:
        krr (float): Knowledge Retention Rate
    """
    original_model.eval()
    unlearned_model.eval()
    
    # Initialize metrics
    total_examples = 0
    total_retained = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            targets = batch['targets'].to(device)
            
            # Forward pass with original model
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits
            
            # Forward pass with unlearned model
            unlearned_outputs = unlearned_model(**inputs)
            unlearned_logits = unlearned_outputs.logits
            
            # Compute predictions
            if original_logits.shape[:-1] != targets.shape:
                # Reshape logits and targets
                original_preds = original_logits.contiguous().view(-1, original_logits.size(-1)).argmax(dim=-1)
                unlearned_preds = unlearned_logits.contiguous().view(-1, unlearned_logits.size(-1)).argmax(dim=-1)
                targets = targets.contiguous().view(-1)
            else:
                original_preds = original_logits.argmax(dim=-1)
                unlearned_preds = unlearned_logits.argmax(dim=-1)
                
            # Count examples where both models are correct
            correct_original = (original_preds == targets)
            correct_unlearned = (unlearned_preds == targets)
            retained = (correct_original & correct_unlearned)
            
            # Update counts
            total_examples += targets.size(0)
            total_retained += retained.sum().item()
    
    # Compute Knowledge Retention Rate
    krr = total_retained / total_examples if total_examples > 0 else 0
    
    return krr


def evaluate_downstream_task(model, task_data_loader, task_type, device):
    """
    Evaluate model performance on a downstream task.
    
    Args:
        model: Language model
        task_data_loader: DataLoader for task data
        task_type: Type of task ('classification', 'qa', etc.)
        device: Device for computation
        
    Returns:
        metrics (dict): Task-specific metrics
    """
    model.eval()
    
    # Initialize predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in task_data_loader:
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            targets = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            preds = logits.argmax(dim=-1)
            
            # Append to lists
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute metrics based on task type
    if task_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted'),
            'recall': recall_score(all_targets, all_preds, average='weighted'),
            'f1': f1_score(all_targets, all_preds, average='weighted')
        }
    elif task_type == 'qa':
        # For QA tasks, we use exact match and F1 score
        metrics = {
            'exact_match': accuracy_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds, average='weighted')
        }
    else:
        # Default to basic metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds)
        }
    
    return metrics


def compute_computational_cost(start_time, end_time, peak_memory=None):
    """
    Compute computational cost metrics.
    
    Args:
        start_time: Start time of computation
        end_time: End time of computation
        peak_memory: Peak memory usage (bytes)
        
    Returns:
        cost_metrics (dict): Computational cost metrics
    """
    # Compute wall-clock time
    wall_clock_time = end_time - start_time
    
    cost_metrics = {
        'wall_clock_time': wall_clock_time,
        'wall_clock_time_minutes': wall_clock_time / 60,
    }
    
    # Add peak memory if provided
    if peak_memory is not None:
        cost_metrics['peak_memory_mb'] = peak_memory / (1024 * 1024)
        cost_metrics['peak_memory_gb'] = peak_memory / (1024 * 1024 * 1024)
    
    return cost_metrics


def evaluate_membership_inference(original_model, unlearned_model, deletion_set, test_set, device):
    """
    Evaluate the model's vulnerability to membership inference attacks.
    
    Args:
        original_model: Original language model
        unlearned_model: Unlearned language model
        deletion_set: Set of examples to delete
        test_set: Set of examples not in the training set
        device: Device for computation
        
    Returns:
        attack_metrics (dict): Membership inference attack metrics
    """
    original_model.eval()
    unlearned_model.eval()
    
    # Function to compute loss for a batch
    def compute_batch_loss(model, batch):
        if isinstance(batch, dict):
            # Single example
            inputs = {
                'input_ids': batch['input_ids'].unsqueeze(0).to(device),
                'attention_mask': batch['attention_mask'].unsqueeze(0).to(device)
            }
            targets = batch['targets'].unsqueeze(0).to(device)
        else:
            # Batch
            inputs = {
                'input_ids': batch[0]['input_ids'].to(device),
                'attention_mask': batch[0]['attention_mask'].to(device)
            }
            targets = batch[1].to(device)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
        if logits.shape[:-1] != targets.shape:
            # Reshape logits and targets
            shifted_logits = logits.contiguous().view(-1, logits.size(-1))
            shifted_targets = targets.contiguous().view(-1)
        else:
            shifted_logits = logits
            shifted_targets = targets
            
        # Compute per-example loss
        losses = loss_fn(shifted_logits, shifted_targets)
        
        return losses.mean().item()
    
    # Compute losses for deletion set and test set
    original_deletion_losses = []
    unlearned_deletion_losses = []
    original_test_losses = []
    unlearned_test_losses = []
    
    with torch.no_grad():
        # Compute losses for deletion set
        for batch in deletion_set:
            original_loss = compute_batch_loss(original_model, batch)
            unlearned_loss = compute_batch_loss(unlearned_model, batch)
            
            original_deletion_losses.append(original_loss)
            unlearned_deletion_losses.append(unlearned_loss)
        
        # Compute losses for test set
        for i in range(min(len(test_set), len(deletion_set))):
            batch = test_set[i]
            original_loss = compute_batch_loss(original_model, batch)
            unlearned_loss = compute_batch_loss(unlearned_model, batch)
            
            original_test_losses.append(original_loss)
            unlearned_test_losses.append(unlearned_loss)
    
    # Compute attack success rate
    # Lower loss on deletion set than test set indicates membership
    original_attack_success = 0
    unlearned_attack_success = 0
    
    for i in range(len(original_deletion_losses)):
        if i < len(original_test_losses):
            if original_deletion_losses[i] < original_test_losses[i]:
                original_attack_success += 1
            if unlearned_deletion_losses[i] < unlearned_test_losses[i]:
                unlearned_attack_success += 1
    
    # Compute attack success rates
    original_attack_rate = original_attack_success / len(original_deletion_losses)
    unlearned_attack_rate = unlearned_attack_success / len(unlearned_deletion_losses)
    
    # Compute attack difficulty (higher is better, means harder to infer membership)
    attack_difficulty = 1 - (unlearned_attack_rate / original_attack_rate) if original_attack_rate > 0 else 1
    
    attack_metrics = {
        'original_attack_success_rate': original_attack_rate,
        'unlearned_attack_success_rate': unlearned_attack_rate,
        'attack_difficulty_improvement': attack_difficulty,
    }
    
    return attack_metrics