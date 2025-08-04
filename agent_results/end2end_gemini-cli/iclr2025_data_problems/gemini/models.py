
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
import logging
from tqdm import tqdm
import numpy as np

def get_student_model(model_name, num_labels=4):
    """
    Loads the student model for sequence classification.
    """
    logging.info(f"Loading student model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True
    )
    # Pythia models might not have a pad token id, set it manually.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model

def train_student(model, dataloader, optimizer, device, num_epochs=1):
    """
    Trains the student model for a specified number of epochs.
    """
    model.train()
    model.to(device)
    
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Handle dictionary batches from Hugging Face datasets
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
            else: # Handle TensorDataset batches
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy})

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct_predictions / total_predictions
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        logging.info(f"Epoch {epoch+1} Summary - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
    return history

def evaluate_student(model, dataloader, device):
    """
    Evaluates the student model on a given dataloader.
    """
    model.eval()
    model.to(device)
    
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle dictionary batches from Hugging Face datasets
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
            else: # Handle TensorDataset batches
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    logging.info(f"Evaluation - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy, avg_loss

def find_hard_examples(model, dataloader, tokenizer, device, num_examples=50):
    """
    Identifies the most difficult examples for the model based on loss.
    """
    model.eval()
    model.to(device)
    
    losses = []
    original_texts = []
    original_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle dictionary batches from Hugging Face datasets
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
            else: # Handle TensorDataset batches
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Calculate loss per example
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(outputs.logits, labels)
            losses.extend(loss.cpu().numpy())
            
            # Decode texts to get the original strings
            texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            original_texts.extend(texts)
            original_labels.extend(labels.cpu().numpy())

    # Get indices of the hardest examples
    hardest_indices = np.argsort(losses)[-num_examples:]
    
    hard_examples = [{
        "text": original_texts[i],
        "label": original_labels[i],
        "loss": losses[i]
    } for i in hardest_indices]
    
    logging.info(f"Identified {len(hard_examples)} hard examples.")
    return hard_examples
