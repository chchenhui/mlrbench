#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from accelerate import Accelerator
import time
import numpy as np
import json
from tqdm import tqdm
import random
from pathlib import Path

# Local imports
from models.transformer_with_compression import TransformerWithCompression
from data.dataset_loader import get_dataset, create_dataloader
from utils.metrics import PerformanceMetrics, TextGenerationMetrics
from utils.visualization import plot_loss_curves

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a transformer model with KV cache compression.")
    
    # Model and training settings
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, default="wikitext-103-v1",
                        help="The name of the dataset to use.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    
    # Compression settings
    parser.add_argument("--max_cache_size", type=int, default=1024,
                        help="Maximum number of KV pairs to retain after pruning.")
    parser.add_argument("--num_clusters", type=int, default=256,
                        help="Number of cluster centroids for low-rank summarization.")
    parser.add_argument("--pruning_interval", type=int, default=512,
                        help="Interval (in tokens) between pruning operations.")
    parser.add_argument("--lookback_window", type=int, default=256,
                        help="Number of recent positions to consider for importance.")
    parser.add_argument("--kmeans_learning_rate", type=float, default=0.01,
                        help="Learning rate for online k-means updates.")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for distillation loss.")
    parser.add_argument("--distillation_weight", type=float, default=0.5,
                        help="Weight of distillation loss.")
    parser.add_argument("--use_compression", action="store_true",
                        help="Whether to use compression during training.")
    
    # Evaluation settings
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch",
                        help="The evaluation strategy to use. One of ['epoch', 'steps']")
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Run evaluation every X steps. Applies when evaluation_strategy='steps'")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    
    # Dataset settings
    parser.add_argument("--dataset_cache_dir", type=str, default="data/cache",
                        help="Directory where datasets are cached.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of examples to use for training and evaluation.")
    
    args = parser.parse_args()
    return args

def train(args):
    """
    Train model with KV cache compression.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(fp16=args.fp16)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = get_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_length,
        cache_dir=args.dataset_cache_dir,
        sample_size=args.sample_size
    )
    
    eval_dataset = get_dataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        split="validation",
        max_length=args.max_length,
        cache_dir=args.dataset_cache_dir,
        sample_size=args.sample_size
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    eval_dataloader = create_dataloader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    
    # Initialize model with compression
    logger.info(f"Initializing model: {args.model_name_or_path}")
    model = TransformerWithCompression(
        model_name_or_path=args.model_name_or_path,
        max_cache_size=args.max_cache_size,
        num_clusters=args.num_clusters,
        pruning_interval=args.pruning_interval,
        lookback_window=args.lookback_window,
        kmeans_learning_rate=args.kmeans_learning_rate,
        temperature=args.temperature,
        distillation_weight=args.distillation_weight,
        use_compression=args.use_compression
    )
    
    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Metrics and tracking
    metrics = PerformanceMetrics()
    train_losses = []
    eval_losses = []
    global_step = 0
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    for epoch in range(args.num_train_epochs):
        metrics.reset()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(progress_bar):
            start_time = time.time()
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                do_compression=args.use_compression
            )
            
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update metrics
            metrics.update(
                batch_size=batch["input_ids"].shape[0],
                seq_length=batch["input_ids"].shape[1],
                elapsed_time=time.time() - start_time,
                loss=loss.item() * args.gradient_accumulation_steps
            )
            
            epoch_loss += loss.item()
            
            # Update model parameters
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log progress
                if global_step % args.logging_steps == 0:
                    current_metrics = metrics.get_metrics()
                    progress_bar.set_postfix({
                        "loss": current_metrics["average_loss"],
                        "ppl": current_metrics["perplexity"],
                        "tok/s": current_metrics["tokens_per_second"]
                    })
                    
                    logger.info(f"Step {global_step}: loss={current_metrics['average_loss']:.4f}, "
                               f"ppl={current_metrics['perplexity']:.2f}, "
                               f"speed={current_metrics['tokens_per_second']:.2f} tok/s")
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save model and tokenizer
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    # Save training args
                    json.dump(vars(args), open(os.path.join(output_dir, "training_args.json"), "w"), indent=2)
                    
                    logger.info(f"Saved checkpoint to {output_dir}")
                
                # Run evaluation
                if args.evaluate_during_training and args.evaluation_strategy == "steps" and args.eval_steps is not None:
                    if global_step % args.eval_steps == 0:
                        eval_results = evaluate(args, model, eval_dataloader, accelerator)
                        eval_losses.append(eval_results["average_loss"])
                        
                        # Log eval results
                        logger.info(f"Eval at step {global_step}: loss={eval_results['average_loss']:.4f}, "
                                  f"ppl={eval_results['perplexity']:.2f}")
        
        # Compute average loss for the epoch
        epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(epoch_loss)
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} completed. Average loss: {epoch_loss:.4f}")
        
        # Run evaluation at the end of each epoch
        if args.evaluate_during_training and args.evaluation_strategy == "epoch":
            eval_results = evaluate(args, model, eval_dataloader, accelerator)
            eval_losses.append(eval_results["average_loss"])
            
            # Log eval results
            logger.info(f"Eval after epoch {epoch+1}: loss={eval_results['average_loss']:.4f}, "
                      f"ppl={eval_results['perplexity']:.2f}")
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Save model and tokenizer
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Save training args
    json.dump(vars(args), open(os.path.join(final_output_dir, "training_args.json"), "w"), indent=2)
    
    logger.info(f"Saved final model to {final_output_dir}")
    
    # Plot loss curves
    if train_losses and eval_losses:
        plot_loss_curves(
            train_losses=train_losses,
            val_losses=eval_losses,
            save_path=os.path.join(args.output_dir, "loss_curves.png")
        )
    
    logger.info("Training completed!")
    
    return {
        "model_path": final_output_dir,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "global_steps": global_step
    }

def evaluate(args, model, eval_dataloader, accelerator):
    """
    Evaluate model on evaluation dataset.
    
    Args:
        args: Command line arguments
        model: Model to evaluate
        eval_dataloader: DataLoader for evaluation dataset
        accelerator: Accelerator instance
        
    Returns:
        eval_metrics: Dictionary of evaluation metrics
    """
    logger.info("Running evaluation...")
    
    model.eval()
    metrics = PerformanceMetrics()
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            start_time = time.time()
            
            # Forward pass without compression (for fair comparison)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                do_compression=False
            )
            
            loss = outputs.loss
            
            # Update metrics
            metrics.update(
                batch_size=batch["input_ids"].shape[0],
                seq_length=batch["input_ids"].shape[1],
                elapsed_time=time.time() - start_time,
                loss=loss.item()
            )
    
    # Get evaluation metrics
    eval_metrics = metrics.get_metrics()
    
    # Also evaluate with compression if enabled
    if args.use_compression:
        logger.info("Running evaluation with compression...")
        
        metrics_compressed = PerformanceMetrics()
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating with compression"):
                start_time = time.time()
                
                # Forward pass with compression
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    do_compression=True
                )
                
                loss = outputs.loss
                
                # Update metrics
                metrics_compressed.update(
                    batch_size=batch["input_ids"].shape[0],
                    seq_length=batch["input_ids"].shape[1],
                    elapsed_time=time.time() - start_time,
                    loss=loss.item()
                )
        
        # Get compressed evaluation metrics
        eval_metrics_compressed = metrics_compressed.get_metrics()
        
        # Log compression comparison
        logger.info(f"Compression results: ")
        logger.info(f"  No compression - Loss: {eval_metrics['average_loss']:.4f}, PPL: {eval_metrics['perplexity']:.2f}, Speed: {eval_metrics['tokens_per_second']:.2f} tok/s")
        logger.info(f"  With compression - Loss: {eval_metrics_compressed['average_loss']:.4f}, PPL: {eval_metrics_compressed['perplexity']:.2f}, Speed: {eval_metrics_compressed['tokens_per_second']:.2f} tok/s")
        logger.info(f"  Speedup: {eval_metrics_compressed['tokens_per_second'] / eval_metrics['tokens_per_second']:.2f}x")
        
        # Add compression metrics to results
        eval_metrics["compressed"] = eval_metrics_compressed
    
    model.train()
    return eval_metrics

if __name__ == "__main__":
    args = parse_args()
    train_results = train(args)
    
    # Save train results
    json.dump(
        {k: v for k, v in train_results.items() if k != "model_path"}, 
        open(os.path.join(args.output_dir, "train_results.json"), "w"), 
        indent=2
    )