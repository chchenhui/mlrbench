#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running the LLM-TAC experiment.
This script orchestrates the entire pipeline for the LLM-TAC framework.
"""

import argparse
import logging
import os
import json
import time
from datetime import datetime
import torch

from data_processing import CoqDataProcessor
from models.contextual_encoding import ContextualEncoder
from models.tactic_generator import TacticGenerator
from models.reinforcement_learner import ReinforcementLearner
from models.baselines import NaiveLLM, ICLModel, TraditionalTactics
from evaluation import Evaluator
from visualization import plot_training_curve, plot_metrics_comparison, plot_rl_progression
from utils import setup_logging, set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM-TAC Experiment Runner")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing the Coq proof dataset")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results and outputs")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B",
                        help="Base LLM model to use")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--rl_iterations", type=int, default=10,
                        help="Number of reinforcement learning iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for training if available")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Run baseline models for comparison")
    parser.add_argument("--ablation_studies", action="store_true",
                        help="Run ablation studies on LLM-TAC components")
    return parser.parse_args()

def main():
    """Main function to run the experiment."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "log.txt")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting LLM-TAC experiment at {datetime.now()}")
    logger.info(f"Arguments: {args}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check for GPU
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up data processor
    data_processor = CoqDataProcessor(args.data_dir)
    train_data, val_data, test_data = data_processor.process_and_split_data()
    logger.info(f"Processed data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
    
    # Initialize models
    contextual_encoder = ContextualEncoder(model_name=args.model_name, device=device)
    tactic_generator = TacticGenerator(model_name=args.model_name, device=device)
    rl_learner = ReinforcementLearner(
        tactic_generator=tactic_generator, 
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Train initial model (supervised fine-tuning)
    logger.info("Starting supervised fine-tuning...")
    training_stats = tactic_generator.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Plot training curves
    plot_training_curve(
        training_stats, 
        os.path.join(args.output_dir, "training_curve.png"),
        "Supervised Fine-tuning Learning Curve"
    )
    
    # Reinforcement learning loop
    logger.info("Starting reinforcement learning...")
    rl_stats = rl_learner.train(
        train_data=train_data,
        val_data=val_data,
        num_iterations=args.rl_iterations,
        batch_size=args.batch_size
    )
    
    # Plot RL progression
    plot_rl_progression(
        rl_stats, 
        os.path.join(args.output_dir, "rl_progression.png"),
        "RL Performance Progression"
    )
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Evaluate LLM-TAC on test set
    logger.info("Evaluating LLM-TAC on test set...")
    llm_tac_results = evaluator.evaluate(
        model=tactic_generator,
        data=test_data,
        contextual_encoder=contextual_encoder
    )
    
    # Run baselines if requested
    baseline_results = {}
    if args.run_baselines:
        logger.info("Running baseline models for comparison...")
        
        # Naive LLM baseline
        naive_llm = NaiveLLM(model_name=args.model_name, device=device)
        naive_results = evaluator.evaluate(
            model=naive_llm,
            data=test_data,
            contextual_encoder=None
        )
        baseline_results["Naive LLM"] = naive_results
        
        # In-Context Learning baseline
        icl_model = ICLModel(model_name=args.model_name, device=device)
        icl_results = evaluator.evaluate(
            model=icl_model,
            data=test_data,
            contextual_encoder=None
        )
        baseline_results["ICL"] = icl_results
        
        # Traditional tactics baseline
        trad_tactics = TraditionalTactics()
        trad_results = evaluator.evaluate(
            model=trad_tactics,
            data=test_data,
            contextual_encoder=None
        )
        baseline_results["Traditional Tactics"] = trad_results
    
    # Run ablation studies if requested
    ablation_results = {}
    if args.ablation_studies:
        logger.info("Running ablation studies...")
        
        # LLM-TAC without RL
        no_rl_generator = TacticGenerator(model_name=args.model_name, device=device)
        no_rl_generator.load_from_pretrained(tactic_generator.pretrained_weights_path)
        no_rl_results = evaluator.evaluate(
            model=no_rl_generator,
            data=test_data,
            contextual_encoder=contextual_encoder
        )
        ablation_results["No RL"] = no_rl_results
        
        # LLM-TAC without retrieval-augmented context
        no_retrieval_encoder = ContextualEncoder(
            model_name=args.model_name, 
            device=device,
            use_retrieval=False
        )
        no_retrieval_results = evaluator.evaluate(
            model=tactic_generator,
            data=test_data,
            contextual_encoder=no_retrieval_encoder
        )
        ablation_results["No Retrieval"] = no_retrieval_results
    
    # Combine and save all results
    all_results = {
        "LLM-TAC": llm_tac_results,
        "Baselines": baseline_results,
        "Ablations": ablation_results
    }
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot comparison of methods
    plot_metrics_comparison(
        results=all_results,
        save_path=os.path.join(args.output_dir, "metrics_comparison.png"),
        title="Performance Comparison of Different Methods"
    )
    
    logger.info(f"Experiment completed at {datetime.now()}")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")