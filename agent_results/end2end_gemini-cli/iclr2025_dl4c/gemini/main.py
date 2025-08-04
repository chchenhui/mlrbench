import os
import json
import pandas as pd
from src.data_loader import get_human_eval_subset
from src.models import get_base_model
from src.train import sft_fine_tune
from src.evaluate import evaluate_model, plot_results

def run_sft_baseline(cache_dir, results_dir, subset_size=10):
    """
    Runs the SFT baseline experiment.
    """
    print("Running SFT Baseline Experiment...")
    
    # Load data
    dataset = get_human_eval_subset(cache_dir, subset_size=subset_size)
    
    # Load model
    model, tokenizer = get_base_model()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Fine-tune the model
    sft_model = sft_fine_tune(model, tokenizer, dataset)
    
    # Evaluate the model
    evaluation_results = evaluate_model(sft_model, tokenizer, dataset, results_dir)
    
    # Plot the results
    plot_results(evaluation_results, results_dir)
    
    print("SFT Baseline Experiment Complete.")

def main():
    """
    Main function to run all experiments.
    """
    human_eval_cache_dir = "data/human_eval"
    
    # SFT Baseline
    sft_results_dir = "gemini/results/sft_baseline"
    if not os.path.exists(sft_results_dir):
        os.makedirs(sft_results_dir)
    run_sft_baseline(human_eval_cache_dir, sft_results_dir)
    

if __name__ == '__main__':
    main()