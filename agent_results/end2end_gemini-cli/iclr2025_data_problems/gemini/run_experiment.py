

import torch
import os
import json
import logging
import shutil
from transformers import AdamW, AutoTokenizer

from data_utils import get_data, create_synthetic_dataloader
from models import get_student_model, train_student, evaluate_student, find_hard_examples
from generation import get_generator_model, generate_static_synthetic_data, generate_symbiotic_data, generate_recursive_data
from plotting import plot_results
from report import generate_report

# --- Configuration ---
CONFIG = {
    "student_model_name": "EleutherAI/pythia-160m",
    "generator_model_name": "Qwen/Qwen2-0.5B-Instruct",
    "max_seq_length": 128,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_generations": 3, # Number of co-evolutionary cycles
    "epochs_per_generation": 2,
    "num_hard_examples": 50, # Number of hard examples to find for symbiosis
    "num_synthetic_samples_per_generation": 200, # Number of samples to generate each time
    "max_generation_length": 40, # Max tokens for generator
    "train_dataset_size": 1000,
    "val_dataset_size": 200,
    "test_dataset_size": 500,
    "results_dir": "results",
    "gemini_dir": "gemini"
}

# --- Main Experiment Logic ---
def run_experiment():
    """
    Main function to run the entire suite of experiments.
    """
    # Setup logging
    log_path = "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting experiment...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, student_tokenizer = get_data(CONFIG)
    
    # Initialize results dictionary
    results = {}

    # --- 1. Real Data Upper Bound (Baseline) ---
    logging.info("\n" + "="*50)
    logging.info("Running Baseline: Real Data Upper Bound")
    logging.info("="*50)
    real_data_model = get_student_model(CONFIG["student_model_name"]).to(device)
    optimizer = AdamW(real_data_model.parameters(), lr=CONFIG["learning_rate"])
    results["Real Data Upper Bound"] = {"accuracy": [], "train_loss": []}
    
    # Initial evaluation
    initial_acc, _ = evaluate_student(real_data_model, test_loader, device)
    results["Real Data Upper Bound"]["accuracy"].append(initial_acc)

    for generation in range(CONFIG["num_generations"]):
        logging.info(f"Generation {generation + 1}/{CONFIG['num_generations']}")
        history = train_student(real_data_model, train_loader, optimizer, device, num_epochs=CONFIG["epochs_per_generation"])
        accuracy, _ = evaluate_student(real_data_model, test_loader, device)
        results["Real Data Upper Bound"]["accuracy"].append(accuracy)
        results["Real Data Upper Bound"]["train_loss"].append(history['train_loss'])
    del real_data_model # Free memory

    # --- Load Generator Model (once) ---
    generator_model, generator_tokenizer = get_generator_model(CONFIG["generator_model_name"], device)

    # --- 2. Static Synthetic Data (Baseline) ---
    logging.info("\n" + "="*50)
    logging.info("Running Baseline: Static Synthetic Data")
    logging.info("="*50)
    static_model = get_student_model(CONFIG["student_model_name"]).to(device)
    optimizer = AdamW(static_model.parameters(), lr=CONFIG["learning_rate"])
    results["Static Synthetic"] = {"accuracy": [], "train_loss": []}
    
    # Generate a static dataset
    static_texts, static_labels = generate_static_synthetic_data(
        generator_model, generator_tokenizer, 
        CONFIG["num_synthetic_samples_per_generation"] * CONFIG["num_generations"], 
        CONFIG
    )
    static_loader = create_synthetic_dataloader(static_texts, static_labels, student_tokenizer, CONFIG)
    
    # Initial evaluation
    initial_acc, _ = evaluate_student(static_model, test_loader, device)
    results["Static Synthetic"]["accuracy"].append(initial_acc)

    for generation in range(CONFIG["num_generations"]):
        logging.info(f"Generation {generation + 1}/{CONFIG['num_generations']}")
        history = train_student(static_model, static_loader, optimizer, device, num_epochs=CONFIG["epochs_per_generation"])
        accuracy, _ = evaluate_student(static_model, test_loader, device)
        results["Static Synthetic"]["accuracy"].append(accuracy)
        results["Static Synthetic"]["train_loss"].append(history['train_loss'])
    del static_model # Free memory

    # --- 3. Recursive Collapse (Baseline) ---
    logging.info("\n" + "="*50)
    logging.info("Running Baseline: Recursive Collapse")
    logging.info("="*50)
    recursive_model = get_student_model(CONFIG["student_model_name"]).to(device)
    optimizer = AdamW(recursive_model.parameters(), lr=CONFIG["learning_rate"])
    results["Recursive Collapse"] = {"accuracy": [], "train_loss": []}

    # Initial evaluation
    initial_acc, _ = evaluate_student(recursive_model, test_loader, device)
    results["Recursive Collapse"]["accuracy"].append(initial_acc)

    for generation in range(CONFIG["num_generations"]):
        logging.info(f"Generation {generation + 1}/{CONFIG['num_generations']}")
        # Generate data using the student model itself
        recursive_texts, recursive_labels = generate_recursive_data(
            recursive_model, student_tokenizer, CONFIG["num_synthetic_samples_per_generation"], CONFIG
        )
        if not recursive_texts:
            logging.warning("Recursive generation produced no text. Stopping this baseline.")
            results["Recursive Collapse"]["accuracy"].append(results["Recursive Collapse"]["accuracy"][-1]) # flatline
            continue
            
        recursive_loader = create_synthetic_dataloader(recursive_texts, recursive_labels, student_tokenizer, CONFIG)
        history = train_student(recursive_model, recursive_loader, optimizer, device, num_epochs=CONFIG["epochs_per_generation"])
        accuracy, _ = evaluate_student(recursive_model, test_loader, device)
        results["Recursive Collapse"]["accuracy"].append(accuracy)
        results["Recursive Collapse"]["train_loss"].append(history['train_loss'])
    del recursive_model # Free memory

    # --- 4. Generative Data Symbiosis (Proposed Method) ---
    logging.info("\n" + "="*50)
    logging.info("Running Proposed Method: Generative Symbiosis")
    logging.info("="*50)
    symbiotic_model = get_student_model(CONFIG["student_model_name"]).to(device)
    optimizer = AdamW(symbiotic_model.parameters(), lr=CONFIG["learning_rate"])
    results["Generative Symbiosis"] = {"accuracy": [], "train_loss": []}

    # Initial evaluation
    initial_acc, _ = evaluate_student(symbiotic_model, test_loader, device)
    results["Generative Symbiosis"]["accuracy"].append(initial_acc)

    for generation in range(CONFIG["num_generations"]):
        logging.info(f"Generation {generation + 1}/{CONFIG['num_generations']}")
        # Find hard examples using the validation set
        hard_examples = find_hard_examples(symbiotic_model, val_loader, student_tokenizer, device, num_examples=CONFIG["num_hard_examples"])
        # Generate new data based on these hard examples
        symbiotic_texts, symbiotic_labels = generate_symbiotic_data(generator_model, generator_tokenizer, hard_examples, CONFIG)
        
        if not symbiotic_texts:
            logging.warning("Symbiotic generation produced no text. Stopping this method.")
            results["Generative Symbiosis"]["accuracy"].append(results["Generative Symbiosis"]["accuracy"][-1]) # flatline
            continue

        symbiotic_loader = create_synthetic_dataloader(symbiotic_texts, symbiotic_labels, student_tokenizer, CONFIG)
        history = train_student(symbiotic_model, symbiotic_loader, optimizer, device, num_epochs=CONFIG["epochs_per_generation"])
        accuracy, _ = evaluate_student(symbiotic_model, test_loader, device)
        results["Generative Symbiosis"]["accuracy"].append(accuracy)
        results["Generative Symbiosis"]["train_loss"].append(history['train_loss'])
    del symbiotic_model, generator_model # Free memory

    # --- Finalization ---
    logging.info("\n" + "="*50)
    logging.info("Experiment finished. Generating results...")
    logging.info("="*50)

    # Save raw results
    results_path = os.path.join(CONFIG["gemini_dir"], "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Raw results saved to {results_path}")

    # Generate plots
    fig_paths = plot_results(results_path, CONFIG["gemini_dir"])
    
    # Generate final report
    report_path = os.path.join(CONFIG["gemini_dir"], "results.md")
    generate_report(results_path, report_path, fig_paths)

    # Move final files to results directory
    final_results_dir = CONFIG["results_dir"]
    shutil.move(log_path, os.path.join(final_results_dir, "log.txt"))
    shutil.move(report_path, os.path.join(final_results_dir, "results.md"))
    for fig_path in fig_paths:
        if os.path.exists(fig_path):
            shutil.move(fig_path, os.path.join(final_results_dir, os.path.basename(fig_path)))

    # Clean up large cache files
    logging.info("Cleaning up Hugging Face cache...")
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            logging.info(f"Removed dataset cache: {cache_dir}")
        except OSError as e:
            logging.error(f"Error removing cache directory {cache_dir}: {e.strerror}")

    logging.info("="*50)
    logging.info(f"Process complete. Final report and figures are in the '{final_results_dir}' directory.")
    logging.info("="*50)


if __name__ == "__main__":
    # Ensure the gemini and results directories exist
    os.makedirs(CONFIG["gemini_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    run_experiment()
