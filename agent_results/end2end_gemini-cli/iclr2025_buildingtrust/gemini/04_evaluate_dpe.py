

import os
import pandas as pd
import torch
import json
import time
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "dynosafe_benchmark.csv"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct" # Must be the same as in the training script
ADAPTER_DIR = "dpe_model_adapter"
OUTPUT_FILE = "dpe_results.json"

def create_inference_prompt(example):
    """Creates the structured prompt for inference, stopping before the verdict."""
    return f"""### Policy:
{example['policy_text']}

### LLM Response:
{example['response']}

### Verdict:
"""

def main():
    """Main function to evaluate the fine-tuned DPE model."""
    print("Starting DPE model evaluation...")

    # --- 1. Load and Prepare Dataset ---
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please run '01_generate_dataset.py' first.")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({}, f)
        return

    dataset = Dataset.from_pandas(df)
    # Use the same split as in training to get the validation set
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = train_test_split['test']
    
    ground_truth = test_dataset['ground_truth']
    
    print(f"Loaded {len(test_dataset)} examples for evaluation.")

    # --- 2. Load Fine-Tuned Model ---
    if not os.path.exists(ADAPTER_DIR):
        print(f"Error: Adapter directory '{ADAPTER_DIR}' not found.")
        print("Please run '03_finetune_dpe.py' first.")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({}, f)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = "auto"
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    
    # Apply the LoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload() # Merge adapter for faster inference
    model.eval()

    print("Fine-tuned DPE model loaded successfully.")

    # --- 3. Run Inference ---
    predictions = []
    total_latency = 0
    
    print("\nRunning inference on the test set...")
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating DPE"):
            prompt = create_inference_prompt(example)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            
            # Generate output. We only need a few tokens to get the verdict.
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id
            )
            
            total_latency += time.time() - start_time
            
            # Decode the generated tokens and extract the verdict
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # The generated text includes the prompt, so we extract the part after it
            verdict_part = output_text[len(prompt):].strip().upper()
            
            if "ALLOW" in verdict_part:
                predictions.append("ALLOW")
            elif "BLOCK" in verdict_part:
                predictions.append("BLOCK")
            else:
                # If the model gives an unexpected output, default to a safe choice
                predictions.append("BLOCK")

    # --- 4. Evaluate and Save Results ---
    avg_latency = total_latency / len(test_dataset) if test_dataset else 0
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, pos_label='BLOCK')

    results = {
        "dpe_model": {
            "accuracy": accuracy,
            "f1_score_block": f1,
            "latency_ms": avg_latency * 1000,
            "predictions": predictions
        },
        "ground_truth": list(ground_truth)
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

    print("\nDPE model evaluation complete.")
    print(f"DPE Model | Accuracy: {accuracy:.4f} | F1 (Block): {f1:.4f} | Latency: {avg_latency*1000:.2f} ms")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

