

import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

# --- Configuration ---
INPUT_FILE = "dynosafe_benchmark.csv"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct" # Using a smaller model for faster execution
ADAPTER_OUTPUT_DIR = "dpe_model_adapter"
TRAINING_HISTORY_FILE = "training_history.json"
NUM_EPOCHS = 3
BATCH_SIZE = 2 # Use a small batch size
LEARNING_RATE = 2e-4

def create_prompt(example):
    """Creates the structured prompt for fine-tuning."""
    prompt = f"""### Policy:
{example['policy_text']}

### LLM Response:
{example['response']}

### Verdict:
{example['ground_truth']}"""
    return {"text": prompt}

def main():
    """Main function to fine-tune the DPE model."""
    print(f"Starting DPE model fine-tuning using model: {MODEL_ID}")

    # --- 1. Load and Prepare Dataset ---
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please run '01_generate_dataset.py' first.")
        # Create empty files to avoid breaking the pipeline
        os.makedirs(ADAPTER_OUTPUT_DIR, exist_ok=True)
        with open(TRAINING_HISTORY_FILE, 'w') as f:
            json.dump([], f)
        return

    dataset = Dataset.from_pandas(df)
    
    # Apply prompt formatting
    dataset = dataset.map(create_prompt)
    
    # Split dataset (e.g., 80% train, 20% validation)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_split = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })
    
    print(f"\nDataset prepared: {len(dataset_split['train'])} training examples, {len(dataset_split['validation'])} validation examples.")

    # --- 2. Load Model and Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use 4-bit quantization to save memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = "auto"
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map=device_map, # Automatically use GPU if available
    )
    
    model = prepare_model_for_kbit_training(model)

    # --- 3. Configure LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Set up Trainer ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset_split.map(tokenize_function, batched=True, remove_columns=["text", "policy_name", "policy_text", "response", "ground_truth"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results_temp",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none", # Disable wandb/tensorboard reporting for this script
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. Train ---
    print("\nStarting model training...")
    trainer.train()
    print("Training complete.")

    # --- 6. Save Artifacts ---
    print(f"Saving LoRA adapter to {ADAPTER_OUTPUT_DIR}")
    trainer.save_model(ADAPTER_OUTPUT_DIR)
    
    # Save training history for plotting
    with open(TRAINING_HISTORY_FILE, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=4)
    print(f"Training history saved to {TRAINING_HISTORY_FILE}")

if __name__ == "__main__":
    main()

