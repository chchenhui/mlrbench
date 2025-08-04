
import torch
from transformers import Trainer, TrainingArguments
from src.models import get_base_model
from src.data_loader import get_human_eval_subset
from tqdm import tqdm

def sft_fine_tune(model, tokenizer, dataset):
    """
    Performs Supervised Fine-Tuning (SFT) on a given model.
    """
    
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Add labels to the tokenized dataset
    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

    training_args = TrainingArguments(
        output_dir="./results/sft_model",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    return model
