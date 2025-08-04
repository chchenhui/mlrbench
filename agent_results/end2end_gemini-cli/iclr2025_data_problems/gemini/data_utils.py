
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data(config):
    """
    Loads the ag_news dataset, tokenizes it, and creates DataLoaders.
    """
    logging.info("Loading ag_news dataset...")
    dataset = load_dataset("ag_news")
    
    tokenizer = AutoTokenizer.from_pretrained(config["student_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Tokenizing and preparing datasets...")
    
    # Use a smaller subset for faster experimentation
    train_size = config.get("train_dataset_size", 1000)
    val_size = config.get("val_dataset_size", 200)
    test_size = config.get("test_dataset_size", 500)

    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
    # This validation set is for finding hard examples for the generator
    validation_dataset = dataset["train"].shuffle(seed=24).select(range(train_size, train_size + val_size))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(test_size))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["max_seq_length"])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])

    logging.info(f"Data loaded. Train size: {len(train_dataset)}, Val size: {len(validation_dataset)}, Test size: {len(test_dataset)}")
    
    return train_dataloader, validation_dataloader, test_dataloader, tokenizer

def create_synthetic_dataloader(synthetic_texts, labels, tokenizer, config):
    """
    Creates a DataLoader from synthetic text data.
    """
    encodings = tokenizer(synthetic_texts, padding="max_length", truncation=True, max_length=config["max_seq_length"], return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    return DataLoader(dataset, batch_size=config["batch_size"])

if __name__ == '__main__':
    # For testing the data loading process
    config = {
        "student_model_name": "EleutherAI/pythia-160m",
        "max_seq_length": 128,
        "batch_size": 8,
        "train_dataset_size": 100,
        "val_dataset_size": 50,
        "test_dataset_size": 50
    }
    train_loader, val_loader, test_loader, _ = get_data(config)
    logging.info(f"Train batches: {len(train_loader)}")
    logging.info(f"Validation batches: {len(val_loader)}")
    logging.info(f"Test batches: {len(test_loader)}")
    
    for batch in train_loader:
        logging.info(f"Batch input_ids shape: {batch[0].shape}")
        break
