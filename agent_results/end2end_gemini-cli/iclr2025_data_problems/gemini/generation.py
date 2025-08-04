

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import bitsandbytes
from accelerate import Accelerator

# Mapping from label index to text for ag_news
LABEL_TO_TEXT = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def get_generator_model(model_name, device):
    """
    Loads the generator model and tokenizer.
    Uses 4-bit quantization to save memory.
    """
    logging.info(f"Loading generator model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=None, # Disabled for now to avoid potential issues
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Generator model loaded.")
    return model, tokenizer

def generate_static_synthetic_data(generator_model, generator_tokenizer, num_samples, config):
    """
    Generates a fixed dataset of synthetic examples.
    """
    logging.info(f"Generating {num_samples} static synthetic samples...")
    synthetic_texts = []
    labels = []
    
    # Use a pipeline for easier generation
    text_generator = pipeline(
        "text-generation",
        model=generator_model,
        tokenizer=generator_tokenizer,
        max_new_tokens=config["max_generation_length"]
    )

    for i in range(num_samples):
        label_idx = i % len(LABEL_TO_TEXT)
        label_name = LABEL_TO_TEXT[label_idx]
        prompt = f"Write a short news article headline about {label_name}."
        
        try:
            generated = text_generator(prompt, num_return_sequences=1, pad_token_id=generator_tokenizer.eos_token_id)
            text = generated[0]['generated_text'].replace(prompt, "").strip()
            if text:
                synthetic_texts.append(text)
                labels.append(label_idx)
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            # Add a placeholder if generation fails
            synthetic_texts.append(f"Placeholder for {label_name}")
            labels.append(label_idx)

        if (i + 1) % 50 == 0:
            logging.info(f"Generated {i + 1}/{num_samples} samples.")

    return synthetic_texts, labels

def generate_symbiotic_data(generator_model, generator_tokenizer, hard_examples, config):
    """
    Generates synthetic data based on hard examples identified by the Student.
    """
    logging.info(f"Generating symbiotic data based on {len(hard_examples)} hard examples...")
    synthetic_texts = []
    labels = []

    text_generator = pipeline(
        "text-generation",
        model=generator_model,
        tokenizer=generator_tokenizer,
        max_new_tokens=config["max_generation_length"]
    )

    for example in hard_examples:
        text_input = example['text']
        label_idx = example['label']
        label_name = LABEL_TO_TEXT[label_idx]

        prompt = f"The following news headline was hard for a model to classify: '{text_input}'. " \
                 f"To help it learn, write a new, different headline about '{label_name}' that captures a similar idea."

        try:
            generated = text_generator(prompt, num_return_sequences=1, pad_token_id=generator_tokenizer.eos_token_id)
            text = generated[0]['generated_text'].replace(prompt, "").strip()
            if text:
                synthetic_texts.append(text)
                labels.append(label_idx)
        except Exception as e:
            logging.error(f"Error during symbiotic generation: {e}")
            synthetic_texts.append(f"Placeholder for {label_name}")
            labels.append(label_idx)

    logging.info(f"Generated {len(synthetic_texts)} symbiotic samples.")
    return synthetic_texts, labels

def generate_recursive_data(student_model, student_tokenizer, num_samples, config):
    """
    Generates data using the student model itself (for the recursive collapse baseline).
    """
    logging.info(f"Generating {num_samples} recursive samples using the student model...")
    synthetic_texts = []
    labels = []

    # Ensure model is in eval mode and on the correct device
    student_model.eval()
    device = next(student_model.parameters()).device
    
    text_generator = pipeline(
        "text-generation",
        model=student_model,
        tokenizer=student_tokenizer,
        device=device,
        max_new_tokens=config["max_generation_length"]
    )

    for i in range(num_samples):
        label_idx = i % len(LABEL_TO_TEXT)
        label_name = LABEL_TO_TEXT[label_idx]
        prompt = f"News about {label_name}:"
        
        try:
            generated = text_generator(prompt, num_return_sequences=1, pad_token_id=student_tokenizer.eos_token_id)
            text = generated[0]['generated_text'].replace(prompt, "").strip()
            if text:
                synthetic_texts.append(text)
                labels.append(label_idx)
        except Exception as e:
            logging.error(f"Error during recursive generation: {e}")
            synthetic_texts.append(f"Placeholder for {label_name}")
            labels.append(label_idx)

    return synthetic_texts, labels

