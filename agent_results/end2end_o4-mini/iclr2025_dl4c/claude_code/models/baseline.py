import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

class StaticCodeT5Plus:
    """
    Baseline model: Static CodeT5+ without adaptation to user feedback.
    
    This model loads a pre-trained CodeT5+ and uses it for code generation without
    any adaptive fine-tuning based on developer feedback.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5p-220m-py",  # Smaller model for faster experiments
        device: torch.device = torch.device('cpu'),
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special handling for pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def train(
        self,
        train_data: Dataset,
        valid_data: Dataset,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        output_dir: str = "./models/baseline"
    ):
        """
        Fine-tune the model on the given dataset.
        
        For the baseline model, this is just standard fine-tuning without
        any reinforcement learning or adaptation.
        """
        # Prepare for training
        self.model.train()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            report_to=None,  # Don't report to any tracking system
        )
        
        # Prepare data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We want to train a causal LM, not a masked LM
        )
        
        # Prepare trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(f"{output_dir}/final")
        self.tokenizer.save_pretrained(f"{output_dir}/final")
        
        # Set model back to evaluation mode
        self.model.eval()
    
    def generate_suggestion(
        self,
        context: str,
        developer_profile = None,  # Unused in baseline but kept for API consistency
        device: Optional[torch.device] = None
    ) -> str:
        """
        Generate code suggestion for the given context.

        For the baseline model, we ignore the developer profile.
        """
        device = device or self.device

        # Prepare input - T5 expects a prompt
        inputs = self.tokenizer(context, return_tensors="pt").to(device)

        # Generate suggestion
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode suggestion
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return suggestion