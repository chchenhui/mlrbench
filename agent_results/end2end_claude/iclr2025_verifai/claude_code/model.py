"""
Model and learning components for the VERIL framework.

This module implements the Recursive Improvement Learning (RIL) component
of the VERIL framework, which uses verification feedback to improve the
model's code generation capabilities.
"""

import os
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
import tempfile

import torch
import numpy as np
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
import anthropic
import openai

from config import (
    MODELS, 
    TRAINING, 
    CHECKPOINTS_DIR, 
    ERROR_TAXONOMY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY
)
from data import CodeProblem
from verification import VerificationResult, ErrorToExplanationConverter
from utils import logger, time_function, extract_code_from_response, save_json, load_json


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LLMCodeGenerator:
    """Base class for LLM code generators."""
    
    def __init__(
        self, 
        model_name: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_verification: bool = False,
        verification_types: List[str] = ["static", "dynamic"],
    ):
        """
        Initialize the code generator.
        
        Args:
            model_name: Name of the model
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation
            use_verification: Whether to use verification feedback
            verification_types: Types of verification to use
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_verification = use_verification
        self.verification_types = verification_types
        
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
    
    def generate(self, prompt: str, n_samples: int = 1) -> List[str]:
        """
        Generate code samples.
        
        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate
            
        Returns:
            List of generated code samples
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def learn_from_verification(self, verification_results: List[VerificationResult]) -> None:
        """
        Learn from verification results.
        
        Args:
            verification_results: List of verification results
        """
        if not self.use_verification:
            logger.info("Verification-based learning is disabled for this model")
            return
        
        logger.info(f"Learning from {len(verification_results)} verification results")
        # This is a placeholder. Subclasses should implement the actual learning logic.


class OpenSourceLLMGenerator(LLMCodeGenerator):
    """Code generator using open-source LLMs via Hugging Face."""
    
    def __init__(
        self, 
        model_name: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_verification: bool = False,
        verification_types: List[str] = ["static", "dynamic"],
        device=None,
        quantization: Optional[str] = "4bit",  # "4bit", "8bit", or None
    ):
        """
        Initialize the open-source LLM generator.
        
        Args:
            model_name: Name of the model on Hugging Face
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation
            use_verification: Whether to use verification feedback
            verification_types: Types of verification to use
            device: PyTorch device
            quantization: Quantization level ("4bit", "8bit", or None)
        """
        super().__init__(
            model_name, 
            max_new_tokens, 
            temperature, 
            use_verification, 
            verification_types
        )
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = quantization
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        logger.info(f"Loading model {model_name} with {quantization} quantization...")
        if quantization == "4bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
        elif quantization == "8bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        logger.info(f"Model loaded on device: {self.device}")
    
    @time_function
    def generate(self, prompt: str, n_samples: int = 1) -> List[str]:
        """
        Generate code samples using the open-source LLM.
        
        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate
            
        Returns:
            List of generated code samples
        """
        logger.info(f"Generating {n_samples} code samples...")
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        samples = []
        for i in range(n_samples):
            # Generate with different random seeds for diversity
            set_seed(i + 42)
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the generated code part
            generated_code = generated_text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
            
            samples.append(generated_code)
        
        return samples
    
    def learn_from_verification(self, verification_results: List[VerificationResult]) -> None:
        """
        Learn from verification results using LoRA fine-tuning.
        
        Args:
            verification_results: List of verification results
        """
        if not self.use_verification or len(verification_results) == 0:
            return
        
        logger.info(f"Learning from {len(verification_results)} verification results using LoRA fine-tuning")
        
        # Prepare training examples
        train_examples = []
        
        for result in verification_results:
            if result.passed:
                # For correct solutions, just add them to the training set
                train_examples.append({
                    "text": result.code,
                    "label": 1,  # Correct solution
                })
            else:
                # For incorrect solutions, generate explanations
                explanation, _ = ErrorToExplanationConverter.generate_explanation(result)
                
                # Add incorrect solution with explanation
                train_examples.append({
                    "text": f"Incorrect solution:\n{result.code}\n\nExplanation:\n{explanation}",
                    "label": 0,  # Incorrect solution
                })
        
        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_dict({
            "text": [example["text"] for example in train_examples],
            "label": [example["label"] for example in train_examples],
        })
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Prepare model for training
        if self.quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        model = get_peft_model(self.model, lora_config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=CHECKPOINTS_DIR / f"{self.model_name.split('/')[-1]}_veril",
            learning_rate=TRAINING["learning_rate"],
            num_train_epochs=TRAINING["max_epochs"],
            per_device_train_batch_size=TRAINING["batch_size"],
            gradient_accumulation_steps=TRAINING["gradient_accumulation_steps"],
            warmup_ratio=TRAINING["warmup_ratio"],
            weight_decay=TRAINING["weight_decay"],
            logging_dir=CHECKPOINTS_DIR / "logs",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        try:
            trainer.train()
            
            # Save adapters
            model.save_pretrained(CHECKPOINTS_DIR / f"{self.model_name.split('/')[-1]}_veril_final")
            logger.info("Model fine-tuning completed and saved successfully")
            
            # Update the model
            self.model = model
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")


class OpenAICodeGenerator(LLMCodeGenerator):
    """Code generator using OpenAI API."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_verification: bool = False,
        verification_types: List[str] = ["static", "dynamic"],
    ):
        """
        Initialize the OpenAI code generator.
        
        Args:
            model_name: Name of the OpenAI model
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation
            use_verification: Whether to use verification feedback
            verification_types: Types of verification to use
        """
        super().__init__(
            model_name, 
            max_new_tokens, 
            temperature, 
            use_verification, 
            verification_types
        )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"Initialized OpenAI client")
    
    @time_function
    def generate(self, prompt: str, n_samples: int = 1) -> List[str]:
        """
        Generate code samples using the OpenAI API.
        
        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate
            
        Returns:
            List of generated code samples
        """
        logger.info(f"Generating {n_samples} code samples using OpenAI API...")
        
        samples = []
        
        system_prompt = """You are an expert Python programmer. Generate Python code to solve the given problem. 
Your response should be complete, correct, and efficiently implemented Python code. Do not include explanations, 
just the code solution.

Submit your response as a Python code block:
```python
def solution():
    # Your code here
```
"""
        
        for i in range(n_samples):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    n=1,
                )
                
                generated_text = response.choices[0].message.content
                samples.append(generated_text)
                
                # Add a small delay between requests
                if i < n_samples - 1:
                    time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating with OpenAI API: {str(e)}")
                # Return empty string on error
                samples.append("")
        
        return samples
    
    def learn_from_verification(self, verification_results: List[VerificationResult]) -> None:
        """
        Learn from verification results.
        
        For API-based models, we simulate learning by including verification feedback
        in the prompt for future generations.
        
        Args:
            verification_results: List of verification results
        """
        if not self.use_verification or len(verification_results) == 0:
            return
        
        logger.info(f"Simulating learning from {len(verification_results)} verification results for API model")
        
        # For API models, we don't actually learn, but we could store the
        # verification results for use in future generations
        pass


class AnthropicCodeGenerator(LLMCodeGenerator):
    """Code generator using Anthropic API."""
    
    def __init__(
        self, 
        model_name: str = "claude-3-sonnet-20240229", 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        use_verification: bool = False,
        verification_types: List[str] = ["static", "dynamic"],
    ):
        """
        Initialize the Anthropic code generator.
        
        Args:
            model_name: Name of the Anthropic model
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation
            use_verification: Whether to use verification feedback
            verification_types: Types of verification to use
        """
        super().__init__(
            model_name, 
            max_new_tokens, 
            temperature, 
            use_verification, 
            verification_types
        )
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info(f"Initialized Anthropic client")
    
    @time_function
    def generate(self, prompt: str, n_samples: int = 1) -> List[str]:
        """
        Generate code samples using the Anthropic API.
        
        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate
            
        Returns:
            List of generated code samples
        """
        logger.info(f"Generating {n_samples} code samples using Anthropic API...")
        
        samples = []
        
        system_prompt = """You are an expert Python programmer. Generate Python code to solve the given problem. 
Your response should be complete, correct, and efficiently implemented Python code. Do not include explanations, 
just the code solution.

Submit your response as a Python code block:
```python
def solution():
    # Your code here
```
"""
        
        for i in range(n_samples):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                
                generated_text = response.content[0].text
                samples.append(generated_text)
                
                # Add a small delay between requests
                if i < n_samples - 1:
                    time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating with Anthropic API: {str(e)}")
                # Return empty string on error
                samples.append("")
        
        return samples
    
    def learn_from_verification(self, verification_results: List[VerificationResult]) -> None:
        """
        Learn from verification results.
        
        For API-based models, we simulate learning by including verification feedback
        in the prompt for future generations.
        
        Args:
            verification_results: List of verification results
        """
        if not self.use_verification or len(verification_results) == 0:
            return
        
        logger.info(f"Simulating learning from {len(verification_results)} verification results for API model")
        
        # For API models, we don't actually learn, but we could store the
        # verification results for use in future generations
        pass


class RecursiveImprovementLearning:
    """
    Recursive Improvement Learning (RIL) component of the VERIL framework.
    
    This component implements a multi-tiered learning strategy that uses
    error explanations and remediation examples to improve the model's
    code generation capabilities.
    """
    
    def __init__(
        self,
        generator: LLMCodeGenerator,
        verification_types: List[str] = ["static", "dynamic"],
        num_iterations: int = 3,
    ):
        """
        Initialize the recursive improvement learning component.
        
        Args:
            generator: LLM code generator
            verification_types: Types of verification to use
            num_iterations: Number of recursive improvement iterations
        """
        self.generator = generator
        self.verification_types = verification_types
        self.num_iterations = num_iterations
        
        # Import verification module here to avoid circular import
        from verification import VerificationIntegrationLayer
        self.verification_layer = VerificationIntegrationLayer(verification_types)
        
        logger.info(f"Initialized Recursive Improvement Learning with {num_iterations} iterations")
    
    def _apply_error_focused_fine_tuning(self, problems: List[CodeProblem]) -> None:
        """
        Apply error-focused fine-tuning.
        
        Args:
            problems: List of code problems
        """
        logger.info("Applying error-focused fine-tuning...")
        
        # Generate code for each problem
        all_verification_results = []
        
        for problem in problems:
            # Generate code
            prompt = f"""Write a Python function to solve the following problem:

{problem.prompt}

Your solution should be complete and correct.
"""
            
            generations = self.generator.generate(prompt, n_samples=3)
            
            # Verify generations
            for generation in generations:
                result = self.verification_layer.verify(generation, problem.test_cases)
                all_verification_results.append(result)
        
        # Learn from verification results
        self.generator.learn_from_verification(all_verification_results)
    
    def _apply_contrastive_learning(self, problems: List[CodeProblem]) -> None:
        """
        Apply contrastive learning.
        
        Args:
            problems: List of code problems
        """
        # Not implemented in this simplified version
        pass
    
    def _apply_priority_weighted_learning(self, problems: List[CodeProblem]) -> None:
        """
        Apply priority weighted learning.
        
        Args:
            problems: List of code problems
        """
        # Not implemented in this simplified version
        pass
    
    def _apply_iterative_refinement(self, problems: List[CodeProblem]) -> Dict[str, Any]:
        """
        Apply iterative refinement.
        
        Args:
            problems: List of code problems
            
        Returns:
            Dictionary of learning metrics
        """
        logger.info("Applying iterative refinement...")
        
        metrics = {
            "iterations": [],
            "pass_rate": [],
            "error_rate": [],
            "veri_pass_rate": [],
        }
        
        for iteration in range(self.num_iterations):
            logger.info(f"Starting iteration {iteration+1}/{self.num_iterations}")
            
            # Apply error-focused fine-tuning
            self._apply_error_focused_fine_tuning(problems)
            
            # Evaluate the model
            pass_count = 0
            error_count = 0
            veri_pass_count = 0
            
            for problem in problems:
                prompt = f"""Write a Python function to solve the following problem:

{problem.prompt}

Your solution should be complete and correct.
"""
                
                # Generate a solution
                generation = self.generator.generate(prompt, n_samples=1)[0]
                
                # Verify the solution
                result = self.verification_layer.verify(generation, problem.test_cases)
                
                if result.passed:
                    pass_count += 1
                    veri_pass_count += 1
                elif len(result.errors) == 0:
                    veri_pass_count += 1
                
                error_count += len(result.errors)
            
            # Record metrics
            metrics["iterations"].append(iteration + 1)
            metrics["pass_rate"].append(pass_count / len(problems))
            metrics["error_rate"].append(error_count / len(problems))
            metrics["veri_pass_rate"].append(veri_pass_count / len(problems))
            
            logger.info(f"Iteration {iteration+1} metrics: "
                       f"pass_rate={metrics['pass_rate'][-1]:.4f}, "
                       f"error_rate={metrics['error_rate'][-1]:.4f}, "
                       f"veri_pass_rate={metrics['veri_pass_rate'][-1]:.4f}")
        
        return metrics
    
    def train(self, problems: List[CodeProblem]) -> Dict[str, Any]:
        """
        Train the model using recursive improvement learning.
        
        Args:
            problems: List of code problems
            
        Returns:
            Dictionary of learning metrics
        """
        logger.info(f"Starting recursive improvement learning with {len(problems)} problems")
        
        # Apply the multi-tiered learning strategy
        metrics = self._apply_iterative_refinement(problems)
        
        return metrics


def create_model(
    model_config: Dict[str, Any],
    device=None,
) -> LLMCodeGenerator:
    """
    Create a code generator model based on configuration.
    
    Args:
        model_config: Model configuration
        device: PyTorch device
        
    Returns:
        LLMCodeGenerator instance
    """
    model_name = model_config["name"]
    
    # Determine model type based on name
    if "gpt" in model_name.lower():
        logger.info(f"Creating OpenAI model: {model_name}")
        return OpenAICodeGenerator(
            model_name=model_name,
            max_new_tokens=model_config.get("max_new_tokens", 512),
            temperature=model_config.get("temperature", 0.7),
            use_verification=model_config.get("use_verification", False),
            verification_types=model_config.get("verification_types", ["static", "dynamic"]),
        )
    elif "claude" in model_name.lower():
        logger.info(f"Creating Anthropic model: {model_name}")
        return AnthropicCodeGenerator(
            model_name=model_name,
            max_new_tokens=model_config.get("max_new_tokens", 512),
            temperature=model_config.get("temperature", 0.7),
            use_verification=model_config.get("use_verification", False),
            verification_types=model_config.get("verification_types", ["static", "dynamic"]),
        )
    else:
        logger.info(f"Creating open-source model: {model_name}")
        return OpenSourceLLMGenerator(
            model_name=model_name,
            max_new_tokens=model_config.get("max_new_tokens", 512),
            temperature=model_config.get("temperature", 0.7),
            use_verification=model_config.get("use_verification", False),
            verification_types=model_config.get("verification_types", ["static", "dynamic"]),
            device=device,
            quantization="4bit",  # Use 4-bit quantization by default
        )