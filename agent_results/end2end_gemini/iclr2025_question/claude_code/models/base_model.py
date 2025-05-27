"""
Base model implementations for the AUG-RAG system.
"""

import os
import torch
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all models in the AUG-RAG system."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        max_length: int = 512,
        **kwargs
    ):
        """
        Initialize the base model.
        
        Args:
            model_name: The name of the model to load (HF model ID or local path).
            device: The device to use for inference (cpu, cuda, etc.).
            max_length: The maximum sequence length.
            **kwargs: Additional model initialization arguments.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model(**kwargs)
        logger.info(f"Initialized {self.__class__.__name__} with {model_name} on {self.device}")
    
    def initialize_model(self, **kwargs):
        """Initialize the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode by default
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {e}")
            raise
    
    def generate(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """
        Generate text based on input prompts.
        
        Args:
            inputs: The input prompt(s).
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional generation parameters.
        
        Returns:
            The generated text.
        """
        # Handle single input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Tokenize inputs
        input_encodings = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove the input prompt from the output
        cleaned_outputs = []
        for i, output in enumerate(decoded_outputs):
            # Find and remove the input prompt
            input_text = self.tokenizer.decode(input_encodings.input_ids[i], skip_special_tokens=True)
            if output.startswith(input_text):
                output = output[len(input_text):].strip()
            cleaned_outputs.append(output)
        
        return cleaned_outputs
    
    def get_logits(
        self,
        inputs: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Get logits for the next token prediction given the input.
        
        Args:
            inputs: The input prompt(s).
        
        Returns:
            The logits tensor.
        """
        # Handle single input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Tokenize inputs
        input_encodings = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask
            )
        
        # Return logits
        return outputs.logits
    
    def save(self, path: str):
        """
        Save the model and tokenizer.
        
        Args:
            path: The path to save the model to.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str):
        """
        Load the model and tokenizer.
        
        Args:
            path: The path to load the model from.
        """
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Loaded model from {path}")


class APIBasedModel(BaseModel):
    """Base class for API-based models (e.g., GPT-4, Claude)."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = None,
        max_length: int = 512,
        **kwargs
    ):
        """
        Initialize the API-based model.
        
        Args:
            model_name: The name of the model to use.
            api_key: The API key for the model provider.
            max_length: The maximum sequence length.
            **kwargs: Additional model initialization arguments.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.initialize_client(**kwargs)
        logger.info(f"Initialized {self.__class__.__name__} with {model_name}")
    
    def initialize_client(self, **kwargs):
        """Initialize the API client."""
        self.client_type = "openai"  # Default
        
        if "gpt" in self.model_name.lower():
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.client_type = "openai"
        
        elif "claude" in self.model_name.lower():
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.client_type = "anthropic"
        
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def generate(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate text based on input prompts using API calls.
        
        Args:
            inputs: The input prompt(s).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.
        
        Returns:
            The generated text.
        """
        # Handle single input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        responses = []
        
        for input_text in inputs:
            try:
                if self.client_type == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": input_text}],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                    responses.append(response.choices[0].message.content)
                
                elif self.client_type == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": input_text}
                        ],
                        **kwargs
                    )
                    responses.append(response.content[0].text)
            
            except Exception as e:
                logger.error(f"Error generating with API model {self.model_name}: {e}")
                responses.append("")
        
        return responses
    
    def get_token_probabilities(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> List[Dict]:
        """
        Get token probabilities for logprobs-supporting API models.
        
        Args:
            inputs: The input prompt(s).
            **kwargs: Additional API call parameters.
        
        Returns:
            List of token probability dictionaries.
        """
        # Handle single input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        probabilities = []
        
        for input_text in inputs:
            try:
                if self.client_type == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": input_text}],
                        max_tokens=1,  # Just need one token for probability
                        logprobs=True,
                        top_logprobs=5,
                        **kwargs
                    )
                    if hasattr(response.choices[0], 'logprobs'):
                        probabilities.append(response.choices[0].logprobs.content[0].top_logprobs)
                    else:
                        logger.warning(f"Model {self.model_name} does not support logprobs")
                        probabilities.append({})
                
                else:
                    logger.warning(f"Token probabilities not supported for {self.client_type}")
                    probabilities.append({})
            
            except Exception as e:
                logger.error(f"Error getting token probabilities from API model: {e}")
                probabilities.append({})
        
        return probabilities