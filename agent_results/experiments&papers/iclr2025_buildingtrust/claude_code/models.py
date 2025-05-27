"""
Model implementations for the Self-Correcting Language Model experiment.
"""
import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import anthropic
import re

from config import (
    MODEL_CONFIGS,
    API_MODELS,
    SCLM_CONFIG,
    logger
)
from utils import time_function


class BaseModel:
    """Base class for language models."""
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model to load
        """
        self.model_name = model_name
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional keyword arguments
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results


class HuggingFaceModel(BaseModel):
    """Wrapper for Hugging Face models."""
    
    def __init__(self, model_name: str):
        """
        Initialize Hugging Face model.
        
        Args:
            model_name: Name of the model to load
        """
        super().__init__(model_name)
        self.config = MODEL_CONFIGS.get(model_name)
        if not self.config:
            raise ValueError(f"Model {model_name} not found in MODEL_CONFIGS")
        
        logger.info(f"Loading HuggingFace model: {self.config['huggingface_id']}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['huggingface_id'],
            revision=self.config.get('revision', 'main'),
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['huggingface_id'],
            revision=self.config.get('revision', 'main'),
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            low_cpu_mem_usage=True,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        # Move model to device if not using device_map='auto'
        if self.device == 'cpu' or not hasattr(self.model, 'hf_device_map'):
            self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
    
    @time_function
    def generate(self, prompt: str, max_length: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return generated text
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):]
        
        return output_text.strip()
    
    def get_attention_values(self, prompt: str) -> Dict[str, List[float]]:
        """
        Get attention values for each token in the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dictionary mapping tokens to their attention values across layers
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Extract attention values
        attentions = outputs.attentions  # Tuple of tensors of shape (batch_size, num_heads, seq_len, seq_len)
        
        # Calculate attention entropy
        token_attentions = {}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        for i, token in enumerate(tokens):
            token_attentions[token] = []
            
            for layer_idx, layer_attention in enumerate(attentions):
                # Average over attention heads
                avg_attention = layer_attention[0, :, i, :].mean(dim=0)
                
                # Calculate entropy
                # Clamp to avoid log(0)
                avg_attention = torch.clamp(avg_attention, min=1e-10)
                entropy = -torch.sum(avg_attention * torch.log(avg_attention))
                
                token_attentions[token].append(float(entropy))
        
        return token_attentions
    
    def get_confidence_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate confidence scores for segments of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping text segments to confidence scores
        """
        # Tokenize text into segments (for simplicity, split by sentence)
        segments = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        segment_scores = {}
        for segment in segments:
            attention_values = self.get_attention_values(segment)
            
            # Calculate confidence score from attention entropy
            # Lower entropy means higher confidence
            avg_entropy = np.mean([
                np.mean(values) for values in attention_values.values()
            ])
            
            # Convert to confidence score (0-1)
            confidence = 1.0 - (avg_entropy / 10.0)  # Normalize
            confidence = max(0.0, min(1.0, confidence))  # Clip to [0, 1]
            
            segment_scores[segment] = confidence
        
        return segment_scores


class APIModel(BaseModel):
    """Wrapper for API-based models (OpenAI, Anthropic)."""
    
    def __init__(self, model_name: str):
        """
        Initialize API model.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__(model_name)
        self.config = API_MODELS.get(model_name)
        if not self.config:
            raise ValueError(f"Model {model_name} not found in API_MODELS")
        
        # Initialize the appropriate client
        self.provider = self.config["provider"]
        
        if self.provider == "openai":
            # Check for API key
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info(f"Initialized OpenAI client for model {model_name}")
            
        elif self.provider == "anthropic":
            # Check for API key
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            logger.info(f"Initialized Anthropic client for model {model_name}")
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    @time_function
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate text from prompt using API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config["name"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config["name"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"Error: {str(e)}"
        
    def estimate_confidence(self, text: str, question: str) -> Dict[str, float]:
        """
        Estimate confidence for segments of text by asking the model.
        
        Args:
            text: Input text to evaluate confidence for
            question: Original question that prompted the response
            
        Returns:
            Dictionary mapping text segments to confidence scores
        """
        segments = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        confidence_prompt = f"""
        Given this question: "{question}"
        
        And this answer: "{text}"
        
        For each statement below, estimate your confidence in its factual accuracy on a scale from 0.0 to 1.0, where:
        - 0.0 means completely uncertain or likely incorrect
        - 1.0 means completely certain and factually correct
        
        For each statement, just respond with a number between 0.0 and 1.0, nothing else.
        """
        
        segment_scores = {}
        for segment in segments:
            segment_prompt = confidence_prompt + f"\n\nStatement: \"{segment}\"\nConfidence (0.0-1.0):"
            
            try:
                confidence_text = self.generate(segment_prompt, max_tokens=10, temperature=0.0)
                # Extract the confidence score
                match = re.search(r'([01]?\.\d+)', confidence_text)
                if match:
                    confidence = float(match.group(1))
                    # Ensure it's in [0, 1]
                    confidence = max(0.0, min(1.0, confidence))
                else:
                    confidence = 0.5  # Default if parsing fails
            except Exception as e:
                logger.warning(f"Failed to estimate confidence: {e}")
                confidence = 0.5  # Default if API call fails
            
            segment_scores[segment] = confidence
        
        return segment_scores


class SelfCorrectingModel:
    """
    Self-Correcting Language Model implementation.
    
    This model detects low-confidence spans in generated text and corrects them
    using a retrieval-augmented approach.
    """
    
    def __init__(
        self,
        base_model_name: str,
        use_api: bool = True,
        confidence_threshold: float = None,
        max_iterations: int = None,
        retrieval_k: int = None
    ):
        """
        Initialize Self-Correcting Language Model.
        
        Args:
            base_model_name: Name of the base model to use
            use_api: Whether to use API models (True) or local models (False)
            confidence_threshold: Threshold for confidence scoring
            max_iterations: Maximum number of correction iterations
            retrieval_k: Number of documents to retrieve
        """
        self.use_api = use_api
        
        # Initialize base model
        if use_api:
            self.base_model = APIModel(base_model_name)
        else:
            self.base_model = HuggingFaceModel(base_model_name)
        
        # Get configuration
        self.confidence_threshold = confidence_threshold or SCLM_CONFIG["confidence_threshold"]
        self.max_iterations = max_iterations or SCLM_CONFIG["max_iterations"]
        self.retrieval_k = retrieval_k or SCLM_CONFIG["retrieval_k"]
        
        logger.info(f"Initialized SCLM with base model {base_model_name}")
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  Max iterations: {self.max_iterations}")
        logger.info(f"  Retrieval k: {self.retrieval_k}")
    
    def _simulate_retrieval(self, span: str, query: str) -> List[str]:
        """
        Simulate document retrieval for correction.
        
        In a real implementation, this would query external knowledge bases.
        For this experiment, we'll simulate retrieval by asking the model to generate
        factual information about the topic.
        
        Args:
            span: Text span to retrieve information for
            query: Query derived from the span
            
        Returns:
            List of retrieved document snippets
        """
        retrieval_prompt = f"""
        I need factual, accurate information about the following topic:
        
        {query}
        
        Please provide {self.retrieval_k} concise, factually accurate statements about this topic.
        Focus only on well-established facts, not opinions or speculations.
        Provide citations for your information when possible.
        """
        
        # Generate factual information
        retrieval_result = self.base_model.generate(retrieval_prompt, temperature=0.2)
        
        # Split into separate "documents"
        documents = [s.strip() for s in retrieval_result.split('\n') if s.strip()]
        
        # If we didn't get enough documents, just duplicate what we have
        while len(documents) < self.retrieval_k:
            documents.append(documents[0] if documents else "No information found.")
        
        return documents[:self.retrieval_k]  # Limit to retrieval_k documents
    
    def _generate_correction_query(self, span: str) -> str:
        """
        Generate a query for retrieving information to correct a span.
        
        Args:
            span: Text span to correct
            
        Returns:
            Query for retrieval
        """
        # Extract key entities and claims from the span
        query_prompt = f"""
        Extract the main factual claims or key entities from this text. 
        Focus on extracting the core claim that needs to be verified:

        Text: "{span}"
        
        Factual claim or question:
        """
        
        query = self.base_model.generate(query_prompt, max_tokens=100, temperature=0.3)
        
        # Clean up the query
        query = query.strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        
        return query
    
    def _correct_span(self, span: str, retrieved_docs: List[str]) -> str:
        """
        Correct a span using retrieved documents.
        
        Args:
            span: Text span to correct
            retrieved_docs: List of retrieved document snippets
            
        Returns:
            Corrected span
        """
        correction_prompt = f"""
        I need to rewrite the following statement to ensure it is factually accurate.
        
        Original statement: "{span}"
        
        Here are some relevant facts:
        {' '.join([f"- {doc}" for doc in retrieved_docs])}
        
        Please rewrite the original statement to be factually accurate, based on the information provided.
        If the original statement was already accurate, you can keep it as is.
        Just provide the rewritten statement, nothing else.
        """
        
        corrected_span = self.base_model.generate(correction_prompt, max_tokens=200, temperature=0.3)
        
        # Clean up the corrected span
        corrected_span = corrected_span.strip()
        if corrected_span.startswith('"') and corrected_span.endswith('"'):
            corrected_span = corrected_span[1:-1]
        
        return corrected_span
    
    def _detect_low_confidence_spans(self, text: str, question: str = "") -> Dict[str, float]:
        """
        Detect spans with low confidence scores.
        
        Args:
            text: Text to analyze
            question: Original question that prompted the response
            
        Returns:
            Dictionary mapping low-confidence spans to their confidence scores
        """
        # Get confidence scores for all segments
        if self.use_api:
            confidence_scores = self.base_model.estimate_confidence(text, question)
        else:
            confidence_scores = self.base_model.get_confidence_scores(text)
        
        # Filter spans below threshold
        low_confidence_spans = {
            span: score for span, score in confidence_scores.items()
            if score < self.confidence_threshold
        }
        
        return low_confidence_spans
    
    @time_function
    def generate_with_correction(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text with self-correction.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the final text and metadata
        """
        # Initial generation
        logger.info(f"Generating initial response for prompt: {prompt[:50]}...")
        initial_response = self.base_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Prepare result dictionary
        result = {
            "original_text": initial_response,
            "final_text": initial_response,
            "corrections": [],
            "metrics": {
                "num_iterations": 0,
                "num_spans_corrected": 0,
                "confidence_improvement": 0.0,
            }
        }
        
        # Correction loop
        current_text = initial_response
        iteration = 0
        total_spans_corrected = 0
        
        initial_confidence = self._calculate_avg_confidence(current_text, prompt)
        
        while iteration < self.max_iterations:
            # Detect low-confidence spans
            low_confidence_spans = self._detect_low_confidence_spans(current_text, prompt)
            
            # If no low-confidence spans, we're done
            if not low_confidence_spans:
                logger.info(f"No low-confidence spans detected at iteration {iteration}. Stopping.")
                break
            
            logger.info(f"Iteration {iteration}: Found {len(low_confidence_spans)} low-confidence spans")
            
            # Process each low-confidence span
            corrections_made = False
            for span, confidence in low_confidence_spans.items():
                # Generate query for retrieval
                query = self._generate_correction_query(span)
                
                # Retrieve documents
                retrieved_docs = self._simulate_retrieval(span, query)
                
                # Correct span
                corrected_span = self._correct_span(span, retrieved_docs)
                
                # Only replace if the correction is different
                if corrected_span != span:
                    logger.info(f"Corrected: '{span}' -> '{corrected_span}'")
                    
                    # Replace in the current text
                    current_text = current_text.replace(span, corrected_span)
                    
                    # Record correction
                    result["corrections"].append({
                        "iteration": iteration,
                        "original_span": span,
                        "corrected_span": corrected_span,
                        "confidence_before": confidence,
                        "retrieved_docs": retrieved_docs
                    })
                    
                    total_spans_corrected += 1
                    corrections_made = True
            
            # If no corrections were made, stop
            if not corrections_made:
                logger.info(f"No corrections made at iteration {iteration}. Stopping.")
                break
            
            iteration += 1
        
        # Update result
        result["final_text"] = current_text
        result["metrics"]["num_iterations"] = iteration
        result["metrics"]["num_spans_corrected"] = total_spans_corrected
        
        # Calculate final confidence
        final_confidence = self._calculate_avg_confidence(current_text, prompt)
        result["metrics"]["confidence_improvement"] = final_confidence - initial_confidence
        
        logger.info(f"Self-correction complete. Made {total_spans_corrected} corrections in {iteration} iterations.")
        logger.info(f"Confidence improvement: {result['metrics']['confidence_improvement']:.4f}")
        
        return result
    
    def _calculate_avg_confidence(self, text: str, question: str = "") -> float:
        """
        Calculate average confidence score for text.
        
        Args:
            text: Text to analyze
            question: Original question that prompted the response
            
        Returns:
            Average confidence score
        """
        if self.use_api:
            confidence_scores = self.base_model.estimate_confidence(text, question)
        else:
            confidence_scores = self.base_model.get_confidence_scores(text)
        
        if not confidence_scores:
            return 0.0
        
        return sum(confidence_scores.values()) / len(confidence_scores)
    
    @time_function
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single sample.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            Evaluation results
        """
        # Construct prompt
        prompt = sample["question"]
        if sample.get("context"):
            prompt = f"Context: {sample['context']}\n\nQuestion: {prompt}"
        
        # Generate with self-correction
        result = self.generate_with_correction(
            prompt,
            max_tokens=1024,
            temperature=0.7
        )
        
        # Add sample info to result
        result["sample_id"] = sample.get("id", "")
        result["question"] = sample["question"]
        result["context"] = sample.get("context", "")
        result["correct_answers"] = sample.get("correct_answers", [])
        
        return result


def get_model(model_name: str, use_api: bool = True, **kwargs) -> Union[BaseModel, SelfCorrectingModel]:
    """
    Factory function to get the appropriate model.
    
    Args:
        model_name: Name of the model
        use_api: Whether to use API models (True) or local models (False)
        **kwargs: Additional model configuration
    
    Returns:
        Model instance
    """
    if model_name.lower() == "sclm":
        # Self-Correcting Language Model
        base_model_name = kwargs.pop("base_model", "llama-3.1-8b" if not use_api else "claude-3.7-sonnet")
        return SelfCorrectingModel(base_model_name=base_model_name, use_api=use_api, **kwargs)
    elif use_api and model_name in API_MODELS:
        # API model
        return APIModel(model_name)
    elif not use_api and model_name in MODEL_CONFIGS:
        # Local model
        return HuggingFaceModel(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name} (use_api={use_api})")