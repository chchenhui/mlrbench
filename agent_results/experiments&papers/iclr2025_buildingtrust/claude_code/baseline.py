"""
Baseline models for the Self-Correcting Language Model experiment.
"""
import time
from typing import Dict, List, Any, Optional, Union
import re

from config import logger
from models import BaseModel, APIModel, HuggingFaceModel
from utils import time_function


class ZeroShotBaselineModel:
    """
    Zero-shot baseline model without self-correction.
    
    This model simply uses the base model to generate responses without any
    correction mechanism.
    """
    
    def __init__(self, model_name: str, use_api: bool = True):
        """
        Initialize zero-shot baseline model.
        
        Args:
            model_name: Name of the base model to use
            use_api: Whether to use API models (True) or local models (False)
        """
        self.model_name = model_name
        
        # Initialize base model
        if use_api:
            self.base_model = APIModel(model_name)
        else:
            self.base_model = HuggingFaceModel(model_name)
        
        logger.info(f"Initialized zero-shot baseline with model {model_name}")
    
    @time_function
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text without self-correction.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the text and metadata
        """
        # Generate response
        response = self.base_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Return result
        result = {
            "original_text": response,
            "final_text": response,
            "corrections": [],
            "metrics": {
                "num_iterations": 0,
                "num_spans_corrected": 0,
                "confidence_improvement": 0.0,
                "latency": 0.0  # Will be filled by time_function decorator
            }
        }
        
        return result
    
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
        
        # Generate response
        result = self.generate(
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


class RetrievalBaselineModel:
    """
    Retrieval-augmented baseline model.
    
    This model uses retrieval to enhance the base model's responses, but without
    the self-correction loop.
    """
    
    def __init__(self, model_name: str, use_api: bool = True, retrieval_k: int = 5):
        """
        Initialize retrieval-augmented baseline model.
        
        Args:
            model_name: Name of the base model to use
            use_api: Whether to use API models (True) or local models (False)
            retrieval_k: Number of documents to retrieve
        """
        self.model_name = model_name
        self.retrieval_k = retrieval_k
        
        # Initialize base model
        if use_api:
            self.base_model = APIModel(model_name)
        else:
            self.base_model = HuggingFaceModel(model_name)
        
        logger.info(f"Initialized retrieval baseline with model {model_name}")
        logger.info(f"  Retrieval k: {self.retrieval_k}")
    
    def _simulate_retrieval(self, query: str) -> List[str]:
        """
        Simulate document retrieval.
        
        In a real implementation, this would query external knowledge bases.
        For this experiment, we'll simulate retrieval by asking the model to generate
        factual information about the query.
        
        Args:
            query: Query to retrieve information for
            
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
    
    @time_function
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text with retrieval augmentation.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the text and metadata
        """
        # Extract query from prompt
        query = prompt
        
        # Retrieve documents
        retrieved_docs = self._simulate_retrieval(query)
        
        # Augment prompt with retrieved documents
        augmented_prompt = prompt + "\n\nRelevant information:\n"
        for i, doc in enumerate(retrieved_docs):
            augmented_prompt += f"{i+1}. {doc}\n"
        
        # Generate response
        response = self.base_model.generate(
            augmented_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Return result
        result = {
            "original_text": response,
            "final_text": response,
            "corrections": [],
            "retrieved_docs": retrieved_docs,
            "metrics": {
                "num_iterations": 0,
                "num_spans_corrected": 0,
                "confidence_improvement": 0.0,
                "latency": 0.0  # Will be filled by time_function decorator
            }
        }
        
        return result
    
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
        
        # Generate response
        result = self.generate(
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


class RuleCorrectionModel:
    """
    Simple rule-based correction model.
    
    This model uses a set of rules to detect and correct common errors without
    retrieval or self-attention analysis.
    """
    
    def __init__(self, model_name: str, use_api: bool = True):
        """
        Initialize rule-based correction model.
        
        Args:
            model_name: Name of the base model to use
            use_api: Whether to use API models (True) or local models (False)
        """
        self.model_name = model_name
        
        # Initialize base model
        if use_api:
            self.base_model = APIModel(model_name)
        else:
            self.base_model = HuggingFaceModel(model_name)
        
        logger.info(f"Initialized rule-based correction model with base model {model_name}")
    
    def _apply_correction_rules(self, text: str) -> Dict[str, Any]:
        """
        Apply correction rules to text.
        
        Args:
            text: Text to correct
            
        Returns:
            Dictionary containing the corrected text and correction metadata
        """
        original_text = text
        corrections = []
        
        # Rule 1: Hedging and uncertain language
        hedging_patterns = [
            (r'\bI think\b', ''),
            (r'\bprobably\b', ''),
            (r'\bmight\b', ''),
            (r'\bcould be\b', 'is'),
            (r'\bpossibly\b', ''),
            (r'\bperhaps\b', '')
        ]
        
        for pattern, replacement in hedging_patterns:
            # Find all matches
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                span = match.group(0)
                start, end = match.span()
                context_before = text[max(0, start-20):start]
                context_after = text[end:min(end+20, len(text))]
                
                # Replace the span
                if replacement:
                    corrected_span = re.sub(pattern, replacement, span, flags=re.IGNORECASE)
                else:
                    # If replacement is empty, just remove the matched pattern and surrounding space
                    corrected_span = ''
                
                text = text[:start] + corrected_span + text[end:]
                
                # Adjust indices for subsequent replacements
                adjustment = len(corrected_span) - len(span)
                end += adjustment
                
                # Record correction
                corrections.append({
                    "original_span": span,
                    "corrected_span": corrected_span,
                    "context_before": context_before,
                    "context_after": context_after,
                    "rule": "hedging_removal"
                })
        
        # Rule 2: Convert questions to statements
        question_pattern = r'((?:What|Who|When|Where|Why|How).*?\?)'
        
        matches = list(re.finditer(question_pattern, text))
        for match in matches:
            question = match.group(0)
            start, end = match.span()
            
            # Ask the model to convert the question to a statement
            statement_prompt = f"Convert this question to a direct statement: '{question}'"
            statement = self.base_model.generate(statement_prompt, max_tokens=50, temperature=0.3)
            
            # Replace the question with the statement
            text = text[:start] + statement + text[end:]
            
            # Record correction
            corrections.append({
                "original_span": question,
                "corrected_span": statement,
                "rule": "question_to_statement"
            })
        
        # Result
        result = {
            "original_text": original_text,
            "final_text": text,
            "corrections": corrections,
            "metrics": {
                "num_iterations": 1,
                "num_spans_corrected": len(corrections),
            }
        }
        
        return result
    
    @time_function
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text with rule-based correction.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing the text and metadata
        """
        # Generate initial response
        start_time = time.time()
        
        initial_response = self.base_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        generation_time = time.time() - start_time
        
        # Apply correction rules
        correction_start_time = time.time()
        
        result = self._apply_correction_rules(initial_response)
        
        correction_time = time.time() - correction_start_time
        
        # Update metrics
        result["metrics"]["latency"] = generation_time + correction_time
        result["metrics"]["generation_time"] = generation_time
        result["metrics"]["correction_time"] = correction_time
        
        return result
    
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
        
        # Generate response
        result = self.generate(
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


def get_baseline_model(baseline_type: str, model_name: str, use_api: bool = True, **kwargs) -> Any:
    """
    Factory function to get the appropriate baseline model.
    
    Args:
        baseline_type: Type of baseline model
        model_name: Name of the base model to use
        use_api: Whether to use API models (True) or local models (False)
        **kwargs: Additional model configuration
    
    Returns:
        Baseline model instance
    """
    if baseline_type.lower() == "zero_shot":
        return ZeroShotBaselineModel(model_name, use_api)
    elif baseline_type.lower() == "retrieval":
        return RetrievalBaselineModel(model_name, use_api, **kwargs)
    elif baseline_type.lower() == "rule_based":
        return RuleCorrectionModel(model_name, use_api)
    else:
        raise ValueError(f"Unsupported baseline type: {baseline_type}")