"""
Implementation of the Adaptive Uncertainty-Gated RAG (AUG-RAG) model.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
from .base_model import BaseModel, APIBasedModel
from .rag_model import RetrieverModule, StandardRAGModel
from .uncertainty import UncertaintyEstimationModule, UncertaintyFactory

logger = logging.getLogger(__name__)

class AdaptiveRetrievalTrigger:
    """
    Adaptive Retrieval Trigger module that decides when to trigger retrieval
    based on uncertainty estimation.
    """
    
    def __init__(
        self,
        threshold_type: str = "fixed",
        fixed_threshold: float = 0.5,
        window_size: int = 5,
        **kwargs
    ):
        """
        Initialize the adaptive retrieval trigger.
        
        Args:
            threshold_type: Type of threshold ("fixed", "dynamic", "learned").
            fixed_threshold: Fixed threshold value (used if threshold_type is "fixed").
            window_size: Size of the rolling window for average uncertainty.
            **kwargs: Additional parameters for specific threshold types.
        """
        self.threshold_type = threshold_type
        self.fixed_threshold = fixed_threshold
        self.window_size = window_size
        self.kwargs = kwargs
        
        # For rolling window average
        self.uncertainty_history = []
        
        # For dynamic global threshold
        self.global_uncertainties = []
        self.global_threshold = fixed_threshold
        
        # For learned threshold (placeholder)
        self.learned_model = None
        
        logger.info(f"Initialized AdaptiveRetrievalTrigger with {threshold_type} threshold")
    
    def should_retrieve(self, uncertainty: float, context: Dict = None) -> bool:
        """
        Decide whether to trigger retrieval based on uncertainty and context.
        
        Args:
            uncertainty: Current uncertainty estimate.
            context: Additional context information (e.g., topic, query type).
        
        Returns:
            Boolean indicating whether to trigger retrieval.
        """
        # Update history
        self.uncertainty_history.append(uncertainty)
        if len(self.uncertainty_history) > self.window_size:
            self.uncertainty_history.pop(0)
        
        # Update global statistics
        self.global_uncertainties.append(uncertainty)
        if len(self.global_uncertainties) > 100:  # Limit history size
            self.global_uncertainties.pop(0)
        
        # Get threshold based on the specified type
        threshold = self._get_threshold(context)
        
        # Decision based on threshold
        should_retrieve = uncertainty > threshold
        
        logger.debug(f"Uncertainty: {uncertainty:.4f}, Threshold: {threshold:.4f}, Retrieve: {should_retrieve}")
        return should_retrieve
    
    def _get_threshold(self, context: Dict = None) -> float:
        """
        Get the current threshold based on the threshold type.
        
        Args:
            context: Additional context information.
        
        Returns:
            The current threshold value.
        """
        if self.threshold_type == "fixed":
            return self.fixed_threshold
        
        elif self.threshold_type == "rolling_window":
            if len(self.uncertainty_history) > 0:
                avg_uncertainty = sum(self.uncertainty_history) / len(self.uncertainty_history)
                # Adjust threshold based on average uncertainty
                return min(0.8, max(0.2, avg_uncertainty * 1.1))  # 10% higher than average
            return self.fixed_threshold
        
        elif self.threshold_type == "dynamic_global":
            if len(self.global_uncertainties) > 10:
                # Update global threshold based on distribution of uncertainties
                # Use a percentile-based approach
                sorted_uncertainties = sorted(self.global_uncertainties)
                percentile_idx = int(len(sorted_uncertainties) * 0.7)  # 70th percentile
                self.global_threshold = sorted_uncertainties[percentile_idx]
            return self.global_threshold
        
        elif self.threshold_type == "context_specific":
            # Context-specific threshold (simplistic implementation)
            if context and "topic" in context:
                # Adjust threshold based on topic
                topic_adjustments = {
                    "science": 0.4,  # Lower threshold for science (factual)
                    "politics": 0.6,  # Higher threshold for politics (opinions)
                    "fiction": 0.7   # Higher threshold for fiction (creative)
                }
                adjustment = topic_adjustments.get(context["topic"], 0)
                return self.fixed_threshold + adjustment
            return self.fixed_threshold
        
        elif self.threshold_type == "learned":
            # Placeholder for learned threshold
            # In a real implementation, this would use a trained model
            # to predict the optimal threshold based on context features
            if self.learned_model is not None:
                return self.learned_model.predict(context)
            return self.fixed_threshold
        
        else:
            logger.warning(f"Unknown threshold type: {self.threshold_type}. Using fixed threshold.")
            return self.fixed_threshold


class AdaptiveRAGModel:
    """
    Adaptive Uncertainty-Gated RAG model that selectively triggers retrieval
    based on model uncertainty.
    """
    
    def __init__(
        self,
        base_model: Union[BaseModel, APIBasedModel],
        retriever: RetrieverModule,
        uncertainty_estimator: UncertaintyEstimationModule,
        retrieval_trigger: AdaptiveRetrievalTrigger,
        num_documents: int = 3,
        prompt_template: str = None,
        generation_chunk_size: int = 10,
        segment_mode: bool = False
    ):
        """
        Initialize the Adaptive RAG model.
        
        Args:
            base_model: The base LLM model.
            retriever: The retriever module.
            uncertainty_estimator: The uncertainty estimation module.
            retrieval_trigger: The adaptive retrieval trigger.
            num_documents: Number of documents to retrieve when triggered.
            prompt_template: Template for constructing the prompt with retrieved docs.
            generation_chunk_size: Number of tokens to generate in each iteration.
            segment_mode: Whether to estimate uncertainty at segment level
                         (multiple tokens) instead of token level.
        """
        self.base_model = base_model
        self.retriever = retriever
        self.uncertainty_estimator = uncertainty_estimator
        self.retrieval_trigger = retrieval_trigger
        self.num_documents = num_documents
        self.generation_chunk_size = generation_chunk_size
        self.segment_mode = segment_mode
        
        # Default prompt template if none provided
        self.prompt_template = prompt_template or (
            "Answer the following question based on the information provided.\n\n"
            "Context information:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        # For tracking retrieval stats
        self.retrieval_count = 0
        self.total_steps = 0
        
        logger.info(f"Initialized AdaptiveRAGModel with {num_documents} documents per retrieval")
    
    def format_prompt_with_context(self, question: str, context_docs: List[str]) -> str:
        """
        Format the prompt with the question and retrieved context.
        
        Args:
            question: The input question.
            context_docs: List of retrieved context documents.
        
        Returns:
            The formatted prompt.
        """
        context_str = "\n".join([f"- {doc}" for doc in context_docs])
        return self.prompt_template.format(question=question, context=context_str)
    
    def generate_with_adaptive_retrieval(
        self,
        input_text: str,
        max_new_tokens: int = 128,
        **kwargs
    ) -> str:
        """
        Generate text with adaptive retrieval based on uncertainty.
        
        Args:
            input_text: The input question.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional generation parameters.
        
        Returns:
            The generated answer.
        """
        # Reset counters
        self.retrieval_count = 0
        self.total_steps = 0
        
        # Initialize context and generated text
        current_context = input_text
        generated_text = ""
        retrieved_docs = []
        
        # Track whether we've already done retrieval for the current segment
        retrieval_done_for_segment = False
        
        # Generate tokens iteratively
        while len(generated_text.split()) < max_new_tokens:
            self.total_steps += 1
            
            # Estimate uncertainty for the current context
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(current_context, **kwargs)
            
            # Decide whether to trigger retrieval
            if not retrieval_done_for_segment and self.retrieval_trigger.should_retrieve(uncertainty):
                # Retrieve documents
                query = input_text + " " + generated_text
                new_docs = self.retriever.retrieve(query, self.num_documents)
                
                # Only add non-duplicate documents
                for doc in new_docs:
                    if doc not in retrieved_docs:
                        retrieved_docs.append(doc)
                
                # Update current context with retrieved documents
                if retrieved_docs:
                    current_context = self.format_prompt_with_context(
                        input_text + " " + generated_text,
                        retrieved_docs
                    )
                
                self.retrieval_count += 1
                retrieval_done_for_segment = True
                logger.debug(f"Retrieval triggered at step {self.total_steps} with uncertainty {uncertainty:.4f}")
            
            # Generate next chunk
            chunk_size = min(self.generation_chunk_size, max_new_tokens - len(generated_text.split()))
            generated_chunk = self.base_model.generate(
                current_context,
                max_new_tokens=chunk_size,
                **kwargs
            )[0]
            
            # Update generated text and current context
            if not generated_chunk:
                # Break if no more tokens generated
                break
            
            generated_text += " " + generated_chunk if generated_text else generated_chunk
            current_context = input_text + " " + generated_text
            
            # Reset segment flag if in segment mode
            if self.segment_mode:
                retrieval_done_for_segment = False
        
        # Log statistics
        logger.info(f"Generated {len(generated_text.split())} tokens with {self.retrieval_count} retrievals in {self.total_steps} steps")
        
        return generated_text.strip()
    
    def generate(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """
        Generate answers with adaptive retrieval augmentation.
        
        Args:
            inputs: The input question(s).
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional generation parameters.
        
        Returns:
            The generated answers.
        """
        # Handle single input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        outputs = []
        
        # Generate for each input
        for input_text in inputs:
            output = self.generate_with_adaptive_retrieval(
                input_text,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            outputs.append(output)
        
        return outputs
    
    def get_retrieval_stats(self) -> Dict[str, float]:
        """
        Get statistics about retrieval frequency.
        
        Returns:
            Dictionary with retrieval statistics.
        """
        retrieval_frequency = self.retrieval_count / max(1, self.total_steps)
        return {
            "retrieval_count": self.retrieval_count,
            "total_steps": self.total_steps,
            "retrieval_frequency": retrieval_frequency
        }


class AUGRAGFactory:
    """Factory class for creating AUG-RAG models with different configurations."""
    
    @staticmethod
    def create(
        config: Dict[str, Any]
    ) -> AdaptiveRAGModel:
        """
        Create an AUG-RAG model based on the provided configuration.
        
        Args:
            config: Configuration dictionary with model settings.
        
        Returns:
            An initialized AUG-RAG model.
        """
        # Create base model
        model_type = config.get("model_type", "local")
        model_name = config.get("model_name", "gpt2")
        
        if model_type == "api":
            base_model = APIBasedModel(
                model_name=model_name,
                api_key=config.get("api_key", None)
            )
        else:
            base_model = BaseModel(
                model_name=model_name,
                device=config.get("device", None)
            )
        
        # Create retriever
        retriever = RetrieverModule(
            knowledge_base_path=config.get("knowledge_base_path", "data/dummy_kb.json"),
            embedding_model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            use_sparse=config.get("use_sparse_retriever", False)
        )
        
        # Create uncertainty estimator
        uncertainty_method = config.get("uncertainty_method", "entropy")
        uncertainty_estimator = UncertaintyFactory.create(
            method=uncertainty_method,
            model=base_model,
            **config.get("uncertainty_params", {})
        )
        
        # Create adaptive retrieval trigger
        retrieval_trigger = AdaptiveRetrievalTrigger(
            threshold_type=config.get("threshold_type", "fixed"),
            fixed_threshold=config.get("fixed_threshold", 0.5),
            window_size=config.get("window_size", 5),
            **config.get("trigger_params", {})
        )
        
        # Create the AUG-RAG model
        aug_rag_model = AdaptiveRAGModel(
            base_model=base_model,
            retriever=retriever,
            uncertainty_estimator=uncertainty_estimator,
            retrieval_trigger=retrieval_trigger,
            num_documents=config.get("num_documents", 3),
            prompt_template=config.get("prompt_template", None),
            generation_chunk_size=config.get("generation_chunk_size", 10),
            segment_mode=config.get("segment_mode", False)
        )
        
        return aug_rag_model