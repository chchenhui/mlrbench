"""
LLM-based hypothesis generation for identifying spurious correlations.
"""

import os
import json
import logging
import time
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import openai
import anthropic
from transformers import pipeline
import matplotlib.pyplot as plt

logger = logging.getLogger("LASS.llm_hypothesis")

class Hypothesis:
    """Class representing a hypothesis about a spurious correlation."""
    
    def __init__(self, description: str, source: str, 
                confidence: float = 0.0, validated: bool = False,
                affects_groups: Optional[List[int]] = None):
        """
        Initialize a hypothesis.
        
        Args:
            description: Textual description of the hypothesis.
            source: Source of the hypothesis (e.g., "llm", "human").
            confidence: Confidence score (0-1).
            validated: Whether the hypothesis has been validated by a human.
            affects_groups: List of group IDs affected by this spurious correlation.
        """
        self.description = description
        self.source = source
        self.confidence = confidence
        self.validated = validated
        self.affects_groups = affects_groups or []
        self.id = f"hyp_{int(time.time())}_{hash(description) % 10000}"
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.interventions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary."""
        return {
            'id': self.id,
            'description': self.description,
            'source': self.source,
            'confidence': self.confidence,
            'validated': self.validated,
            'affects_groups': self.affects_groups,
            'created_at': self.created_at,
            'interventions': self.interventions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """Create hypothesis from dictionary."""
        hyp = cls(
            description=data['description'],
            source=data['source'],
            confidence=data['confidence'],
            validated=data['validated'],
            affects_groups=data['affects_groups']
        )
        hyp.id = data['id']
        hyp.created_at = data['created_at']
        hyp.interventions = data['interventions']
        return hyp
    
    def validate(self, is_valid: bool = True) -> None:
        """
        Mark hypothesis as validated or invalidated.
        
        Args:
            is_valid: Whether the hypothesis is valid.
        """
        self.validated = is_valid
    
    def add_intervention(self, intervention_type: str, description: str) -> None:
        """
        Add an intervention based on this hypothesis.
        
        Args:
            intervention_type: Type of intervention (e.g., "counterfactual", "reweighting").
            description: Description of the intervention.
        """
        self.interventions.append({
            'type': intervention_type,
            'description': description,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
        })

class HypothesisManager:
    """Manager for storing and retrieving hypotheses."""
    
    def __init__(self, save_dir: str):
        """
        Initialize the hypothesis manager.
        
        Args:
            save_dir: Directory to save hypotheses.
        """
        self.save_dir = save_dir
        self.hypotheses = []
        os.makedirs(save_dir, exist_ok=True)
    
    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """
        Add a hypothesis to the manager.
        
        Args:
            hypothesis: Hypothesis to add.
        """
        self.hypotheses.append(hypothesis)
    
    def get_hypothesis(self, hyp_id: str) -> Optional[Hypothesis]:
        """
        Get a hypothesis by ID.
        
        Args:
            hyp_id: Hypothesis ID.
            
        Returns:
            hypothesis: Hypothesis object if found, None otherwise.
        """
        for hyp in self.hypotheses:
            if hyp.id == hyp_id:
                return hyp
        return None
    
    def get_validated_hypotheses(self) -> List[Hypothesis]:
        """
        Get all validated hypotheses.
        
        Returns:
            hypotheses: List of validated hypotheses.
        """
        return [hyp for hyp in self.hypotheses if hyp.validated]
    
    def save(self, filename: str = "hypotheses.json") -> None:
        """
        Save hypotheses to a file.
        
        Args:
            filename: Name of the file to save to.
        """
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump([hyp.to_dict() for hyp in self.hypotheses], f, indent=2)
        logger.info(f"Saved {len(self.hypotheses)} hypotheses to {path}")
    
    def load(self, filename: str = "hypotheses.json") -> None:
        """
        Load hypotheses from a file.
        
        Args:
            filename: Name of the file to load from.
        """
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.hypotheses = [Hypothesis.from_dict(item) for item in data]
            logger.info(f"Loaded {len(self.hypotheses)} hypotheses from {path}")
        else:
            logger.warning(f"No hypotheses file found at {path}")

class LLMHypothesisGenerator:
    """Base class for LLM-based hypothesis generators."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM hypothesis generator.
        
        Args:
            api_key: API key for the LLM service.
        """
        self.api_key = api_key
    
    def generate_hypotheses(self, error_samples: List[Dict[str, Any]], 
                          true_class: str, pred_class: str, 
                          modality: str = "image") -> List[Hypothesis]:
        """
        Generate hypotheses about spurious correlations.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            modality: Data modality ("image" or "text").
            
        Returns:
            hypotheses: List of generated hypotheses.
        """
        raise NotImplementedError

class OpenAIHypothesisGenerator(LLMHypothesisGenerator):
    """Hypothesis generator using OpenAI's GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI hypothesis generator.
        
        Args:
            api_key: OpenAI API key.
            model: GPT model to use.
        """
        super().__init__(api_key)
        self.model = model
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            logger.warning("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image as base64 string.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            encoded_image: Base64-encoded image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_image_prompt(self, error_samples: List[Dict[str, Any]], 
                           true_class: str, pred_class: str) -> List[Dict[str, Any]]:
        """
        Create prompt for image analysis.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            
        Returns:
            messages: List of message objects for the OpenAI API.
        """
        messages = [
            {"role": "system", "content": "You are an AI assistant helping to debug an image classification model that has learned spurious correlations. Your task is to analyze images where the model made confident errors and identify potential spurious features that might be misleading the model."},
            {"role": "user", "content": [
                {"type": "text", "text": f"I'm training a model to classify images into categories including '{true_class}' and '{pred_class}'. The model is making systematic errors where it classifies images that should be '{true_class}' as '{pred_class}' with high confidence. Below are some examples of these errors.\n\nPlease carefully examine these images and identify potential spurious correlations - visual patterns or features that are present in these misclassified images that might be misleading the model, but are NOT actually defining characteristics of the true class '{true_class}'.\n\nFor example, if a model misclassified images of 'dogs' as 'wolves', you might notice that all the misclassified dogs are in snowy backgrounds, suggesting the model has learned to associate 'snow' with 'wolf' rather than focusing on the animal's actual features.\n\nAfter analysis, list the top 5 potential spurious features you've identified, ranked by how likely you think they are to be causing these errors. For each one, briefly explain your reasoning."}
            ]}
        ]
        
        # Add the error samples as images
        for i, sample in enumerate(error_samples[:10]):  # Limit to 10 images
            if 'image_path' in sample:
                try:
                    base64_image = self._encode_image(sample['image_path'])
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                except Exception as e:
                    logger.error(f"Error encoding image {sample['image_path']}: {e}")
        
        return messages
    
    def _create_text_prompt(self, error_samples: List[Dict[str, Any]], 
                          true_class: str, pred_class: str) -> List[Dict[str, Any]]:
        """
        Create prompt for text analysis.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            
        Returns:
            messages: List of message objects for the OpenAI API.
        """
        # Extract text from samples
        sample_texts = []
        for i, sample in enumerate(error_samples[:15]):  # Limit to 15 texts
            if 'text' in sample:
                sample_texts.append(f"Example {i+1}: {sample['text']}")
        
        text_samples = "\n\n".join(sample_texts)
        
        messages = [
            {"role": "system", "content": "You are an AI assistant helping to debug a text classification model that has learned spurious correlations. Your task is to analyze text examples where the model made confident errors and identify potential spurious features that might be misleading the model."},
            {"role": "user", "content": f"I'm training a model to classify text into categories including '{true_class}' and '{pred_class}'. The model is making systematic errors where it classifies texts that should be '{true_class}' as '{pred_class}' with high confidence. Below are some examples of these errors.\n\n{text_samples}\n\nPlease carefully examine these texts and identify potential spurious correlations - linguistic patterns, specific words, phrases, or stylistic elements that are present in these misclassified texts that might be misleading the model, but are NOT actually defining characteristics of the true class '{true_class}'.\n\nFor example, if a model misclassified 'positive reviews' as 'negative reviews', you might notice that all the misclassified positive reviews contain the phrase 'could have been better', suggesting the model has learned to associate this phrase with negative sentiment despite the overall review being positive.\n\nAfter analysis, list the top 5 potential spurious features you've identified, ranked by how likely you think they are to be causing these errors. For each one, briefly explain your reasoning."}
        ]
        
        return messages
    
    def _parse_response(self, response: str) -> List[Hypothesis]:
        """
        Parse LLM response to extract hypotheses.
        
        Args:
            response: LLM response text.
            
        Returns:
            hypotheses: List of extracted hypotheses.
        """
        hypotheses = []
        
        # Simple parsing: look for numbered lists or bullet points
        lines = response.split('\n')
        current_hyp = None
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered hypotheses (1., 2., etc.)
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and len(line) > 3:
                # If we were building a hypothesis, add it
                if current_hyp:
                    hypotheses.append(Hypothesis(current_hyp.strip(), "llm", confidence=0.8))
                
                # Start new hypothesis
                current_hyp = line[line.find('.')+1:].strip()
            
            # Check for bullet points
            elif line.startswith(('•', '-', '*')) and len(line) > 2:
                # If we were building a hypothesis, add it
                if current_hyp:
                    hypotheses.append(Hypothesis(current_hyp.strip(), "llm", confidence=0.8))
                
                # Start new hypothesis
                current_hyp = line[1:].strip()
            
            # Continue building current hypothesis
            elif current_hyp and line:
                current_hyp += " " + line
        
        # Add the last hypothesis if any
        if current_hyp:
            hypotheses.append(Hypothesis(current_hyp.strip(), "llm", confidence=0.8))
        
        return hypotheses
    
    def generate_hypotheses(self, error_samples: List[Dict[str, Any]], 
                          true_class: str, pred_class: str, 
                          modality: str = "image") -> List[Hypothesis]:
        """
        Generate hypotheses about spurious correlations using OpenAI API.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            modality: Data modality ("image" or "text").
            
        Returns:
            hypotheses: List of generated hypotheses.
        """
        if not error_samples:
            logger.warning("No error samples provided for hypothesis generation")
            return []
        
        try:
            # Create appropriate prompt based on modality
            if modality == "image":
                messages = self._create_image_prompt(error_samples, true_class, pred_class)
            elif modality == "text":
                messages = self._create_text_prompt(error_samples, true_class, pred_class)
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract hypotheses from response
            hypotheses = self._parse_response(response.choices[0].message.content)
            
            logger.info(f"Generated {len(hypotheses)} hypotheses using OpenAI")
            for i, hyp in enumerate(hypotheses):
                logger.info(f"Hypothesis {i+1}: {hyp.description[:100]}...")
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error generating hypotheses with OpenAI: {e}")
            return []

class AnthropicHypothesisGenerator(LLMHypothesisGenerator):
    """Hypothesis generator using Anthropic's Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the Anthropic hypothesis generator.
        
        Args:
            api_key: Anthropic API key.
            model: Claude model to use.
        """
        super().__init__(api_key)
        self.model = model
        
        # Set API key
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            logger.warning("No Anthropic API key provided. Please set the ANTHROPIC_API_KEY environment variable.")
            self.client = None
    
    def _encode_image(self, image_path: str) -> bytes:
        """
        Read image file as bytes.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            image_bytes: Image file contents as bytes.
        """
        with open(image_path, "rb") as image_file:
            return image_file.read()
    
    def _create_image_prompt(self, error_samples: List[Dict[str, Any]], 
                           true_class: str, pred_class: str) -> List:
        """
        Create prompt for image analysis.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            
        Returns:
            messages: List of message objects for the Anthropic API.
        """
        system_prompt = "You are an AI assistant helping to debug an image classification model that has learned spurious correlations. Your task is to analyze images where the model made confident errors and identify potential spurious features that might be misleading the model."
        
        user_text = f"I'm training a model to classify images into categories including '{true_class}' and '{pred_class}'. The model is making systematic errors where it classifies images that should be '{true_class}' as '{pred_class}' with high confidence. I'll show you some examples of these errors.\n\nPlease carefully examine these images and identify potential spurious correlations - visual patterns or features that are present in these misclassified images that might be misleading the model, but are NOT actually defining characteristics of the true class '{true_class}'.\n\nFor example, if a model misclassified images of 'dogs' as 'wolves', you might notice that all the misclassified dogs are in snowy backgrounds, suggesting the model has learned to associate 'snow' with 'wolf' rather than focusing on the animal's actual features.\n\nAfter analysis, list the top 5 potential spurious features you've identified, ranked by how likely you think they are to be causing these errors. For each one, briefly explain your reasoning."
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": user_text}
            ]}
        ]
        
        # Add the error samples as images
        for i, sample in enumerate(error_samples[:8]):  # Limit to 8 images
            if 'image_path' in sample:
                try:
                    image_bytes = self._encode_image(sample['image_path'])
                    image_media = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode('utf-8')}}
                    messages[0]["content"].append(image_media)
                except Exception as e:
                    logger.error(f"Error encoding image {sample['image_path']}: {e}")
        
        return system_prompt, messages
    
    def _create_text_prompt(self, error_samples: List[Dict[str, Any]], 
                          true_class: str, pred_class: str) -> List:
        """
        Create prompt for text analysis.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            
        Returns:
            messages: List of message objects for the Anthropic API.
        """
        # Extract text from samples
        sample_texts = []
        for i, sample in enumerate(error_samples[:15]):  # Limit to 15 texts
            if 'text' in sample:
                sample_texts.append(f"Example {i+1}: {sample['text']}")
        
        text_samples = "\n\n".join(sample_texts)
        
        system_prompt = "You are an AI assistant helping to debug a text classification model that has learned spurious correlations. Your task is to analyze text examples where the model made confident errors and identify potential spurious features that might be misleading the model."
        
        user_prompt = f"I'm training a model to classify text into categories including '{true_class}' and '{pred_class}'. The model is making systematic errors where it classifies texts that should be '{true_class}' as '{pred_class}' with high confidence. Below are some examples of these errors.\n\n{text_samples}\n\nPlease carefully examine these texts and identify potential spurious correlations - linguistic patterns, specific words, phrases, or stylistic elements that are present in these misclassified texts that might be misleading the model, but are NOT actually defining characteristics of the true class '{true_class}'.\n\nFor example, if a model misclassified 'positive reviews' as 'negative reviews', you might notice that all the misclassified positive reviews contain the phrase 'could have been better', suggesting the model has learned to associate this phrase with negative sentiment despite the overall review being positive.\n\nAfter analysis, list the top 5 potential spurious features you've identified, ranked by how likely you think they are to be causing these errors. For each one, briefly explain your reasoning."
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        return system_prompt, messages
    
    def _parse_response(self, response: str) -> List[Hypothesis]:
        """
        Parse LLM response to extract hypotheses.
        
        Args:
            response: LLM response text.
            
        Returns:
            hypotheses: List of extracted hypotheses.
        """
        hypotheses = []
        
        # Simple parsing: look for numbered lists or bullet points
        lines = response.split('\n')
        current_hyp = None
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered hypotheses (1., 2., etc.)
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and len(line) > 3:
                # If we were building a hypothesis, add it
                if current_hyp:
                    hypotheses.append(Hypothesis(current_hyp.strip(), "llm", confidence=0.8))
                
                # Start new hypothesis
                current_hyp = line[line.find('.')+1:].strip()
            
            # Check for bullet points
            elif line.startswith(('•', '-', '*')) and len(line) > 2:
                # If we were building a hypothesis, add it
                if current_hyp:
                    hypotheses.append(Hypothesis(current_hyp.strip(), "llm", confidence=0.8))
                
                # Start new hypothesis
                current_hyp = line[1:].strip()
            
            # Continue building current hypothesis
            elif current_hyp and line:
                current_hyp += " " + line
        
        # Add the last hypothesis if any
        if current_hyp:
            hypotheses.append(Hypothesis(current_hyp.strip(), "llm", confidence=0.8))
        
        return hypotheses
    
    def generate_hypotheses(self, error_samples: List[Dict[str, Any]], 
                          true_class: str, pred_class: str, 
                          modality: str = "image") -> List[Hypothesis]:
        """
        Generate hypotheses about spurious correlations using Anthropic API.
        
        Args:
            error_samples: List of samples where the model made errors.
            true_class: True class name.
            pred_class: Predicted class name.
            modality: Data modality ("image" or "text").
            
        Returns:
            hypotheses: List of generated hypotheses.
        """
        if not error_samples:
            logger.warning("No error samples provided for hypothesis generation")
            return []
        
        if self.client is None:
            logger.error("Anthropic client not initialized. Please provide a valid API key.")
            return []
        
        try:
            # Create appropriate prompt based on modality
            if modality == "image":
                system_prompt, messages = self._create_image_prompt(error_samples, true_class, pred_class)
            elif modality == "text":
                system_prompt, messages = self._create_text_prompt(error_samples, true_class, pred_class)
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            
            # Extract hypotheses from response
            hypotheses = self._parse_response(response.content[0].text)
            
            logger.info(f"Generated {len(hypotheses)} hypotheses using Anthropic")
            for i, hyp in enumerate(hypotheses):
                logger.info(f"Hypothesis {i+1}: {hyp.description[:100]}...")
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error generating hypotheses with Anthropic: {e}")
            return []

def get_hypothesis_generator(provider: str = "openai", **kwargs) -> LLMHypothesisGenerator:
    """
    Factory function to get the appropriate hypothesis generator.
    
    Args:
        provider: LLM provider ("openai" or "anthropic").
        **kwargs: Additional arguments for the generator.
        
    Returns:
        generator: Hypothesis generator object.
    """
    if provider.lower() == "openai":
        return OpenAIHypothesisGenerator(**kwargs)
    elif provider.lower() == "anthropic":
        return AnthropicHypothesisGenerator(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def extract_error_clusters(embeddings: torch.Tensor, labels: torch.Tensor, 
                        predictions: torch.Tensor, confidences: torch.Tensor,
                        confidence_threshold: float = 0.7, 
                        cluster_method: str = 'kmeans',
                        n_clusters: int = 5, **kwargs) -> List[Dict[str, Any]]:
    """
    Extract clusters of confident model errors.
    
    Args:
        embeddings: Feature embeddings.
        labels: True labels.
        predictions: Model predictions.
        confidences: Prediction confidences.
        confidence_threshold: Minimum confidence threshold for "confident" errors.
        cluster_method: Clustering method ('kmeans' or 'dbscan').
        n_clusters: Number of clusters (for K-means).
        **kwargs: Additional arguments for clustering algorithms.
        
    Returns:
        clusters: List of error clusters.
    """
    # Find confident errors
    errors = (predictions != labels) & (confidences > confidence_threshold)
    error_indices = torch.where(errors)[0].cpu().numpy()
    
    if len(error_indices) == 0:
        logger.warning("No confident errors found")
        return []
    
    # Extract embeddings and labels for errors
    error_embeddings = embeddings[error_indices].cpu().numpy()
    error_labels = labels[error_indices].cpu().numpy()
    error_preds = predictions[error_indices].cpu().numpy()
    
    # Cluster error embeddings
    from sklearn.cluster import KMeans, DBSCAN
    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=min(n_clusters, len(error_indices)), random_state=42, **kwargs)
    elif cluster_method == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {cluster_method}")
    
    cluster_labels = clusterer.fit_predict(error_embeddings)
    
    # Extract clusters
    clusters = []
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove outliers (-1) from DBSCAN
    
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        # Only consider clusters with at least 5 samples
        if len(cluster_indices) < 5:
            continue
        
        # Get most common true class and predicted class in this cluster
        cluster_true_labels = error_labels[cluster_indices]
        cluster_pred_labels = error_preds[cluster_indices]
        
        from collections import Counter
        true_counter = Counter(cluster_true_labels)
        pred_counter = Counter(cluster_pred_labels)
        
        most_common_true = true_counter.most_common(1)[0][0]
        most_common_pred = pred_counter.most_common(1)[0][0]
        
        # Only consider "clean" clusters where at least 70% of samples have the same true and pred labels
        true_ratio = true_counter[most_common_true] / len(cluster_indices)
        pred_ratio = pred_counter[most_common_pred] / len(cluster_indices)
        
        if true_ratio < 0.7 or pred_ratio < 0.7:
            continue
        
        # Original indices of samples in this cluster
        original_indices = error_indices[cluster_indices]
        
        clusters.append({
            'cluster_id': int(cluster_id),
            'size': len(cluster_indices),
            'true_class': int(most_common_true),
            'pred_class': int(most_common_pred),
            'true_ratio': float(true_ratio),
            'pred_ratio': float(pred_ratio),
            'sample_indices': original_indices.tolist()
        })
    
    # Sort clusters by size
    clusters = sorted(clusters, key=lambda c: c['size'], reverse=True)
    
    return clusters

def visualize_error_cluster(cluster: Dict[str, Any], dataset, class_names: List[str],
                          save_dir: str, num_samples: int = 9) -> str:
    """
    Visualize samples from an error cluster.
    
    Args:
        cluster: Error cluster information.
        dataset: Dataset containing the samples.
        class_names: List of class names.
        save_dir: Directory to save visualization.
        num_samples: Number of samples to visualize.
        
    Returns:
        save_path: Path to the saved visualization.
    """
    # Get random samples from the cluster
    sample_indices = cluster['sample_indices']
    if len(sample_indices) > num_samples:
        import random
        sample_indices = random.sample(sample_indices, num_samples)
    
    # Create grid of images
    rows = int(np.ceil(np.sqrt(len(sample_indices))))
    cols = int(np.ceil(len(sample_indices) / rows))
    
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows * cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    true_class_name = class_names[cluster['true_class']]
    pred_class_name = class_names[cluster['pred_class']]
    
    for i, idx in enumerate(sample_indices):
        if i >= len(axs):
            break
            
        # Get sample
        img, label, group = dataset[idx]
        
        # Convert tensor to image
        if isinstance(img, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
        
        # Display image
        axs[i].imshow(img)
        axs[i].set_title(f"True: {true_class_name}\nPred: {pred_class_name}")
        axs[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(sample_indices), len(axs)):
        axs[i].axis('off')
    
    plt.suptitle(f"Error Cluster: True={true_class_name}, Predicted={pred_class_name}\nSize: {cluster['size']} samples")
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cluster_{cluster['cluster_id']}.png")
    plt.savefig(save_path)
    plt.close()
    
    return save_path