"""
Evaluation metrics for language generation models, including hallucination detection.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import evaluate
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sacrebleu
from rouge_score import rouge_scorer
import logging

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class Evaluator:
    """Evaluator for language generation models."""
    
    def __init__(self, tokenizer, device=None):
        """
        Initialize the evaluator.
        
        Args:
            tokenizer: The tokenizer for the language model.
            device: The device to run the evaluation on.
        """
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = evaluate.load('bleu')
        
    def evaluate_bleu(self, predictions, references):
        """
        Evaluate BLEU score.
        
        Args:
            predictions: The predicted texts.
            references: The reference texts.
        
        Returns:
            The BLEU score.
        """
        tokenized_predictions = [pred.strip() for pred in predictions]
        tokenized_references = [[ref.strip()] for ref in references]
        
        return self.bleu.compute(predictions=tokenized_predictions, references=tokenized_references)['bleu']
    
    def evaluate_rouge(self, predictions, references):
        """
        Evaluate ROUGE scores.
        
        Args:
            predictions: The predicted texts.
            references: The reference texts.
        
        Returns:
            The ROUGE scores.
        """
        scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            rouge_scores = self.rouge_scorer.score(ref, pred)
            for metric, score in rouge_scores.items():
                scores[metric].append(score.fmeasure)
        
        return {
            "rouge1": np.mean(scores["rouge1"]),
            "rouge2": np.mean(scores["rouge2"]),
            "rougeL": np.mean(scores["rougeL"]),
        }
    
    def evaluate_perplexity(self, model, input_ids, target_ids, attention_mask=None):
        """
        Evaluate perplexity.
        
        Args:
            model: The language model.
            input_ids: The input token IDs.
            target_ids: The target token IDs.
            attention_mask: The attention mask.
        
        Returns:
            The perplexity score.
        """
        # Move tensors to the correct device
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            
        # Calculate perplexity
        loss = outputs.loss
        perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def evaluate_hallucination_rate(self, predictions, references, contexts=None, threshold=0.7):
        """
        Evaluate hallucination rate using n-gram overlap.
        
        This is a simple heuristic method that uses n-gram overlap between
        the generated text and the reference or context to estimate hallucination.
        In practice, more sophisticated methods would be used.
        
        Args:
            predictions: The predicted texts.
            references: The reference texts.
            contexts: The context texts (optional).
            threshold: The threshold for classifying a sentence as a hallucination.
        
        Returns:
            The hallucination rate.
        """
        hallucination_rates = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Get the context if available
            context = contexts[i] if contexts else None
            
            # Tokenize the texts
            pred_sentences = sent_tokenize(pred)
            ref_sentences = sent_tokenize(ref)
            context_sentences = sent_tokenize(context) if context else []
            
            # Count hallucinations
            hallucinations = 0
            total_sentences = len(pred_sentences)
            
            for pred_sent in pred_sentences:
                # Check if the sentence is in the reference or context
                is_hallucination = True
                
                # Check against reference
                for ref_sent in ref_sentences:
                    if self._sentence_overlap(pred_sent, ref_sent) >= threshold:
                        is_hallucination = False
                        break
                
                # Check against context if still considered a hallucination
                if is_hallucination and context:
                    for context_sent in context_sentences:
                        if self._sentence_overlap(pred_sent, context_sent) >= threshold:
                            is_hallucination = False
                            break
                
                if is_hallucination:
                    hallucinations += 1
            
            # Calculate hallucination rate
            hallucination_rate = hallucinations / total_sentences if total_sentences > 0 else 0
            hallucination_rates.append(hallucination_rate)
        
        return np.mean(hallucination_rates)
    
    def _sentence_overlap(self, sent1, sent2):
        """
        Calculate the n-gram overlap between two sentences.
        
        Args:
            sent1: The first sentence.
            sent2: The second sentence.
        
        Returns:
            The overlap score.
        """
        # Tokenize the sentences
        tokens1 = word_tokenize(sent1.lower())
        tokens2 = word_tokenize(sent2.lower())
        
        # Create n-grams
        n = 3  # Use trigrams
        ngrams1 = set(self._get_ngrams(tokens1, n))
        ngrams2 = set(self._get_ngrams(tokens2, n))
        
        # Calculate overlap
        overlap = len(ngrams1.intersection(ngrams2))
        total = len(ngrams1) if len(ngrams1) > 0 else 1
        
        return overlap / total
    
    def _get_ngrams(self, tokens, n):
        """
        Generate n-grams from a list of tokens.
        
        Args:
            tokens: The list of tokens.
            n: The n-gram size.
        
        Returns:
            A list of n-grams.
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def evaluate(self, predictions, references, contexts=None, model=None, input_ids=None, target_ids=None, attention_mask=None):
        """
        Evaluate the model performance using multiple metrics.
        
        Args:
            predictions: The predicted texts.
            references: The reference texts.
            contexts: The context texts (optional).
            model: The language model (optional, for perplexity).
            input_ids: The input token IDs (optional, for perplexity).
            target_ids: The target token IDs (optional, for perplexity).
            attention_mask: The attention mask (optional, for perplexity).
        
        Returns:
            A dictionary with the evaluation results.
        """
        results = {}
        
        # BLEU score
        results["bleu"] = self.evaluate_bleu(predictions, references)
        
        # ROUGE scores
        rouge_scores = self.evaluate_rouge(predictions, references)
        results.update(rouge_scores)
        
        # Hallucination rate
        results["hallucination_rate"] = self.evaluate_hallucination_rate(predictions, references, contexts)
        
        # Perplexity (if model and input/target IDs are provided)
        if model is not None and input_ids is not None and target_ids is not None:
            results["perplexity"] = self.evaluate_perplexity(model, input_ids, target_ids, attention_mask)
        
        return results