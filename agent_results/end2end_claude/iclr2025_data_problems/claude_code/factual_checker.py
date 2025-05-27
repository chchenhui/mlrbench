#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factual checking and verification utilities for attribution evaluation.
"""

import torch
import numpy as np
import re
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import Counter
import difflib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModel
import logging

# Check if NLTK resources are available, and download if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class FactualChecker:
    """
    Utility for checking factual consistency and attribution accuracy.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device = None
    ):
        """
        Initialize the factual checker.
        
        Args:
            similarity_threshold: Threshold for similarity matching
            embedding_model: Model to use for semantic embeddings
            device: Device to use for computation
        """
        self.similarity_threshold = similarity_threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load embedding model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
            self.embedding_model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            logger.info("Falling back to simpler text matching methods")
            self.embedding_model_loaded = False
        
        logger.info(f"Initialized FactualChecker with similarity threshold {similarity_threshold}")
    
    def _get_sentence_embedding(self, text: str) -> torch.Tensor:
        """
        Get sentence embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor
        """
        if not self.embedding_model_loaded:
            return None
        
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512,
                               padding=True).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Normalize
        embedding = embeddings[0] / embeddings[0].norm()
        
        return embedding
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        if self.embedding_model_loaded:
            # Use embedding similarity
            emb1 = self._get_sentence_embedding(text1)
            emb2 = self._get_sentence_embedding(text2)
            similarity = torch.dot(emb1, emb2).item()
        else:
            # Fallback to basic text similarity
            bleu_score = sentence_bleu(
                [word_tokenize(text1)], word_tokenize(text2), 
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            
            # Also use character-level similarity
            seq_matcher = difflib.SequenceMatcher(None, text1, text2)
            char_similarity = seq_matcher.ratio()
            
            # Average the two
            similarity = (bleu_score + char_similarity) / 2
        
        return similarity
    
    def check_content_overlap(
        self,
        generated_text: str,
        source_texts: List[str],
        window_size: int = 5
    ) -> Dict[int, float]:
        """
        Check for content overlap between generated text and source texts.
        
        Args:
            generated_text: Generated text to check
            source_texts: List of source texts to compare against
            window_size: Size of sliding window for comparison
            
        Returns:
            Dictionary mapping source index to overlap score
        """
        # Tokenize texts
        gen_sentences = sent_tokenize(generated_text)
        source_sentences = [sent_tokenize(source) for source in source_texts]
        
        # Compute sentence-level similarities
        overlap_scores = {}
        
        for gen_idx, gen_sent in enumerate(gen_sentences):
            for src_idx, src_sents in enumerate(source_sentences):
                # Skip empty sources
                if not src_sents:
                    continue
                
                # Find most similar sentence
                max_sim = 0
                for src_sent in src_sents:
                    sim = self._compute_similarity(gen_sent, src_sent)
                    max_sim = max(max_sim, sim)
                
                # Store score for this source
                if src_idx in overlap_scores:
                    overlap_scores[src_idx] = max(overlap_scores[src_idx], max_sim)
                else:
                    overlap_scores[src_idx] = max_sim
        
        return overlap_scores
    
    def detect_attributable_segments(
        self,
        generated_text: str,
        source_texts: List[str],
        min_length: int = 10,
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect segments in generated text that should be attributed to sources.
        
        Args:
            generated_text: Generated text to check
            source_texts: List of source texts to compare against
            min_length: Minimum length of segment to consider
            window_size: Size of sliding window for comparison
            
        Returns:
            List of dictionaries with segment information
        """
        # Tokenize texts
        gen_tokens = word_tokenize(generated_text)
        source_tokens = [word_tokenize(source) for source in source_texts]
        
        # Find attributable segments
        attributable_segments = []
        
        for start_idx in range(len(gen_tokens) - min_length + 1):
            # Extract segment
            segment_length = min_length
            while start_idx + segment_length <= len(gen_tokens) and segment_length <= 30:
                segment = gen_tokens[start_idx:start_idx + segment_length]
                segment_text = " ".join(segment)
                
                # Check each source
                for src_idx, src_tokens in enumerate(source_tokens):
                    # Skip if source is too short
                    if len(src_tokens) < min_length:
                        continue
                    
                    # Sliding window over source
                    max_sim = 0
                    for src_start in range(len(src_tokens) - min_length + 1):
                        src_segment_length = segment_length
                        if src_start + src_segment_length > len(src_tokens):
                            src_segment_length = len(src_tokens) - src_start
                            
                        src_segment = src_tokens[src_start:src_start + src_segment_length]
                        src_segment_text = " ".join(src_segment)
                        
                        # Compute similarity
                        sim = self._compute_similarity(segment_text, src_segment_text)
                        max_sim = max(max_sim, sim)
                    
                    # If similarity is high enough, consider it attributable
                    if max_sim >= self.similarity_threshold:
                        attributable_segments.append({
                            "start": start_idx,
                            "end": start_idx + segment_length,
                            "text": segment_text,
                            "source": src_idx,
                            "similarity": max_sim
                        })
                        break
                
                segment_length += 1
        
        # Merge overlapping segments
        if attributable_segments:
            merged_segments = [attributable_segments[0]]
            for segment in attributable_segments[1:]:
                prev_segment = merged_segments[-1]
                
                # Check if segments overlap and have the same source
                if (segment["start"] <= prev_segment["end"] and 
                    segment["source"] == prev_segment["source"]):
                    # Merge segments
                    prev_segment["end"] = max(prev_segment["end"], segment["end"])
                    prev_segment["text"] = " ".join(gen_tokens[prev_segment["start"]:prev_segment["end"]])
                    prev_segment["similarity"] = max(prev_segment["similarity"], segment["similarity"])
                else:
                    # Add as new segment
                    merged_segments.append(segment)
            
            attributable_segments = merged_segments
        
        return attributable_segments
    
    def compute_attribution_metrics(
        self,
        generated_text: str,
        claimed_sources: List[int],
        true_sources: List[int],
        source_texts: List[str]
    ) -> Dict[str, float]:
        """
        Compute attribution metrics for generated text.
        
        Args:
            generated_text: Generated text to evaluate
            claimed_sources: Sources claimed by the model
            true_sources: True sources for the text
            source_texts: List of source texts
            
        Returns:
            Dictionary with attribution metrics
        """
        # Detect attributable segments
        attributable_segments = self.detect_attributable_segments(
            generated_text=generated_text,
            source_texts=source_texts
        )
        
        # Get sources that actually need attribution
        actual_sources = set([segment["source"] for segment in attributable_segments])
        
        # Check claimed sources
        claimed_sources_set = set(claimed_sources)
        true_sources_set = set(true_sources)
        
        # Compute metrics
        # Precision = |correct claimed| / |claimed|
        if len(claimed_sources_set) > 0:
            precision = len(claimed_sources_set.intersection(actual_sources)) / len(claimed_sources_set)
        else:
            precision = 0.0
            
        # Recall = |correct claimed| / |actual|
        if len(actual_sources) > 0:
            recall = len(claimed_sources_set.intersection(actual_sources)) / len(actual_sources)
        else:
            # If no attribution needed, recall is perfect
            recall = 1.0
            
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        # Content originality
        attributable_content = sum(segment["end"] - segment["start"] for segment in attributable_segments)
        total_content = len(word_tokenize(generated_text))
        
        if total_content > 0:
            content_originality = 1 - (attributable_content / total_content)
        else:
            content_originality = 1.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "content_originality": content_originality,
            "attributable_segments": len(attributable_segments),
            "attributable_content": attributable_content,
            "total_content": total_content
        }
    
    def evaluate_attribution_quality(
        self,
        generated_texts: List[str],
        claimed_sources_list: List[List[int]],
        true_sources_list: List[List[int]],
        source_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate attribution quality across multiple generated texts.
        
        Args:
            generated_texts: List of generated texts to evaluate
            claimed_sources_list: List of claimed sources for each text
            true_sources_list: List of true sources for each text
            source_texts: List of source texts
            
        Returns:
            Dictionary with averaged attribution metrics
        """
        all_metrics = []
        
        for gen_text, claimed, true in zip(generated_texts, claimed_sources_list, true_sources_list):
            metrics = self.compute_attribution_metrics(
                generated_text=gen_text,
                claimed_sources=claimed,
                true_sources=true,
                source_texts=source_texts
            )
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != "attributable_segments":
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            
        # Also include total counts
        avg_metrics["total_attributable_segments"] = sum(m["attributable_segments"] for m in all_metrics)
        avg_metrics["total_samples"] = len(generated_texts)
        
        return avg_metrics

class VerificationAgent:
    """
    Agent for verifying attributions in generated text.
    This is a lightweight version that can be used without complex NLP pipelines.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        device = None
    ):
        """
        Initialize the verification agent.
        
        Args:
            similarity_threshold: Threshold for similarity matching
            device: Device to use for computation
        """
        self.factual_checker = FactualChecker(
            similarity_threshold=similarity_threshold,
            device=device
        )
        
        logger.info(f"Initialized VerificationAgent with similarity threshold {similarity_threshold}")
    
    def _extract_citations(self, text: str) -> List[Tuple[str, List[int]]]:
        """
        Extract citations from text using regular expressions.
        
        Args:
            text: Input text with citations
            
        Returns:
            List of tuples (text segment, cited sources)
        """
        # Pattern for finding citations like [1], [2, 3], etc.
        citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        
        # Find all citations
        citations = re.finditer(citation_pattern, text)
        
        # Extract citation positions
        positions = []
        for match in citations:
            # Extract source numbers
            sources_str = match.group(1)
            sources = [int(s.strip()) for s in sources_str.split(',')]
            
            # Store position and sources
            positions.append((match.span(), sources))
        
        # Split text by citations
        if not positions:
            # No citations found
            return [(text, [])]
        
        segments = []
        prev_end = 0
        
        for (start, end), sources in positions:
            # Add segment before citation
            if start > prev_end:
                segment_text = text[prev_end:start].strip()
                if segment_text:
                    segments.append((segment_text, []))
            
            # Get preceding text to associate with this citation
            # Either from previous citation or start of text
            segment_start = prev_end
            segment_text = text[segment_start:start].strip()
            
            if segment_text:
                segments.append((segment_text, sources))
            
            prev_end = end
        
        # Add final segment if needed
        if prev_end < len(text):
            segment_text = text[prev_end:].strip()
            if segment_text:
                segments.append((segment_text, []))
        
        return segments
    
    def verify_attributions(
        self,
        generated_text: str,
        source_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Verify attributions in generated text.
        
        Args:
            generated_text: Generated text with citations
            source_texts: List of source texts
            
        Returns:
            Dictionary with verification results
        """
        # Extract citations
        segments = self._extract_citations(generated_text)
        
        # Verify each segment
        verified_segments = []
        correct_attributions = 0
        incorrect_attributions = 0
        missing_attributions = 0
        
        for segment_text, claimed_sources in segments:
            # Detect if this segment needs attribution
            attributable_segments = self.factual_checker.detect_attributable_segments(
                generated_text=segment_text,
                source_texts=source_texts
            )
            
            # Get actual sources
            actual_sources = [s["source"] for s in attributable_segments]
            
            # Check claimed vs. actual
            if claimed_sources:
                # There are claimed sources for this segment
                claimed_set = set(claimed_sources)
                actual_set = set(actual_sources)
                
                # Calculate correct and incorrect attributions
                correct = len(claimed_set.intersection(actual_set))
                incorrect = len(claimed_set - actual_set)
                missing = len(actual_set - claimed_set)
                
                correct_attributions += correct
                incorrect_attributions += incorrect
                missing_attributions += missing
                
                verified_segments.append({
                    "text": segment_text,
                    "claimed_sources": claimed_sources,
                    "actual_sources": actual_sources,
                    "correct": correct,
                    "incorrect": incorrect,
                    "missing": missing
                })
            elif actual_sources:
                # No claimed sources but there should be
                missing_attributions += len(actual_sources)
                
                verified_segments.append({
                    "text": segment_text,
                    "claimed_sources": [],
                    "actual_sources": actual_sources,
                    "correct": 0,
                    "incorrect": 0,
                    "missing": len(actual_sources)
                })
            else:
                # Correctly no attribution
                verified_segments.append({
                    "text": segment_text,
                    "claimed_sources": [],
                    "actual_sources": [],
                    "correct": 0,
                    "incorrect": 0,
                    "missing": 0
                })
        
        # Calculate metrics
        total_claimed = correct_attributions + incorrect_attributions
        total_needed = correct_attributions + missing_attributions
        
        if total_claimed > 0:
            precision = correct_attributions / total_claimed
        else:
            precision = 1.0 if total_needed == 0 else 0.0
            
        if total_needed > 0:
            recall = correct_attributions / total_needed
        else:
            recall = 1.0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "segments": verified_segments,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct_attributions": correct_attributions,
                "incorrect_attributions": incorrect_attributions,
                "missing_attributions": missing_attributions
            }
        }
    
    def batch_verify(
        self,
        generated_texts: List[str],
        source_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Verify attributions for multiple generated texts.
        
        Args:
            generated_texts: List of generated texts with citations
            source_texts: List of source texts
            
        Returns:
            Dictionary with verification results
        """
        results = []
        
        for text in generated_texts:
            result = self.verify_attributions(text, source_texts)
            results.append(result)
        
        # Aggregate metrics
        total_correct = sum(r["metrics"]["correct_attributions"] for r in results)
        total_incorrect = sum(r["metrics"]["incorrect_attributions"] for r in results)
        total_missing = sum(r["metrics"]["missing_attributions"] for r in results)
        
        total_claimed = total_correct + total_incorrect
        total_needed = total_correct + total_missing
        
        if total_claimed > 0:
            precision = total_correct / total_claimed
        else:
            precision = 1.0 if total_needed == 0 else 0.0
            
        if total_needed > 0:
            recall = total_correct / total_needed
        else:
            recall = 1.0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "individual_results": results,
            "aggregate_metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct_attributions": total_correct,
                "incorrect_attributions": total_incorrect,
                "missing_attributions": total_missing,
                "total_samples": len(generated_texts)
            }
        }

if __name__ == "__main__":
    # Test the factual checker and verification agent
    logging.basicConfig(level=logging.INFO)
    
    # Sample texts
    source_texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the English alphabet.",
        "Python is a high-level, interpreted programming language known for its readability and simplicity.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to learn from data without being explicitly programmed."
    ]
    
    generated_text = "I think Python is a great programming language because of its readability and simplicity [1]. " \
                    "Machine learning systems can learn from data without explicit programming [2]. " \
                    "Did you know the quick brown fox sentence contains all letters of the alphabet [0]?"
    
    claimed_sources = [1, 2, 0]
    true_sources = [1, 2, 0]
    
    print("Testing FactualChecker...")
    
    checker = FactualChecker()
    
    overlap_scores = checker.check_content_overlap(
        generated_text=generated_text,
        source_texts=source_texts
    )
    
    print(f"Content overlap scores: {overlap_scores}")
    
    segments = checker.detect_attributable_segments(
        generated_text=generated_text,
        source_texts=source_texts
    )
    
    print(f"Attributable segments: {segments}")
    
    metrics = checker.compute_attribution_metrics(
        generated_text=generated_text,
        claimed_sources=claimed_sources,
        true_sources=true_sources,
        source_texts=source_texts
    )
    
    print(f"Attribution metrics: {metrics}")
    
    print("\nTesting VerificationAgent...")
    
    agent = VerificationAgent()
    
    verification = agent.verify_attributions(
        generated_text=generated_text,
        source_texts=source_texts
    )
    
    print(f"Verification results: {verification}")
    
    batch_results = agent.batch_verify(
        generated_texts=[generated_text, generated_text],
        source_texts=source_texts
    )
    
    print(f"Batch verification aggregate metrics: {batch_results['aggregate_metrics']}")
    
    print("All tests passed!")