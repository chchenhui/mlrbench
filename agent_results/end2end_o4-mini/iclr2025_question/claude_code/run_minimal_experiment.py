#!/usr/bin/env python3
"""
Minimal experiment runner for testing the SCEC implementation.

This script runs a simplified version of the SCEC experiment on a small subset of data
to verify that all components work correctly.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
import traceback

import numpy as np
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('minimal_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_synthetic_dataset(num_examples=10):
    """
    Create a synthetic QA dataset for testing.
    
    Args:
        num_examples: Number of examples to create
        
    Returns:
        Tuple of (examples, references)
    """
    # Create synthetic examples
    examples = []
    references = []
    
    qa_pairs = [
        ("What is the capital of France?", ["Paris", "Paris, France"]),
        ("Who wrote the novel '1984'?", ["George Orwell", "Eric Arthur Blair"]),
        ("What is the largest planet in our solar system?", ["Jupiter"]),
        ("What is the chemical symbol for gold?", ["Au"]),
        ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "Leonardo"]),
        ("What is the tallest mountain in the world?", ["Mount Everest", "Everest"]),
        ("What year did World War II end?", ["1945"]),
        ("What is the capital of Japan?", ["Tokyo"]),
        ("Who discovered penicillin?", ["Alexander Fleming", "Fleming"]),
        ("What is the chemical formula for water?", ["H2O"]),
        ("What is the largest ocean on Earth?", ["Pacific Ocean", "Pacific"]),
        ("Who was the first person to walk on the moon?", ["Neil Armstrong", "Armstrong"]),
        ("What is the speed of light in vacuum?", ["299,792,458 m/s", "3×10^8 m/s", "300,000 km/s"]),
        ("What is the capital of Australia?", ["Canberra"]),
        ("Who wrote 'Romeo and Juliet'?", ["William Shakespeare", "Shakespeare"]),
    ]
    
    # Ensure we have enough QA pairs
    if num_examples > len(qa_pairs):
        # Repeat QA pairs if needed
        qa_pairs = qa_pairs * (num_examples // len(qa_pairs) + 1)
    
    # Select subset
    selected_pairs = qa_pairs[:num_examples]
    
    # Format examples
    for i, (question, answer) in enumerate(selected_pairs):
        examples.append({
            "id": f"example-{i}",
            "prompt": f"Question: {question}\nAnswer:",
            "question": question,
        })
        references.append(answer)
    
    return examples, references


def run_minimal_experiment(args):
    """Run a minimal experiment to test the SCEC implementation."""
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create synthetic dataset
    examples, references = create_synthetic_dataset(args.num_examples)
    logger.info(f"Created synthetic dataset with {len(examples)} examples")
    
    # Import modules only when needed to avoid loading everything for a minimal test
    from models.llm_interface import get_llm_interface
    from models.self_consistency import SelfConsistencySampler
    from models.evidence_retrieval import (
        ClaimExtractor, 
        EntailmentScorer, 
        WikipediaRetriever,
        create_synthetic_corpus
    )
    
    # Initialize LLM
    logger.info(f"Initializing LLM: {args.model}")
    llm = get_llm_interface(model_name=args.model)
    
    # Test LLM generation
    logger.info("Testing LLM generation")
    try:
        test_prompt = "What is the capital of France?"
        result = llm.generate(prompt=test_prompt, max_tokens=100)
        logger.info(f"LLM test response: {result['text']}")
    except Exception as e:
        logger.error(f"Error testing LLM generation: {e}")
        return
    
    # Initialize self-consistency sampler
    logger.info("Initializing self-consistency sampler")
    sampler = SelfConsistencySampler(
        llm=llm,
        num_samples=args.k,
        temperature=0.7,
        use_cot=True,
    )
    
    # Test self-consistency sampling
    logger.info("Testing self-consistency sampling")
    try:
        sc_results = sampler.analyze_samples(examples[0]["prompt"], max_tokens=100)
        logger.info(f"Generated {len(sc_results['samples'])} samples")
        # Save a sample to verify
        sample_path = os.path.join(args.output_dir, "sample_sc_result.json")
        with open(sample_path, 'w') as f:
            json.dump(sc_results, f, indent=2)
        logger.info(f"Saved sample self-consistency result to {sample_path}")
    except Exception as e:
        logger.error(f"Error testing self-consistency sampling: {e}")
        traceback.print_exc()
        return
    
    # Create synthetic corpus for retriever
    logger.info("Creating synthetic corpus for retriever")
    corpus_path = os.path.join(args.output_dir, "synthetic_corpus.json")
    try:
        create_synthetic_corpus(corpus_path, num_documents=100)
    except Exception as e:
        logger.error(f"Error creating synthetic corpus: {e}")
        return
    
    # Initialize evidence retrieval components
    logger.info("Initializing evidence retrieval components")
    try:
        from models.evidence_retrieval import BM25Retriever, EvidenceAligner
        
        claim_extractor = ClaimExtractor(use_spacy=False)  # Use simple regex for testing
        entailment_scorer = EntailmentScorer(model_name="facebook/bart-large-mnli")
        retriever = BM25Retriever(corpus_path=corpus_path)
        aligner = EvidenceAligner(retriever, claim_extractor, entailment_scorer)
    except Exception as e:
        logger.error(f"Error initializing evidence retrieval: {e}")
        traceback.print_exc()
        return
    
    # Test evidence alignment
    logger.info("Testing evidence alignment")
    try:
        test_text = "Paris is the capital of France."
        alignment_result = aligner.align(test_text, k_evidence=2)
        logger.info(f"Evidence alignment score: {alignment_result['overall_agreement']}")
        # Save alignment result
        alignment_path = os.path.join(args.output_dir, "sample_alignment_result.json")
        with open(alignment_path, 'w') as f:
            json.dump(alignment_result, f, indent=2)
        logger.info(f"Saved sample alignment result to {alignment_path}")
    except Exception as e:
        logger.error(f"Error testing evidence alignment: {e}")
        traceback.print_exc()
        return
    
    # Initialize uncertainty scorer
    logger.info("Initializing uncertainty scorer")
    try:
        from models.uncertainty_scoring import UncertaintyScorer
        
        scorer = UncertaintyScorer(
            self_consistency_sampler=sampler,
            evidence_aligner=aligner,
            alpha=args.alpha,
        )
    except Exception as e:
        logger.error(f"Error initializing uncertainty scorer: {e}")
        traceback.print_exc()
        return
    
    # Test uncertainty scoring
    logger.info("Testing uncertainty scoring")
    try:
        uncertainty_result = scorer.get_token_uncertainty(examples[0]["prompt"])
        logger.info(f"Uncertainty score: {uncertainty_result.get('sequence_uncertainty', 'N/A')}")
        # Save uncertainty result
        uncertainty_path = os.path.join(args.output_dir, "sample_uncertainty_result.json")
        with open(uncertainty_path, 'w') as f:
            json.dump(uncertainty_result, f, indent=2)
        logger.info(f"Saved sample uncertainty result to {uncertainty_path}")
    except Exception as e:
        logger.error(f"Error testing uncertainty scoring: {e}")
        traceback.print_exc()
        return
    
    # Test guided decoding
    logger.info("Testing guided decoding")
    try:
        from models.guided_decoding import APIGuidedDecoder
        
        decoder = APIGuidedDecoder(
            llm=llm,
            uncertainty_scorer=scorer,
            beta=args.beta,
        )
        
        guided_result = decoder.generate(examples[0]["prompt"], max_tokens=100)
        logger.info(f"Guided generation result: {guided_result['text']}")
        # Save guided decoding result
        decoding_path = os.path.join(args.output_dir, "sample_guided_result.json")
        with open(decoding_path, 'w') as f:
            json.dump(guided_result, f, indent=2)
        logger.info(f"Saved sample guided result to {decoding_path}")
    except Exception as e:
        logger.error(f"Error testing guided decoding: {e}")
        traceback.print_exc()
        return
    
    # Test evaluation metrics
    logger.info("Testing evaluation metrics")
    try:
        from utils.evaluation import QAMetrics, CalibrationMetrics
        
        # Simple test of metrics
        prediction = "Paris"
        reference = ["Paris", "Paris, France"]
        em = QAMetrics.exact_match(prediction, reference)
        f1 = QAMetrics.f1_score(prediction, reference)
        
        logger.info(f"Test metrics - EM: {em}, F1: {f1}")
        
        # Test calibration metrics
        confidence_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        correctness = [True, True, False, False, False]
        
        ece = CalibrationMetrics.expected_calibration_error(confidence_scores, correctness)
        brier = CalibrationMetrics.brier_score(confidence_scores, correctness)
        
        logger.info(f"Test calibration metrics - ECE: {ece}, Brier: {brier}")
    except Exception as e:
        logger.error(f"Error testing evaluation metrics: {e}")
        traceback.print_exc()
        return
    
    # Test visualization utilities
    logger.info("Testing visualization utilities")
    try:
        from utils.visualization import CalibrationPlots, TaskPerformancePlots
        import matplotlib.pyplot as plt
        
        # Simple calibration plot
        fig = CalibrationPlots.reliability_diagram(
            confidence_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            correctness=[True, True, True, False, False, False, False, False, False],
            method_name="Test",
            num_bins=5
        )
        
        # Save figure
        viz_path = os.path.join(args.output_dir, "sample_calibration_plot.png")
        fig.savefig(viz_path)
        plt.close(fig)
        logger.info(f"Saved sample visualization to {viz_path}")
    except Exception as e:
        logger.error(f"Error testing visualization utilities: {e}")
        traceback.print_exc()
        return
    
    logger.info("All components tested successfully!")
    logger.info(f"Check {args.output_dir} for test outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal SCEC experiment for testing")
    
    # Model selection
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet",
                        help="Model to use (claude-3-7-sonnet, gpt-4o-mini, etc.)")
    
    # SCEC parameters
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for balancing variance and evidence alignment")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Strength of hallucination penalty")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of samples for self-consistency")
    
    # Dataset size
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of synthetic examples to create")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="test_outputs",
                        help="Directory to save test outputs")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    run_minimal_experiment(args)