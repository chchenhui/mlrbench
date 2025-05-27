"""
Main experiment runner for RAG-Informed Dynamic Data Valuation.
"""
import os
import argparse
import logging
import sys
import time
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import List, Dict, Any, Tuple, Optional, Union

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code.utils.data_utils import load_data, create_synthetic_data
from claude_code.models.rag_system import RAGSystem
from claude_code.models.data_valuation import (
    StaticPricing, 
    PopularityBasedPricing, 
    DynamicRAGValuation, 
    DataShapleyValuation,
    DataMarketplace
)
from claude_code.utils.visualization import (
    create_summary_dashboard,
    create_results_markdown,
    generate_comparison_table
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('claude_code', 'run_experiments.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(args):
    """
    Run the main experiment.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Start timing
    start_time = time.time()
    logger.info("Starting RAG-Informed Dynamic Data Valuation experiment")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save experiment parameters
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data or create synthetic data
    try:
        logger.info(f"Loading data from {args.data_dir}...")
        data_chunks, qa_pairs = load_data(args.data_dir)
    except Exception as e:
        logger.warning(f"Failed to load data from {args.data_dir}: {e}")
        logger.info("Creating synthetic data instead...")
        data_chunks, qa_pairs = create_synthetic_data(
            num_chunks=args.num_chunks,
            num_qa_pairs=args.num_qa_pairs,
            seed=args.seed
        )
    
    logger.info(f"Loaded {len(data_chunks)} data chunks and {len(qa_pairs)} QA pairs")
    
    # Initialize RAG system
    logger.info(f"Initializing RAG system with {args.retriever_type} retriever...")
    attribution_methods = args.attribution_methods.split(",")
    
    rag_system = RAGSystem(
        data_chunks=data_chunks,
        retriever_type=args.retriever_type,
        attribution_methods=attribution_methods,
        device=args.device
    )
    
    # Initialize valuation methods
    logger.info("Initializing data valuation methods...")
    static_pricing = StaticPricing(price_per_token=args.static_price_per_token)
    
    popularity_pricing = PopularityBasedPricing(
        base_price=args.popularity_base_price,
        log_factor=args.popularity_log_factor
    )
    
    dynamic_pricing = DynamicRAGValuation(
        attribution_weight=args.dynamic_attribution_weight,
        popularity_weight=args.dynamic_popularity_weight,
        feedback_weight=args.dynamic_feedback_weight,
        recency_weight=args.dynamic_recency_weight,
        base_price=args.dynamic_base_price,
        decay_factor=args.dynamic_decay_factor
    )
    
    data_shapley = DataShapleyValuation()
    
    # Initialize marketplace
    logger.info("Initializing data marketplace...")
    marketplace = DataMarketplace(
        valuation_methods=[static_pricing, popularity_pricing, dynamic_pricing, data_shapley],
        data_chunks=data_chunks,
        transaction_cost=args.transaction_cost,
        save_history=True
    )
    
    # Run initial RAG evaluation on a subset of queries for baseline performance
    logger.info("Running initial RAG evaluation...")
    initial_qa_subset = qa_pairs[:min(args.eval_sample_size, len(qa_pairs))]
    initial_results = rag_system.evaluate_on_qa_pairs(initial_qa_subset, top_k=args.top_k)
    
    logger.info(f"Initial RAG performance: " + 
                f"ROUGE-1={initial_results['metrics']['avg_rouge1']:.4f}, " +
                f"ROUGE-L={initial_results['metrics']['avg_rougeL']:.4f}")
    
    # Save initial results
    with open(os.path.join(args.output_dir, "initial_rag_results.json"), "w") as f:
        # Convert numpy values to float for JSON serialization
        results_json = {
            "metrics": {k: float(v) for k, v in initial_results["metrics"].items()},
            "avg_timings": {k: float(v) for k, v in initial_results["avg_timings"].items()},
        }

        # Skip saving detailed results that might contain non-serializable objects
        json.dump(results_json, f, indent=2)
    
    # Run marketplace simulation
    logger.info(f"Running marketplace simulation for {args.num_iterations} iterations...")
    
    # Track attribution data for visualization
    attribution_data = {}
    
    # Main simulation loop
    for iteration in tqdm(range(args.num_iterations)):
        # Select a random query from the QA dataset
        qa_pair = random.choice(qa_pairs)
        query = qa_pair["question"]
        reference_answer = qa_pair["answer"]
        
        # Process the query through the RAG system
        rag_result = rag_system.process_query(query, top_k=args.top_k)
        
        # Extract attribution scores for visualization
        attribution_data[f"query_{iteration}"] = {}
        for method_name, scores in rag_result["attribution_scores"].items():
            for chunk, score in scores.items():
                attribution_data[f"query_{iteration}"][chunk.chunk_id] = score
        
        # Simulate user feedback based on answer quality
        # In a real system, this would come from actual users
        # Here we'll use ROUGE score as a proxy
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_score = scorer.score(reference_answer, rag_result["answer"])
        user_feedback = rouge_score["rouge1"].fmeasure  # Use F1 score as feedback
        
        # Simulate transactions for all chunks involved
        for chunk, retrieval_score in rag_result["retrieved_chunks"]:
            # Get attribution score for this chunk
            # We'll use the first attribution method if multiple are available
            method_name = list(rag_result["attribution_scores"].keys())[0]
            attribution_score = rag_result["attribution_scores"][method_name].get(chunk, 0.0)
            
            # Simulate transaction
            marketplace.simulate_transaction(
                chunk=chunk,
                user_id=f"user_{iteration % 10}",  # Simulate 10 different users
                query=query,
                answer=rag_result["answer"],
                attribution_score=attribution_score,
                user_feedback=user_feedback,
                timestamp=time.time()
            )
        
        # Update values periodically
        if iteration % args.update_frequency == 0 or iteration == args.num_iterations - 1:
            marketplace.update_values()
            
            # Calculate and log metrics
            ground_truth = {chunk.chunk_id: chunk.quality for chunk in data_chunks}
            metrics = marketplace.calculate_metrics(ground_truth_qualities=ground_truth)
            
            # Log progress
            if iteration % (args.num_iterations // 10) == 0 or iteration == args.num_iterations - 1:
                logger.info(f"Iteration {iteration+1}/{args.num_iterations}")
                for key in ['static_pricing_pearson_correlation', 'popularity_pricing_pearson_correlation', 'dynamic_rag_valuation_pearson_correlation']:
                    if key in metrics:
                        logger.info(f"  {key}: {metrics[key]:.4f}")
    
    # Final marketplace evaluation
    logger.info("Calculating final marketplace metrics...")
    ground_truth = {chunk.chunk_id: chunk.quality for chunk in data_chunks}
    final_metrics = marketplace.calculate_metrics(ground_truth_qualities=ground_truth)
    
    # Save marketplace results
    logger.info(f"Saving marketplace results to {args.output_dir}...")
    marketplace.save_results(args.output_dir)
    
    # Save attribution data for visualization
    with open(os.path.join(args.output_dir, "attribution_data.json"), "w") as f:
        json.dump(attribution_data, f, indent=2)
    
    # Run final RAG evaluation
    logger.info("Running final RAG evaluation...")
    final_qa_subset = qa_pairs[:min(args.eval_sample_size, len(qa_pairs))]
    final_results = rag_system.evaluate_on_qa_pairs(final_qa_subset, top_k=args.top_k)
    
    logger.info(f"Final RAG performance: " + 
                f"ROUGE-1={final_results['metrics']['avg_rouge1']:.4f}, " +
                f"ROUGE-L={final_results['metrics']['avg_rougeL']:.4f}")
    
    # Save final results
    with open(os.path.join(args.output_dir, "final_rag_results.json"), "w") as f:
        # Convert numpy values to float for JSON serialization
        results_json = {
            "metrics": {k: float(v) for k, v in final_results["metrics"].items()},
            "avg_timings": {k: float(v) for k, v in final_results["avg_timings"].items()},
        }

        # Skip saving detailed results that might contain non-serializable objects
        json.dump(results_json, f, indent=2)
    
    # Generate visualizations and results markdown
    logger.info("Generating visualizations and results summary...")
    
    # Get data in the right format for visualization
    price_history_flat = []
    for method_name, chunks in marketplace.price_history.items():
        for chunk_id, history in chunks.items():
            for entry in history:
                price_history_flat.append({
                    "method": method_name,
                    "chunk_id": chunk_id,
                    "timestamp": entry["timestamp"],
                    "price": entry["price"]
                })
    
    chunk_prices = []
    for chunk in data_chunks:
        if hasattr(chunk, 'prices'):
            for method_name, price in chunk.prices.items():
                chunk_prices.append({
                    "chunk_id": chunk.chunk_id,
                    "contributor_id": chunk.contributor_id,
                    "method": method_name,
                    "price": price,
                    "retrieval_count": chunk.retrieval_count,
                    "quality": chunk.quality if hasattr(chunk, 'quality') else None
                })
    
    metrics_history_flat = []
    for metric_name, history in marketplace.metrics_history.items():
        for entry in history:
            metrics_history_flat.append({
                "metric": metric_name,
                "timestamp": entry["timestamp"],
                "value": entry["value"]
            })
    
    # Create results markdown
    markdown = create_results_markdown(
        price_history_flat,
        chunk_prices,
        metrics_history_flat,
        marketplace.transactions,
        final_results,
        args.output_dir
    )
    
    # Save results markdown
    with open(os.path.join(args.output_dir, "results.md"), "w") as f:
        f.write(markdown)
    
    # Log completion time
    logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Results saved to {args.output_dir}")
    
    return {
        "marketplace_metrics": final_metrics,
        "rag_results": final_results
    }

def main():
    parser = argparse.ArgumentParser(description="Run RAG-Informed Dynamic Data Valuation experiments")
    
    # Data parameters
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="claude_code/data",
        help="Directory containing prepared datasets"
    )
    parser.add_argument(
        "--num_chunks", 
        type=int, 
        default=200,
        help="Number of chunks for synthetic data (if data loading fails)"
    )
    parser.add_argument(
        "--num_qa_pairs", 
        type=int, 
        default=50,
        help="Number of QA pairs for synthetic data (if data loading fails)"
    )
    
    # RAG system parameters
    parser.add_argument(
        "--retriever_type", 
        type=str, 
        choices=["bm25", "dense"], 
        default="bm25",
        help="Type of retriever to use"
    )
    parser.add_argument(
        "--attribution_methods", 
        type=str, 
        default="attention",
        help="Attribution methods to use (comma-separated)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=5,
        help="Number of chunks to retrieve per query"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run models on"
    )
    
    # Valuation method parameters
    parser.add_argument("--static_price_per_token", type=float, default=0.01)
    parser.add_argument("--popularity_base_price", type=float, default=1.0)
    parser.add_argument("--popularity_log_factor", type=float, default=2.0)
    parser.add_argument("--dynamic_attribution_weight", type=float, default=0.4)
    parser.add_argument("--dynamic_popularity_weight", type=float, default=0.2)
    parser.add_argument("--dynamic_feedback_weight", type=float, default=0.3)
    parser.add_argument("--dynamic_recency_weight", type=float, default=0.1)
    parser.add_argument("--dynamic_base_price", type=float, default=0.5)
    parser.add_argument("--dynamic_decay_factor", type=float, default=0.01)
    parser.add_argument("--transaction_cost", type=float, default=0.1)
    
    # Experiment parameters
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=100,
        help="Number of marketplace simulation iterations"
    )
    parser.add_argument(
        "--update_frequency", 
        type=int, 
        default=10,
        help="Frequency of value updates during simulation"
    )
    parser.add_argument(
        "--eval_sample_size", 
        type=int, 
        default=20,
        help="Number of QA pairs to use for evaluation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="claude_code/results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    results = run_experiment(args)
    
    # Print key results to console
    print("\nExperiment completed successfully!")
    print("\nKey marketplace metrics:")
    for key in ['static_pricing_pearson_correlation', 'popularity_pricing_pearson_correlation', 'dynamic_rag_valuation_pearson_correlation']:
        if key in results["marketplace_metrics"]:
            print(f"  {key}: {results['marketplace_metrics'][key]:.4f}")
    
    print("\nRAG performance:")
    for key in ['avg_rouge1', 'avg_rouge2', 'avg_rougeL']:
        if key in results["rag_results"]["metrics"]:
            print(f"  {key}: {results['rag_results']['metrics'][key]:.4f}")
    
    print(f"\nFull results saved to {args.output_dir}")

if __name__ == "__main__":
    main()