"""
Data preparation script for RAG-Informed Dynamic Data Valuation experiments.
This script downloads and processes the necessary datasets for the experiments.
"""
import os
import argparse
import logging
import sys
import time
from tqdm import tqdm

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claude_code.utils.data_utils import (
    load_wiki_qa_dataset, 
    create_synthetic_data, 
    save_data
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('claude_code', 'prepare_data.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to prepare datasets for experiments.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting data preparation with the following settings:")
    logger.info(f"- Data type: {args.data_type}")
    logger.info(f"- Number of samples: {args.num_samples}")
    logger.info(f"- Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load or create datasets based on the specified type
        if args.data_type == "wiki_qa":
            logger.info("Loading Wikipedia and Natural Questions datasets...")
            try:
                data_chunks, qa_pairs = load_wiki_qa_dataset(
                    num_samples=args.num_samples,
                    max_chunk_length=args.max_chunk_length,
                    seed=args.seed
                )
            except Exception as e:
                logger.error(f"Failed to load Wiki-QA dataset: {e}")
                logger.info("Falling back to synthetic data...")
                data_chunks, qa_pairs = create_synthetic_data(
                    num_chunks=args.num_samples * 5,  # 5 chunks per QA pair
                    num_qa_pairs=args.num_samples,
                    seed=args.seed
                )
        elif args.data_type == "synthetic":
            logger.info("Creating synthetic dataset...")
            data_chunks, qa_pairs = create_synthetic_data(
                num_chunks=args.num_samples * 5,  # 5 chunks per QA pair
                num_qa_pairs=args.num_samples,
                seed=args.seed
            )
        else:
            logger.error(f"Unknown data type: {args.data_type}")
            return
        
        # Save the datasets
        logger.info(f"Saving datasets to {args.output_dir}...")
        save_data(data_chunks, qa_pairs, args.output_dir)
        
        # Log some statistics
        logger.info(f"Created dataset with {len(data_chunks)} data chunks and {len(qa_pairs)} QA pairs")
        
        # Calculate the number of chunks by contributor
        contributor_counts = {}
        for chunk in data_chunks:
            contributor_counts[chunk.contributor_id] = contributor_counts.get(chunk.contributor_id, 0) + 1
        
        logger.info(f"Data chunks by contributor:")
        for contributor_id, count in contributor_counts.items():
            logger.info(f"- {contributor_id}: {count} chunks")
        
        # Calculate the average chunk quality
        if all(hasattr(chunk, 'quality') and chunk.quality is not None for chunk in data_chunks):
            avg_quality = sum(chunk.quality for chunk in data_chunks) / len(data_chunks)
            logger.info(f"Average chunk quality: {avg_quality:.4f}")
        
        logger.info(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for data valuation experiments")
    
    parser.add_argument(
        "--data_type", 
        type=str, 
        default="synthetic",
        choices=["wiki_qa", "synthetic"],
        help="Type of dataset to prepare (wiki_qa or synthetic)"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples (QA pairs) to prepare"
    )
    
    parser.add_argument(
        "--max_chunk_length", 
        type=int, 
        default=512,
        help="Maximum length of a data chunk in tokens"
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
        default="claude_code/data",
        help="Directory to save the prepared datasets"
    )
    
    args = parser.parse_args()
    main(args)