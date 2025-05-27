#!/usr/bin/env python3
"""
Test script for the Benchmark Card implementation.
This script creates a simple Benchmark Card and tests its functionality.
"""

import os
import sys
import json
import logging
from main import BenchmarkCard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_benchmark_card():
    """Test the BenchmarkCard class functionality."""
    logger.info("Testing BenchmarkCard class")
    
    # Create a simple benchmark card
    card = BenchmarkCard(
        name="Test Benchmark Card",
        description="A test benchmark card",
        intended_use_cases={
            "use_case_1": "Description of use case 1",
            "use_case_2": "Description of use case 2"
        },
        dataset_composition={
            "name": "test_dataset",
            "num_samples": 1000,
            "num_features": 10
        },
        evaluation_metrics={
            "metric_1": "Description of metric 1",
            "metric_2": "Description of metric 2"
        },
        robustness_metrics={
            "robustness_1": "Description of robustness metric 1"
        },
        limitations=["Limitation 1", "Limitation 2"],
        version="1.0"
    )
    
    # Test adding use case weights
    card.add_use_case_weights("use_case_1", {
        "metric_1": 0.7,
        "metric_2": 0.3
    })
    
    card.add_use_case_weights("use_case_2", {
        "metric_1": 0.2,
        "metric_2": 0.5,
        "robustness_1": 0.3
    })
    
    # Test computing composite score
    try:
        score_1 = card.compute_composite_score(
            {"metric_1": 0.8, "metric_2": 0.6},
            "use_case_1"
        )
        logger.info(f"Composite score for use_case_1: {score_1}")
        
        score_2 = card.compute_composite_score(
            {"metric_1": 0.8, "metric_2": 0.6, "robustness_1": 0.7},
            "use_case_2"
        )
        logger.info(f"Composite score for use_case_2: {score_2}")
        
        # Test with thresholds
        score_with_thresholds = card.compute_composite_score(
            {"metric_1": 0.8, "metric_2": 0.6},
            "use_case_1",
            thresholds={"metric_1": 0.5, "metric_2": 0.7}
        )
        logger.info(f"Composite score with thresholds: {score_with_thresholds}")
        
        logger.info("Composite score computation tests passed")
    except Exception as e:
        logger.error(f"Error computing composite score: {e}")
        return False
    
    # Test serialization
    try:
        card_dict = card.to_dict()
        card_json = card.to_json()
        
        # Save to file
        output_dir = os.path.dirname(os.path.abspath(__file__))
        card.save(os.path.join(output_dir, "test_card.json"))
        
        # Load from file
        loaded_card = BenchmarkCard.load(os.path.join(output_dir, "test_card.json"))
        
        # Check if loaded card has the same properties
        assert loaded_card.name == card.name
        assert loaded_card.description == card.description
        assert loaded_card.intended_use_cases == card.intended_use_cases
        assert loaded_card.use_case_weights == card.use_case_weights
        
        logger.info("Serialization tests passed")
    except Exception as e:
        logger.error(f"Error in serialization tests: {e}")
        return False
    
    # Clean up test file
    try:
        os.remove(os.path.join(output_dir, "test_card.json"))
    except:
        pass
    
    logger.info("All BenchmarkCard tests passed successfully")
    return True


if __name__ == "__main__":
    test_benchmark_card()