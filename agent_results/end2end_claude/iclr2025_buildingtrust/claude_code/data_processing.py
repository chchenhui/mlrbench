"""
Data Processing Module for TrustPath Evaluation.

This module handles dataset creation, processing, and management for
evaluating the TrustPath framework.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import anthropic
from anthropic import Anthropic
import nltk
import pandas as pd

from config import DATASET_CONFIG, LLM_CONFIG, DATA_DIR
from fix_anthropic import fix_anthropic_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """
    Generates datasets for evaluating the TrustPath framework.
    
    This class creates questions, generates LLM responses, and annotates
    errors to create a benchmark dataset for evaluation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the dataset generator.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
        """
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        
        self.n_samples = DATASET_CONFIG["n_samples"]
        self.domains = DATASET_CONFIG["domains"]
        self.error_types = DATASET_CONFIG["error_types"]
        
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DatasetGenerator with model: {self.model}")
        logger.info(f"Will generate {self.n_samples} samples across domains: {', '.join(self.domains)}")
    
    async def generate_questions(self, n_questions: int = None) -> List[Dict[str, Any]]:
        """
        Generate questions for dataset creation.
        
        Args:
            n_questions: Number of questions to generate. If None, uses config value.
            
        Returns:
            A list of dictionaries containing questions and metadata
        """
        n_questions = n_questions or self.n_samples
        logger.info(f"Generating {n_questions} questions...")
        
        questions = []
        
        # Distribute questions across domains
        questions_per_domain = n_questions // len(self.domains)
        remaining = n_questions % len(self.domains)
        
        for domain in self.domains:
            domain_questions = questions_per_domain
            if remaining > 0:
                domain_questions += 1
                remaining -= 1
            
            # Generate questions for this domain
            domain_questions_list = await self._generate_domain_questions(domain, domain_questions)
            questions.extend(domain_questions_list)
        
        # Shuffle questions
        random.shuffle(questions)
        
        return questions
    
    async def _generate_domain_questions(self, domain: str, n_questions: int) -> List[Dict[str, Any]]:
        """
        Generate questions for a specific domain.
        
        Args:
            domain: The domain to generate questions for
            n_questions: Number of questions to generate
            
        Returns:
            A list of dictionaries containing questions and metadata
        """
        logger.info(f"Generating {n_questions} questions for domain: {domain}")
        
        # Prepare the question generation prompt
        question_prompt = f"""
        Generate {n_questions} diverse, fact-based questions in the domain of {domain}.
        
        The questions should:
        1. Be specific and focused
        2. Require factual knowledge to answer
        3. Vary in difficulty
        4. Cover different aspects of the {domain} domain
        
        For each question, provide:
        1. The question itself
        2. The specific topic within {domain} that it addresses
        3. The typical level of difficulty (easy, medium, hard)
        
        Return a JSON array with this format:
        [
          {{
            "question": "The question text",
            "topic": "Specific topic",
            "difficulty": "easy/medium/hard"
          }}
        ]
        
        Return ONLY the JSON array, no additional text.
        """
        
        try:
            # Get questions from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens * 2,  # More tokens for multiple questions
                temperature=0.7,  # Higher temperature for diverse questions
                system=f"You are an expert at creating challenging questions related to {domain}. Create diverse questions that require factual knowledge to answer correctly.",
                messages=[
                    {"role": "user", "content": question_prompt}
                ]
            )
            questions_response = await fix_anthropic_response(message)
            
            # Extract the JSON array from the response
            import re
            json_match = re.search(r'\[[\s\S]*\]', questions_response)
            if json_match:
                json_str = json_match.group(0)
                questions_data = json.loads(json_str)
            else:
                logger.warning(f"Could not extract JSON from questions response for {domain}. Creating fallback questions.")
                questions_data = [
                    {
                        "question": f"What are the key developments in {domain} in the 20th century?",
                        "topic": domain,
                        "difficulty": "medium"
                    }
                ] * n_questions
            
            # Ensure we have the requested number of questions
            if len(questions_data) < n_questions:
                logger.warning(f"Generated fewer questions than requested for {domain}. Duplicating to reach target.")
                questions_data = questions_data * (n_questions // len(questions_data) + 1)
                questions_data = questions_data[:n_questions]
            
            # Add domain and ensure consistent structure
            for q in questions_data:
                q["domain"] = domain
                q["question_id"] = f"{domain}_{random.randint(1000, 9999)}"
                
                # Ensure all expected fields are present
                if "topic" not in q:
                    q["topic"] = domain
                if "difficulty" not in q:
                    q["difficulty"] = "medium"
            
            return questions_data[:n_questions]
            
        except Exception as e:
            logger.error(f"Error during question generation for {domain}: {e}")
            # Create fallback questions
            return [
                {
                    "question": f"Question {i+1} about {domain}",
                    "domain": domain,
                    "topic": domain,
                    "difficulty": "medium",
                    "question_id": f"{domain}_{random.randint(1000, 9999)}"
                } for i in range(n_questions)
            ]
    
    async def generate_responses(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate LLM responses to the questions.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of dictionaries containing questions, responses, and metadata
        """
        logger.info(f"Generating responses for {len(questions)} questions...")
        
        samples = []
        
        for i, question in enumerate(questions):
            logger.info(f"Generating response for question {i+1}/{len(questions)}")
            
            question_text = question["question"]
            
            try:
                # Generate a response
                response = await self._generate_response(question_text)
                
                # Add to samples
                sample = {
                    **question,
                    "response": response,
                    "sample_id": f"sample_{random.randint(10000, 99999)}"
                }
                
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error generating response for question {i+1}: {e}")
        
        return samples
    
    async def _generate_response(self, question: str) -> str:
        """
        Generate an LLM response to a question.
        
        Args:
            question: The question to answer
            
        Returns:
            The LLM's response
        """
        try:
            # Get response from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            response = await fix_anthropic_response(message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return f"[Error generating response: {e}]"
    
    async def create_dataset_with_errors(self, n_samples: int = None) -> List[Dict[str, Any]]:
        """
        Create a dataset with intentionally injected errors for evaluation.
        
        Args:
            n_samples: Number of samples to create. If None, uses config value.
            
        Returns:
            List of dictionaries containing samples with injected errors
        """
        n_samples = n_samples or self.n_samples
        logger.info(f"Creating dataset with {n_samples} samples with injected errors...")
        
        # Generate questions
        questions = await self.generate_questions(n_samples)
        
        # Generate responses
        samples = await self.generate_responses(questions)
        
        # Inject errors into half of the samples
        samples_with_errors = []
        
        for i, sample in enumerate(samples):
            # Decide whether to inject errors (approximately half of samples)
            inject_errors = random.random() < 0.5
            
            if inject_errors:
                # Inject errors
                sample_with_errors = await self._inject_errors(sample)
                samples_with_errors.append(sample_with_errors)
            else:
                # Keep the original sample
                sample["has_injected_errors"] = False
                sample["injected_errors"] = []
                samples_with_errors.append(sample)
        
        return samples_with_errors
    
    async def _inject_errors(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject errors into a sample for evaluation.
        
        Args:
            sample: The sample to inject errors into
            
        Returns:
            The sample with injected errors
        """
        question = sample["question"]
        response = sample["response"]
        domain = sample["domain"]
        
        logger.info(f"Injecting errors into sample for question: {question[:50]}...")
        
        # Prompt to inject errors
        injection_prompt = f"""
        Modify the following response to introduce factual errors. The original response is to the question: "{question}"
        
        Original response:
        {response}
        
        Instructions:
        1. Introduce 2-4 factual errors
        2. Include a mix of error types: major factual errors, subtle inaccuracies, and logical inconsistencies
        3. Make the errors plausible enough that they might not be immediately obvious
        4. Do not change the overall structure or style of the response
        5. Keep track of what you changed and why it's an error
        
        Return a JSON object with this format:
        {{
          "modified_response": "The response with injected errors",
          "injected_errors": [
            {{
              "original_text": "The original correct text",
              "modified_text": "The modified text with error",
              "error_type": "factual/logical/vague",
              "explanation": "Why this is an error"
            }}
          ]
        }}
        
        Return ONLY the JSON object, no additional text.
        """
        
        try:
            # Get error injection from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens * 2,
                temperature=0.5,
                system=f"You are an expert at identifying and creating factual errors in text about {domain}. Your task is to introduce plausible but incorrect information.",
                messages=[
                    {"role": "user", "content": injection_prompt}
                ]
            )
            injection_response = await fix_anthropic_response(message)
            
            # Extract the JSON object from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', injection_response)
            if json_match:
                json_str = json_match.group(0)
                injection_data = json.loads(json_str)
            else:
                logger.warning("Could not extract JSON from injection response. Using original response.")
                injection_data = {
                    "modified_response": response,
                    "injected_errors": []
                }
            
            # Update the sample with the injected errors
            sample_with_errors = {
                **sample,
                "response": injection_data.get("modified_response", response),
                "has_injected_errors": bool(injection_data.get("injected_errors", [])),
                "injected_errors": injection_data.get("injected_errors", [])
            }
            
            return sample_with_errors
            
        except Exception as e:
            logger.error(f"Error during error injection: {e}")
            return {
                **sample,
                "has_injected_errors": False,
                "injected_errors": []
            }
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> str:
        """
        Save the dataset to a file.
        
        Args:
            dataset: The dataset to save
            filename: The filename to save to
            
        Returns:
            The path to the saved file
        """
        file_path = DATA_DIR / filename
        
        with open(file_path, "w") as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved to {file_path}")
        return str(file_path)
    
    def load_dataset(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load a dataset from a file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            The loaded dataset
        """
        file_path = DATA_DIR / filename
        
        try:
            with open(file_path, "r") as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded dataset from {file_path} with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            return []

class DatasetProcessor:
    """
    Processes datasets for TrustPath evaluation.
    
    This class splits datasets into train/test sets, generates
    ground truth annotations, and prepares data for evaluation.
    """
    
    def __init__(self):
        """
        Initialize the dataset processor.
        """
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DatasetProcessor")
    
    def split_dataset(self, dataset: List[Dict[str, Any]], test_ratio: float = 0.3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split a dataset into training and test sets.
        
        Args:
            dataset: The dataset to split
            test_ratio: The ratio of samples to allocate to the test set
            
        Returns:
            A tuple (train_dataset, test_dataset)
        """
        # Shuffle the dataset
        dataset_copy = dataset.copy()
        random.shuffle(dataset_copy)
        
        # Split the dataset
        split_idx = int((1 - test_ratio) * len(dataset_copy))
        train_dataset = dataset_copy[:split_idx]
        test_dataset = dataset_copy[split_idx:]
        
        logger.info(f"Split dataset into {len(train_dataset)} training samples and {len(test_dataset)} test samples")
        return train_dataset, test_dataset
    
    def get_ground_truth_annotations(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get ground truth annotations for the dataset.
        
        For samples with injected errors, use those as ground truth.
        For other samples, use model-based detection as ground truth.
        
        Args:
            dataset: The dataset to annotate
            
        Returns:
            The dataset with ground truth annotations
        """
        logger.info(f"Getting ground truth annotations for {len(dataset)} samples...")
        
        annotated_dataset = []
        
        for sample in dataset:
            # If the sample has injected errors, use those as ground truth
            if sample.get("has_injected_errors", False) and sample.get("injected_errors", []):
                ground_truth = {
                    "errors": [
                        {
                            "content": error.get("modified_text", ""),
                            "explanation": error.get("explanation", ""),
                            "error_type": error.get("error_type", "factual")
                        } for error in sample["injected_errors"]
                    ],
                    "corrections": [
                        error.get("original_text", "") for error in sample["injected_errors"]
                    ]
                }
                
                annotated_sample = {
                    **sample,
                    "ground_truth": ground_truth
                }
                
                annotated_dataset.append(annotated_sample)
            else:
                # For samples without injected errors, assume no errors as ground truth
                # In a real implementation, these might need manual annotation
                ground_truth = {
                    "errors": [],
                    "corrections": []
                }
                
                annotated_sample = {
                    **sample,
                    "ground_truth": ground_truth
                }
                
                annotated_dataset.append(annotated_sample)
        
        return annotated_dataset
    
    def save_processed_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> str:
        """
        Save a processed dataset to a file.
        
        Args:
            dataset: The dataset to save
            filename: The filename to save to
            
        Returns:
            The path to the saved file
        """
        file_path = DATA_DIR / filename
        
        with open(file_path, "w") as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Processed dataset saved to {file_path}")
        return str(file_path)
    
    def load_processed_dataset(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load a processed dataset from a file.
        
        Args:
            filename: The filename to load from
            
        Returns:
            The loaded dataset
        """
        file_path = DATA_DIR / filename
        
        try:
            with open(file_path, "r") as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded processed dataset from {file_path} with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading processed dataset from {file_path}: {e}")
            return []
    
    def dataset_to_dataframe(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert a dataset to a pandas DataFrame for easier analysis.
        
        Args:
            dataset: The dataset to convert
            
        Returns:
            A pandas DataFrame
        """
        # Flatten the nested structure for easier analysis
        flattened_data = []
        
        for sample in dataset:
            # Base sample data
            sample_data = {
                "sample_id": sample.get("sample_id", ""),
                "question_id": sample.get("question_id", ""),
                "question": sample.get("question", ""),
                "domain": sample.get("domain", ""),
                "topic": sample.get("topic", ""),
                "difficulty": sample.get("difficulty", ""),
                "response": sample.get("response", ""),
                "has_injected_errors": sample.get("has_injected_errors", False),
                "num_injected_errors": len(sample.get("injected_errors", []))
            }
            
            # Add ground truth if available
            if "ground_truth" in sample:
                sample_data["num_ground_truth_errors"] = len(sample["ground_truth"].get("errors", []))
            
            flattened_data.append(sample_data)
        
        return pd.DataFrame(flattened_data)

# Synchronous wrapper for easier testing
def create_dataset(n_samples: int = None, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper around DatasetGenerator.create_dataset_with_errors.
    
    Args:
        n_samples: Number of samples to create
        api_key: The API key for the LLM service
        
    Returns:
        The created dataset
    """
    import asyncio
    
    generator = DatasetGenerator(api_key=api_key)
    dataset = asyncio.run(generator.create_dataset_with_errors(n_samples))
    return dataset

if __name__ == "__main__":
    # Simple test of the dataset generation functionality
    print("Testing dataset generation...")
    
    # Create a small test dataset
    dataset = create_dataset(2)
    
    # Save the dataset
    generator = DatasetGenerator()
    file_path = generator.save_dataset(dataset, "test_dataset.json")
    print(f"Test dataset saved to {file_path}")
    
    # Process the dataset
    processor = DatasetProcessor()
    annotated_dataset = processor.get_ground_truth_annotations(dataset)
    train_dataset, test_dataset = processor.split_dataset(annotated_dataset)
    
    processor.save_processed_dataset(train_dataset, "test_train_dataset.json")
    processor.save_processed_dataset(test_dataset, "test_test_dataset.json")
    
    # Convert to DataFrame
    df = processor.dataset_to_dataframe(dataset)
    print(df.head())