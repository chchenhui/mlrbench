"""
Data handling module for Reasoning Uncertainty Networks (RUNs) experiment.
"""
import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm

from config import DATA_DIR, DATASET_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Loader for datasets used in the RUNs experiment."""
    
    def __init__(self, dataset_type: str = "scientific"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_type: Type of dataset to load ("scientific", "legal", or "medical")
        """
        self.dataset_type = dataset_type
        self.config = DATASET_CONFIG[dataset_type]
        self.dataset = None
        self.data_path = DATA_DIR / f"{dataset_type}_data.json"
    
    def load_scientific_dataset(self) -> Dataset:
        """Load the scientific reasoning dataset (SciQ)."""
        logger.info(f"Loading scientific dataset: {self.config['name']}")
        
        if os.path.exists(self.data_path):
            logger.info(f"Loading from cached file: {self.data_path}")
            with open(self.data_path, 'r') as f:
                data_dict = json.load(f)
            return Dataset.from_dict(data_dict)
        
        # Load from HuggingFace datasets
        dataset = load_dataset(self.config["name"], split=self.config["split"])
        
        # Process dataset to ensure it has the required fields
        processed_data = self._process_scientific_data(dataset)
        
        # Save processed data
        with open(self.data_path, 'w') as f:
            json.dump(processed_data, f)
        
        return Dataset.from_dict(processed_data)
    
    def _process_scientific_data(self, dataset: Dataset) -> Dict[str, List]:
        """
        Process scientific dataset to ensure it has the required fields.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Processed dataset as dictionary
        """
        questions = []
        contexts = []
        answers = []
        
        for item in tqdm(dataset, desc="Processing scientific data"):
            if "question" in item and "support" in item and "correct_answer" in item:
                questions.append(item["question"])
                contexts.append(item["support"])
                answers.append(item["correct_answer"])
        
        return {
            "question": questions,
            "context": contexts,
            "answer": answers,
            "reasoning_type": [self.config["reasoning_type"]] * len(questions)
        }
    
    def create_legal_dataset(self) -> Dataset:
        """Create a synthetic legal reasoning dataset."""
        logger.info(f"Creating legal dataset of size: {self.config['size']}")
        
        if os.path.exists(self.data_path):
            logger.info(f"Loading from cached file: {self.data_path}")
            with open(self.data_path, 'r') as f:
                data_dict = json.load(f)
            return Dataset.from_dict(data_dict)
        
        # Create synthetic legal reasoning examples
        legal_cases = [
            {
                "title": "Contract Dispute - Smith v. Johnson",
                "description": "A contract dispute where the plaintiff claims the defendant failed to deliver goods as specified in the agreement.",
                "facts": "Smith paid Johnson $5,000 for custom furniture. Johnson delivered the furniture 3 months late and it did not match the specifications."
            },
            {
                "title": "Negligence Claim - Roberts v. City Hospital",
                "description": "A medical negligence claim where the plaintiff alleges improper treatment led to complications.",
                "facts": "Roberts was prescribed medication X despite their medical records showing an allergy. They suffered a severe reaction requiring hospitalization."
            },
            {
                "title": "Property Dispute - Garcia v. Martinez",
                "description": "A property boundary dispute between neighboring landowners.",
                "facts": "Garcia built a fence that Martinez claims encroaches 2 feet onto their property. A survey from 1985 and another from 2010 show different boundary lines."
            },
            # We would add more templates for a real implementation
        ]
        
        questions = []
        contexts = []
        answers = []
        
        # Generate multiple variations of each case
        for _ in range(self.config['size'] // len(legal_cases) + 1):
            for case in legal_cases:
                # Create variations of the question
                question = f"Based on the facts of {case['title']}, {random.choice([
                    'what legal principle is most relevant?',
                    'what would a court likely rule?',
                    'what elements must the plaintiff prove?',
                    'what defense might be most effective?'
                ])}"
                
                # Context is the case facts
                context = f"{case['description']} {case['facts']}"
                
                # For real implementation, we would have proper answer generation
                # This is a placeholder
                answer = "Based on the facts provided, the relevant legal principles include..."
                
                questions.append(question)
                contexts.append(context)
                answers.append(answer)
                
                if len(questions) >= self.config['size']:
                    break
        
        data_dict = {
            "question": questions[:self.config['size']],
            "context": contexts[:self.config['size']],
            "answer": answers[:self.config['size']],
            "reasoning_type": [self.config["reasoning_type"]] * min(len(questions), self.config['size'])
        }
        
        # Save processed data
        with open(self.data_path, 'w') as f:
            json.dump(data_dict, f)
        
        return Dataset.from_dict(data_dict)
    
    def create_medical_dataset(self) -> Dataset:
        """Create a synthetic medical reasoning dataset."""
        logger.info(f"Creating medical dataset of size: {self.config['size']}")
        
        if os.path.exists(self.data_path):
            logger.info(f"Loading from cached file: {self.data_path}")
            with open(self.data_path, 'r') as f:
                data_dict = json.load(f)
            return Dataset.from_dict(data_dict)
        
        # Create synthetic medical reasoning examples
        medical_cases = [
            {
                "condition": "Type 2 Diabetes",
                "symptoms": "Increased thirst, frequent urination, fatigue, blurred vision, slow-healing sores",
                "test_results": "Fasting blood glucose: 180 mg/dL, HbA1c: 8.2%",
                "patient_history": "45-year-old male with obesity, family history of diabetes"
            },
            {
                "condition": "Pneumonia",
                "symptoms": "Fever, cough with yellow sputum, shortness of breath, chest pain, fatigue",
                "test_results": "Elevated WBC count, chest X-ray showing infiltrates in right lower lobe",
                "patient_history": "67-year-old female with history of COPD"
            },
            {
                "condition": "Migraine",
                "symptoms": "Severe throbbing headache, nausea, vomiting, sensitivity to light and sound, visual aura",
                "test_results": "Neurological exam normal, no abnormalities on CT scan",
                "patient_history": "29-year-old female with family history of migraines, reports stress as trigger"
            },
            # We would add more templates for a real implementation
        ]
        
        questions = []
        contexts = []
        answers = []
        
        # Generate multiple variations of each case
        for _ in range(self.config['size'] // len(medical_cases) + 1):
            for case in medical_cases:
                # Create variations of the question
                question = f"Given the patient presentation, {random.choice([
                    'what is the most likely diagnosis?',
                    'what treatment would you recommend?',
                    'what additional tests would you order?',
                    'what are the differential diagnoses to consider?'
                ])}"
                
                # Context combines symptoms, test results, and history
                context = f"Patient presents with the following symptoms: {case['symptoms']}. "
                context += f"Test results: {case['test_results']}. "
                context += f"Patient history: {case['patient_history']}."
                
                # For real implementation, we would have proper answer generation
                # This is a placeholder
                answer = f"The most likely diagnosis is {case['condition']} based on the constellation of symptoms..."
                
                questions.append(question)
                contexts.append(context)
                answers.append(answer)
                
                if len(questions) >= self.config['size']:
                    break
        
        data_dict = {
            "question": questions[:self.config['size']],
            "context": contexts[:self.config['size']],
            "answer": answers[:self.config['size']],
            "reasoning_type": [self.config["reasoning_type"]] * min(len(questions), self.config['size'])
        }
        
        # Save processed data
        with open(self.data_path, 'w') as f:
            json.dump(data_dict, f)
        
        return Dataset.from_dict(data_dict)
    
    def load_dataset(self) -> Dataset:
        """Load or create the appropriate dataset based on the dataset type."""
        if self.dataset_type == "scientific":
            self.dataset = self.load_scientific_dataset()
        elif self.dataset_type == "legal":
            self.dataset = self.create_legal_dataset()
        elif self.dataset_type == "medical":
            self.dataset = self.create_medical_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        logger.info(f"Loaded dataset with {len(self.dataset)} examples")
        return self.dataset
    
    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.dataset is None:
            self.load_dataset()
        
        dataset_dict = self.dataset.train_test_split(test_size=test_size + val_size, seed=seed)
        train_dataset = dataset_dict["train"]
        
        # Split the test portion into validation and test
        if val_size > 0:
            # Calculate the relative size of the validation set within the combined validation+test set
            relative_val_size = val_size / (test_size + val_size)
            temp_dict = dataset_dict["test"].train_test_split(test_size=1-relative_val_size, seed=seed)
            val_dataset = temp_dict["train"]
            test_dataset = temp_dict["test"]
        else:
            val_dataset = Dataset.from_dict({k: [] for k in self.dataset.column_names})
            test_dataset = dataset_dict["test"]
        
        logger.info(f"Split dataset into {len(train_dataset)} train, {len(val_dataset)} validation, and {len(test_dataset)} test examples")
        return train_dataset, val_dataset, test_dataset
    
    def inject_hallucinations(self, dataset: Dataset, hallucination_rate: float = 0.3, hallucination_types: List[str] = None) -> Tuple[Dataset, List[int]]:
        """
        Inject hallucinations into the dataset for evaluation purposes.
        
        Args:
            dataset: Original dataset
            hallucination_rate: Fraction of examples to inject hallucinations into
            hallucination_types: Types of hallucinations to inject ("factual", "logical", "numerical")
            
        Returns:
            Tuple of (modified_dataset, hallucination_indices)
        """
        if hallucination_types is None:
            hallucination_types = ["factual", "logical", "numerical"]
        
        dataset_dict = {k: dataset[k] for k in dataset.column_names}
        num_examples = len(dataset)
        num_hallucinations = int(num_examples * hallucination_rate)
        
        # Randomly select indices to inject hallucinations
        hallucination_indices = random.sample(range(num_examples), num_hallucinations)
        
        # Add hallucination flags
        dataset_dict["contains_hallucination"] = [False] * num_examples
        dataset_dict["hallucination_type"] = ["none"] * num_examples
        
        answers = dataset_dict["answer"].copy()
        
        for idx in hallucination_indices:
            hallucination_type = random.choice(hallucination_types)
            dataset_dict["contains_hallucination"][idx] = True
            dataset_dict["hallucination_type"][idx] = hallucination_type
            
            # Modify the answer based on the hallucination type
            if hallucination_type == "factual":
                # Introduce incorrect facts
                answers[idx] = self._introduce_factual_hallucination(answers[idx])
            elif hallucination_type == "logical":
                # Introduce logical inconsistencies
                answers[idx] = self._introduce_logical_hallucination(answers[idx])
            elif hallucination_type == "numerical":
                # Introduce numerical errors
                answers[idx] = self._introduce_numerical_hallucination(answers[idx])
        
        dataset_dict["answer"] = answers
        
        logger.info(f"Injected hallucinations into {num_hallucinations} examples ({hallucination_rate*100:.1f}%)")
        return Dataset.from_dict(dataset_dict), hallucination_indices
    
    def _introduce_factual_hallucination(self, text: str) -> str:
        """Introduce factual hallucinations into text."""
        # In a real implementation, we would have more sophisticated hallucination introduction
        # This is a simplified version
        
        factual_hallucinations = [
            "It is well established in the scientific literature that",
            "According to a recent study at Harvard University",
            "As demonstrated by comprehensive research",
            "Experts widely agree that",
            "Historical evidence conclusively shows that"
        ]
        
        incorrect_facts = [
            "water boils at 50 degrees Celsius at standard pressure",
            "humans can survive without oxygen for up to 30 minutes",
            "the speed of light varies significantly depending on the season",
            "the human heart has five chambers",
            "consuming vitamin C prevents all viral infections"
        ]
        
        hallucination = f"{random.choice(factual_hallucinations)} {random.choice(incorrect_facts)}. "
        insertion_point = min(len(text) // 2, 100)  # Insert roughly in the middle
        
        return text[:insertion_point] + hallucination + text[insertion_point:]
    
    def _introduce_logical_hallucination(self, text: str) -> str:
        """Introduce logical inconsistencies into text."""
        # In a real implementation, we would have more sophisticated hallucination introduction
        # This is a simplified version
        
        logical_contradictions = [
            "This clearly indicates X, but X is not possible in this situation.",
            "While all evidence points to Y, we can conclusively state that Y is not the case.",
            "The data suggests both A and not-A simultaneously.",
            "If we assume P, then Q follows, but we know Q and not-P are both true.",
            "The patient exhibits symptoms of the condition but definitely does not have the condition."
        ]
        
        hallucination = f" {random.choice(logical_contradictions)} "
        insertion_point = min(len(text) // 2, 100)  # Insert roughly in the middle
        
        return text[:insertion_point] + hallucination + text[insertion_point:]
    
    def _introduce_numerical_hallucination(self, text: str) -> str:
        """Introduce numerical errors into text."""
        # In a real implementation, we would have more sophisticated hallucination introduction
        # This is a simplified version
        
        numerical_hallucinations = [
            "The probability is exactly 150%",
            "The study showed a 200% reduction in symptoms",
            "Adding 25 and 30 gives us 65",
            "When we multiply 7 by 8, we get 48",
            "If we divide 100 by 4, the result is 20"
        ]
        
        hallucination = f" {random.choice(numerical_hallucinations)}. "
        insertion_point = min(len(text) // 2, 100)  # Insert roughly in the middle
        
        return text[:insertion_point] + hallucination + text[insertion_point:]


# Utility function to load all datasets
def load_all_datasets(test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> Dict[str, Dict[str, Dataset]]:
    """
    Load all datasets for the experiment.
    
    Args:
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of datasets by type and split
    """
    datasets = {}
    
    for dataset_type in DATASET_CONFIG:
        loader = DatasetLoader(dataset_type)
        train_dataset, val_dataset, test_dataset = loader.split_dataset(
            test_size=test_size,
            val_size=val_size,
            seed=seed
        )
        
        # Inject hallucinations only in the test set
        test_dataset, hallucination_indices = loader.inject_hallucinations(
            test_dataset,
            hallucination_rate=0.3
        )
        
        datasets[dataset_type] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "hallucination_indices": hallucination_indices
        }
    
    return datasets


if __name__ == "__main__":
    # Test the data module
    print("Testing data module...")
    datasets = load_all_datasets()
    
    for dataset_type, splits in datasets.items():
        print(f"\nDataset: {dataset_type}")
        for split_name, dataset in splits.items():
            if split_name != "hallucination_indices":
                print(f"  {split_name}: {len(dataset)} examples")
        
        # Sample from test set
        test_dataset = splits["test"]
        print("\nSample from test set:")
        for i in range(min(2, len(test_dataset))):
            print(f"\nExample {i}:")
            print(f"  Question: {test_dataset[i]['question']}")
            print(f"  Context: {test_dataset[i]['context'][:100]}...")
            print(f"  Answer: {test_dataset[i]['answer'][:100]}...")
            print(f"  Contains hallucination: {test_dataset[i]['contains_hallucination']}")
            if test_dataset[i]['contains_hallucination']:
                print(f"  Hallucination type: {test_dataset[i]['hallucination_type']}")
    
    print("\nData module test complete.")