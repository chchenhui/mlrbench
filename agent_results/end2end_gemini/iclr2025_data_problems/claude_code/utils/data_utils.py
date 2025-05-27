"""
Data processing utilities for RAG-Informed Dynamic Data Valuation framework.
"""
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoTokenizer
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DataChunk:
    """Class representing a data chunk in the marketplace."""
    
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        contributor_id: str,
        quality: float = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a data chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: The text content of the chunk
            source: Source document identifier
            contributor_id: ID of the contributor who provided this chunk
            quality: Ground truth quality score (if available, for evaluation only)
            metadata: Additional metadata about the chunk
        """
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.contributor_id = contributor_id
        self.quality = quality
        self.metadata = metadata or {}
        
        # Marketplace-related attributes
        self.retrieval_count = 0
        self.current_price = 1.0  # Default starting price
        self.value_history = []
        self.attribution_scores = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "contributor_id": self.contributor_id,
            "quality": self.quality,
            "metadata": self.metadata,
            "retrieval_count": self.retrieval_count,
            "current_price": self.current_price,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataChunk':
        """Create chunk from dictionary."""
        chunk = cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source=data["source"],
            contributor_id=data["contributor_id"],
            quality=data.get("quality"),
            metadata=data.get("metadata", {})
        )
        chunk.retrieval_count = data.get("retrieval_count", 0)
        chunk.current_price = data.get("current_price", 1.0)
        return chunk

class DatasetProcessor:
    """Processor for creating data chunks from raw datasets."""
    
    def __init__(self, tokenizer=None, max_chunk_length: int = 512):
        """
        Initialize dataset processor.
        
        Args:
            tokenizer: Tokenizer for determining token counts
            max_chunk_length: Maximum length of a chunk in tokens
        """
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
        
    def chunk_text(self, text: str, source_id: str, contributor_id: str) -> List[DataChunk]:
        """
        Split text into chunks of appropriate size.
        
        Args:
            text: Text to split into chunks
            source_id: Source document identifier
            contributor_id: ID of the contributor who provided this text
            
        Returns:
            List of DataChunk objects
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence)) if self.tokenizer else len(sentence.split())
            
            # If adding this sentence would exceed max length, finalize current chunk
            if current_length + sentence_tokens > self.max_chunk_length and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{source_id}_{len(chunks)}"
                # Assign a synthetic quality score for simulation purposes
                quality = random.uniform(0.1, 1.0)
                chunks.append(DataChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source_id,
                    contributor_id=contributor_id,
                    quality=quality
                ))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{source_id}_{len(chunks)}"
            quality = random.uniform(0.1, 1.0)
            chunks.append(DataChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source_id,
                contributor_id=contributor_id,
                quality=quality
            ))
        
        return chunks

def load_wiki_qa_dataset(
    num_samples: int = 1000, 
    max_chunk_length: int = 512,
    seed: int = 42
) -> Tuple[List[DataChunk], List[Dict[str, str]]]:
    """
    Load Wikipedia articles and Natural Questions for QA evaluation.
    
    Args:
        num_samples: Number of samples to load
        max_chunk_length: Maximum chunk length in tokens
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of data chunks, list of QA pairs)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("Loading Wikipedia and Natural Questions datasets...")
    
    # Load tokenizer for chunking
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    processor = DatasetProcessor(tokenizer, max_chunk_length)
    
    # Load a sample of Wikipedia articles
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train[:5000]")
    
    # Load a sample of Natural Questions
    nq_dataset = load_dataset("natural_questions", split="train[:1000]")
    
    # Process Wikipedia articles into chunks
    all_chunks = []
    contributor_ids = [f"contributor_{i}" for i in range(10)]  # Simulated contributors
    
    for i, article in enumerate(tqdm(wiki_dataset, desc="Processing Wiki articles")):
        if i >= 500:  # Limit the number of articles for manageable processing
            break
            
        contributor_id = random.choice(contributor_ids)
        chunks = processor.chunk_text(
            article["text"], 
            source_id=f"wiki_{article['id']}", 
            contributor_id=contributor_id
        )
        all_chunks.extend(chunks)
    
    # Prepare QA pairs
    qa_pairs = []
    for i, item in enumerate(tqdm(nq_dataset, desc="Processing QA pairs")):
        if i >= num_samples:
            break
            
        if item["annotations"] and item["annotations"][0]["short_answers"]:
            question = item["question"]["text"]
            
            # Extract answer from the annotations
            short_answer = item["annotations"][0]["short_answers"][0]
            if short_answer["text"]:
                answer = short_answer["text"]
                qa_pairs.append({"question": question, "answer": answer})
    
    # Keep only a random subset of chunks to match the desired size
    if len(all_chunks) > num_samples * 5:  # Maintain a 5:1 ratio of chunks to questions
        all_chunks = random.sample(all_chunks, num_samples * 5)
    
    print(f"Created {len(all_chunks)} data chunks and {len(qa_pairs)} QA pairs.")
    return all_chunks, qa_pairs

def create_synthetic_data(num_chunks: int = 1000, num_qa_pairs: int = 200, seed: int = 42) -> Tuple[List[DataChunk], List[Dict[str, str]]]:
    """
    Create synthetic data for testing when real datasets cannot be loaded.
    
    Args:
        num_chunks: Number of data chunks to create
        num_qa_pairs: Number of QA pairs to create
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of data chunks, list of QA pairs)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create synthetic contributors
    contributor_ids = [f"contributor_{i}" for i in range(10)]
    
    # Create synthetic data chunks
    all_chunks = []
    
    topics = ["science", "history", "technology", "arts", "sports"]
    
    for i in range(num_chunks):
        topic = random.choice(topics)
        quality = random.uniform(0.1, 1.0)  # Random quality between 0.1 and 1.0
        
        # Generate synthetic text based on the topic
        if topic == "science":
            text = f"Scientists have discovered a new species of {random.choice(['plant', 'animal', 'bacteria'])} " \
                   f"in the {random.choice(['Amazon rainforest', 'Pacific Ocean', 'Antarctic'])}. " \
                   f"This discovery has implications for understanding {random.choice(['biodiversity', 'evolution', 'climate change'])}."
        elif topic == "history":
            text = f"During the {random.choice(['18th', '19th', '20th'])} century, " \
                   f"{random.choice(['European', 'Asian', 'American'])} societies underwent significant " \
                   f"changes due to {random.choice(['industrialization', 'colonization', 'revolution'])}."
        elif topic == "technology":
            text = f"The development of {random.choice(['artificial intelligence', 'quantum computing', 'blockchain'])} " \
                   f"is expected to revolutionize {random.choice(['healthcare', 'finance', 'transportation'])} " \
                   f"by making systems more {random.choice(['efficient', 'secure', 'intelligent'])}."
        elif topic == "arts":
            text = f"The {random.choice(['Renaissance', 'Baroque', 'Modern'])} period in art " \
                   f"was characterized by {random.choice(['realistic portrayals', 'emotional expressions', 'abstract concepts'])} " \
                   f"that influenced {random.choice(['architecture', 'literature', 'music'])} for centuries."
        else:  # sports
            text = f"{random.choice(['Football', 'Basketball', 'Tennis'])} requires athletes to develop " \
                   f"{random.choice(['strength', 'agility', 'endurance'])} through rigorous training, " \
                   f"which has been studied extensively by {random.choice(['coaches', 'scientists', 'analysts'])}."
        
        chunk = DataChunk(
            chunk_id=f"synthetic_{i}",
            text=text,
            source=f"source_{i // 5}",  # Group chunks into sources
            contributor_id=random.choice(contributor_ids),
            quality=quality,
            metadata={"topic": topic}
        )
        all_chunks.append(chunk)
    
    # Create synthetic QA pairs
    qa_pairs = []
    
    for i in range(num_qa_pairs):
        topic = random.choice(topics)
        
        if topic == "science":
            question = f"What was discovered in the {random.choice(['Amazon rainforest', 'Pacific Ocean', 'Antarctic'])}?"
            answer = f"A new species of {random.choice(['plant', 'animal', 'bacteria'])}"
        elif topic == "history":
            question = f"What caused significant changes in {random.choice(['European', 'Asian', 'American'])} societies?"
            answer = f"{random.choice(['Industrialization', 'Colonization', 'Revolution'])}"
        elif topic == "technology":
            question = f"How might {random.choice(['artificial intelligence', 'quantum computing', 'blockchain'])} change industries?"
            answer = f"By making systems more {random.choice(['efficient', 'secure', 'intelligent'])}"
        elif topic == "arts":
            question = f"What characterized the {random.choice(['Renaissance', 'Baroque', 'Modern'])} period in art?"
            answer = f"{random.choice(['Realistic portrayals', 'Emotional expressions', 'Abstract concepts'])}"
        else:  # sports
            question = f"What do {random.choice(['football', 'basketball', 'tennis'])} athletes need to develop?"
            answer = f"{random.choice(['Strength', 'Agility', 'Endurance'])}"
        
        qa_pairs.append({"question": question, "answer": answer})
    
    print(f"Created {len(all_chunks)} synthetic data chunks and {len(qa_pairs)} synthetic QA pairs.")
    return all_chunks, qa_pairs

def save_data(data_chunks: List[DataChunk], qa_pairs: List[Dict[str, str]], output_dir: str):
    """
    Save data chunks and QA pairs to disk.
    
    Args:
        data_chunks: List of data chunks
        qa_pairs: List of QA pairs
        output_dir: Directory to save data to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data chunks
    chunks_data = [chunk.to_dict() for chunk in data_chunks]
    with open(os.path.join(output_dir, "data_chunks.json"), "w") as f:
        json.dump(chunks_data, f, indent=2)
    
    # Save QA pairs
    with open(os.path.join(output_dir, "qa_pairs.json"), "w") as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"Saved {len(data_chunks)} data chunks and {len(qa_pairs)} QA pairs to {output_dir}")

def load_data(data_dir: str) -> Tuple[List[DataChunk], List[Dict[str, str]]]:
    """
    Load data chunks and QA pairs from disk.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (list of data chunks, list of QA pairs)
    """
    # Load data chunks
    with open(os.path.join(data_dir, "data_chunks.json"), "r") as f:
        chunks_data = json.load(f)
    
    data_chunks = [DataChunk.from_dict(chunk_data) for chunk_data in chunks_data]
    
    # Load QA pairs
    with open(os.path.join(data_dir, "qa_pairs.json"), "r") as f:
        qa_pairs = json.load(f)
    
    print(f"Loaded {len(data_chunks)} data chunks and {len(qa_pairs)} QA pairs from {data_dir}")
    return data_chunks, qa_pairs

if __name__ == "__main__":
    # Sample code to test the data utilities
    output_dir = "data"
    
    # Try to load real data, fall back to synthetic data if that fails
    try:
        chunks, qa_pairs = load_wiki_qa_dataset(num_samples=100)
    except Exception as e:
        print(f"Failed to load real datasets: {e}. Using synthetic data instead.")
        chunks, qa_pairs = create_synthetic_data(num_chunks=500, num_qa_pairs=100)
    
    save_data(chunks, qa_pairs, output_dir)
    
    # Test loading
    loaded_chunks, loaded_qa_pairs = load_data(output_dir)
    print(f"Successfully loaded {len(loaded_chunks)} chunks and {len(loaded_qa_pairs)} QA pairs.")