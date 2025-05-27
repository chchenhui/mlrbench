"""
RAG system implementation with attribution mechanisms.
"""
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import time
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    T5ForConditionalGeneration, 
    BartForConditionalGeneration
)
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rouge_score import rouge_scorer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Retriever:
    """Base class for retrieval methods."""
    
    def __init__(self, data_chunks: List[Any], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize retriever.
        
        Args:
            data_chunks: List of data chunks to retrieve from
            device: Device to run models on
        """
        self.data_chunks = data_chunks
        self.device = device
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Retrieve top-k relevant chunks for the query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples
        """
        raise NotImplementedError("Subclasses must implement retrieve method")

class BM25Retriever(Retriever):
    """BM25-based sparse retriever."""
    
    def __init__(self, data_chunks: List[Any], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize BM25 retriever.
        
        Args:
            data_chunks: List of data chunks to retrieve from
            device: Device to run models on (not used for BM25 but kept for API consistency)
        """
        super().__init__(data_chunks, device)
        
        # Preprocess chunks for BM25
        self.chunk_texts = [chunk.text for chunk in data_chunks]
        
        # Tokenize and create BM25 model
        self.tokenized_corpus = []
        stop_words = set(stopwords.words('english'))
        
        for text in self.chunk_texts:
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
            self.tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 retriever initialized with", len(self.tokenized_corpus), "documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Retrieve top-k relevant chunks for the query using BM25.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples
        """
        # Tokenize query
        tokenized_query = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        tokenized_query = [token for token in tokenized_query if token.isalnum() and token not in stop_words]
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return chunks and scores
        results = [(self.data_chunks[idx], scores[idx]) for idx in top_indices]
        
        # Increment retrieval count for retrieved chunks
        for chunk, _ in results:
            chunk.retrieval_count += 1
        
        return results

class DenseRetriever(Retriever):
    """Dense retriever using sentence transformers."""
    
    def __init__(
        self, 
        data_chunks: List[Any], 
        model_name: str = "sentence-transformers/all-mpnet-base-v2", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize dense retriever.
        
        Args:
            data_chunks: List of data chunks to retrieve from
            model_name: Name of the sentence transformer model to use
            device: Device to run model on
        """
        super().__init__(data_chunks, device)
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self.chunk_texts = [chunk.text for chunk in data_chunks]
        
        # Encode all chunks
        print("Encoding chunks for dense retrieval...")
        self.chunk_embeddings = self.model.encode(self.chunk_texts, show_progress_bar=True, convert_to_tensor=True)
        print(f"Dense retriever initialized with {len(self.chunk_texts)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Retrieve top-k relevant chunks for the query using dense embeddings.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples
        """
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        
        # Get top-k indices
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores))).indices.cpu().numpy()
        
        # Return chunks and scores
        results = [(self.data_chunks[idx], cos_scores[idx].item()) for idx in top_indices]
        
        # Increment retrieval count for retrieved chunks
        for chunk, _ in results:
            chunk.retrieval_count += 1
        
        return results

class Generator:
    """Generator model that produces outputs based on query and retrieved documents."""
    
    def __init__(
        self, 
        model_name: str = "google-t5/t5-base", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 128
    ):
        """
        Initialize generator.
        
        Args:
            model_name: Name of the model to use
            device: Device to run model on
            max_length: Maximum length of generated outputs
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        print(f"Generator initialized with model {model_name}")
    
    def generate(self, query: str, retrieved_chunks: List[Tuple[Any, float]]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate answer based on query and retrieved chunks.
        
        Args:
            query: Query string
            retrieved_chunks: List of (chunk, score) tuples from retriever
            
        Returns:
            Tuple of (generated answer, attribution info)
        """
        # Prepare context by concatenating retrieved chunks
        context = ""
        for chunk, score in retrieved_chunks:
            context += chunk.text + " "
        
        # Prepare input
        input_text = f"question: {query} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                output_attentions=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_ids = outputs.sequences[0]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Return answer and attribution data
        attribution_info = {
            "chunks": [chunk for chunk, _ in retrieved_chunks],
            "generation_info": outputs
        }
        
        return answer, attribution_info

class AttentionBasedAttributor:
    """Attributes output to input chunks based on attention weights."""
    
    def __init__(self, generator: Generator):
        """
        Initialize attention-based attributor.
        
        Args:
            generator: Generator model to extract attention from
        """
        self.generator = generator
    
    def attribute(
        self, 
        query: str, 
        retrieved_chunks: List[Tuple[Any, float]], 
        answer: str, 
        generation_info: Dict[str, Any]
    ) -> Dict[Any, float]:
        """
        Compute attribution scores for retrieved chunks.
        
        Args:
            query: Original query
            retrieved_chunks: List of (chunk, score) tuples from retriever
            answer: Generated answer
            generation_info: Additional info from generation process
            
        Returns:
            Dictionary mapping chunks to attribution scores
        """
        # In a real implementation, we would extract and analyze attention patterns
        # from the transformer model. Here, we use a simplified approach for demonstration.
        
        # Extract chunks
        chunks = [chunk for chunk, _ in retrieved_chunks]
        
        # Simplified attribution based on retrieval scores and simulated attention
        attribution_scores = {}
        
        # Normalize retrieval scores
        retrieval_scores = np.array([score for _, score in retrieved_chunks])
        if np.sum(retrieval_scores) > 0:
            retrieval_scores = retrieval_scores / np.sum(retrieval_scores)
        
        # For each chunk, compute an attribution score (simplified simulation)
        for i, chunk in enumerate(chunks):
            # In a real implementation, we would analyze the attention weights
            # between the generated answer tokens and the tokens of each chunk
            
            # For this demonstration, we'll use a combination of:
            # 1. Retrieval score
            # 2. Term overlap between answer and chunk
            
            chunk_terms = set(word_tokenize(chunk.text.lower()))
            answer_terms = set(word_tokenize(answer.lower()))
            
            overlap_ratio = len(chunk_terms.intersection(answer_terms)) / max(1, len(answer_terms))
            
            # Combine factors for final score
            attribution_scores[chunk] = 0.3 * retrieval_scores[i] + 0.7 * overlap_ratio
        
        # Normalize the attribution scores
        total_score = sum(attribution_scores.values())
        if total_score > 0:
            for chunk in attribution_scores:
                attribution_scores[chunk] /= total_score
        
        return attribution_scores

class PerturbationBasedAttributor:
    """Attributes output to input chunks based on leave-one-out perturbation."""
    
    def __init__(self, generator: Generator):
        """
        Initialize perturbation-based attributor.
        
        Args:
            generator: Generator model to use for perturbation analysis
        """
        self.generator = generator
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    def attribute(
        self, 
        query: str, 
        retrieved_chunks: List[Tuple[Any, float]], 
        answer: str, 
        generation_info: Dict[str, Any]
    ) -> Dict[Any, float]:
        """
        Compute attribution scores using leave-one-out perturbation.
        
        Args:
            query: Original query
            retrieved_chunks: List of (chunk, score) tuples from retriever
            answer: Generated answer
            generation_info: Additional info from generation process
            
        Returns:
            Dictionary mapping chunks to attribution scores
        """
        chunks = [chunk for chunk, _ in retrieved_chunks]
        
        # If only one chunk, give it full attribution
        if len(chunks) == 1:
            return {chunks[0]: 1.0}
        
        attribution_scores = {}
        
        # For each chunk, generate an answer without it and measure the difference
        for i, chunk_to_remove in enumerate(chunks):
            # Get all chunks except the current one
            reduced_chunks = [
                (chunk, score) for j, (chunk, score) in enumerate(retrieved_chunks) if j != i
            ]
            
            # Generate answer without this chunk
            reduced_answer, _ = self.generator.generate(query, reduced_chunks)
            
            # Measure difference from original answer using ROUGE score
            rouge_scores = self.scorer.score(answer, reduced_answer)
            rouge1_recall = rouge_scores['rouge1'].recall
            
            # The more the answer changes (lower ROUGE score when chunk is removed),
            # the more important the chunk is
            attribution_scores[chunk_to_remove] = 1.0 - rouge1_recall
        
        # Normalize scores
        total_score = sum(attribution_scores.values())
        if total_score > 0:
            for chunk in attribution_scores:
                attribution_scores[chunk] /= total_score
        
        return attribution_scores

class RAGSystem:
    """Complete RAG system with attribution capabilities."""
    
    def __init__(
        self, 
        data_chunks: List[Any],
        retriever_type: str = "bm25",
        generator_model: str = "google-t5/t5-base",
        attribution_methods: List[str] = ["attention"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RAG system.
        
        Args:
            data_chunks: List of data chunks to retrieve from
            retriever_type: Type of retriever to use ('bm25' or 'dense')
            generator_model: Name of the generator model to use
            attribution_methods: List of attribution methods to use
            device: Device to run models on
        """
        self.data_chunks = data_chunks
        self.device = device
        
        # Initialize retriever
        if retriever_type == "bm25":
            self.retriever = BM25Retriever(data_chunks, device=device)
        elif retriever_type == "dense":
            self.retriever = DenseRetriever(data_chunks, device=device)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        # Initialize generator
        self.generator = Generator(model_name=generator_model, device=device)
        
        # Initialize attributors
        self.attributors = {}
        if "attention" in attribution_methods:
            self.attributors["attention"] = AttentionBasedAttributor(self.generator)
        if "perturbation" in attribution_methods:
            self.attributors["perturbation"] = PerturbationBasedAttributor(self.generator)
    
    def process_query(
        self, 
        query: str, 
        top_k: int = 5,
        store_attribution: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG system.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            store_attribution: Whether to store attribution scores in chunks
            
        Returns:
            Dictionary with query results and attribution info
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        # Generate answer
        answer, generation_info = self.generator.generate(query, retrieved_chunks)
        generation_time = time.time() - start_time - retrieval_time
        
        # Compute attribution scores
        attribution_scores = {}
        for method_name, attributor in self.attributors.items():
            attribution_scores[method_name] = attributor.attribute(
                query, retrieved_chunks, answer, generation_info
            )
        
        attribution_time = time.time() - start_time - retrieval_time - generation_time
        
        # Store attribution scores in chunks if requested
        if store_attribution:
            for method_name, scores in attribution_scores.items():
                for chunk, score in scores.items():
                    chunk.attribution_scores.append({
                        "query": query,
                        "method": method_name,
                        "score": score,
                        "timestamp": time.time()
                    })
        
        # Return results
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "attribution_scores": attribution_scores,
            "timings": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "attribution": attribution_time,
                "total": time.time() - start_time
            }
        }
    
    def evaluate_on_qa_pairs(
        self, 
        qa_pairs: List[Dict[str, str]], 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a set of QA pairs.
        
        Args:
            qa_pairs: List of {"question": str, "answer": str} dictionaries
            top_k: Number of chunks to retrieve per question
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = []
        
        # ROUGE scorer for evaluation
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Process each question
        total_retrieval_time = 0
        total_generation_time = 0
        total_attribution_time = 0
        
        for qa_pair in tqdm(qa_pairs, desc="Evaluating on QA pairs"):
            question = qa_pair["question"]
            reference_answer = qa_pair["answer"]
            
            # Process the question
            result = self.process_query(question, top_k=top_k)
            
            # Compute ROUGE scores
            generated_answer = result["answer"]
            rouge_scores = scorer.score(reference_answer, generated_answer)
            
            # Add to results
            result["reference_answer"] = reference_answer
            result["rouge_scores"] = {
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure
            }
            
            results.append(result)
            
            # Update timing stats
            total_retrieval_time += result["timings"]["retrieval"]
            total_generation_time += result["timings"]["generation"]
            total_attribution_time += result["timings"]["attribution"]
        
        # Compute average scores
        avg_rouge1 = np.mean([r["rouge_scores"]["rouge1"] for r in results])
        avg_rouge2 = np.mean([r["rouge_scores"]["rouge2"] for r in results])
        avg_rougeL = np.mean([r["rouge_scores"]["rougeL"] for r in results])
        
        # Return evaluation metrics
        return {
            "results": results,
            "metrics": {
                "avg_rouge1": avg_rouge1,
                "avg_rouge2": avg_rouge2,
                "avg_rougeL": avg_rougeL
            },
            "avg_timings": {
                "retrieval": total_retrieval_time / len(qa_pairs),
                "generation": total_generation_time / len(qa_pairs),
                "attribution": total_attribution_time / len(qa_pairs),
                "total": (total_retrieval_time + total_generation_time + total_attribution_time) / len(qa_pairs)
            }
        }

if __name__ == "__main__":
    # Sample code to test the RAG system
    from utils.data_utils import create_synthetic_data
    
    # Create synthetic data for testing
    data_chunks, qa_pairs = create_synthetic_data(num_chunks=100, num_qa_pairs=10)
    
    # Initialize RAG system
    rag_system = RAGSystem(
        data_chunks=data_chunks,
        retriever_type="bm25",
        attribution_methods=["attention"]
    )
    
    # Test on a single query
    result = rag_system.process_query("What was discovered in the Amazon rainforest?")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
    
    # Print attribution scores
    for method, scores in result["attribution_scores"].items():
        print(f"\nAttribution method: {method}")
        for chunk, score in scores.items():
            print(f"Chunk: {chunk.chunk_id}, Score: {score:.4f}")
    
    # Evaluate on QA pairs
    eval_results = rag_system.evaluate_on_qa_pairs(qa_pairs[:5])
    print(f"\nEvaluation results:")
    print(f"Average ROUGE-1: {eval_results['metrics']['avg_rouge1']:.4f}")
    print(f"Average ROUGE-L: {eval_results['metrics']['avg_rougeL']:.4f}")
    print(f"Average retrieval time: {eval_results['avg_timings']['retrieval']:.4f}s")
    print(f"Average generation time: {eval_results['avg_timings']['generation']:.4f}s")
    print(f"Average attribution time: {eval_results['avg_timings']['attribution']:.4f}s")