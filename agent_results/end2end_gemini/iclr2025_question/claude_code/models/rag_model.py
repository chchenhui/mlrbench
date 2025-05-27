"""
Implementation of the standard RAG (Retrieval-Augmented Generation) model.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .base_model import BaseModel, APIBasedModel

logger = logging.getLogger(__name__)

class RetrieverModule:
    """Retriever module for fetching relevant documents from a knowledge base."""
    
    def __init__(
        self, 
        knowledge_base_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_sparse: bool = False,
        device: str = None
    ):
        """
        Initialize the retriever module.
        
        Args:
            knowledge_base_path: Path to the knowledge base JSON file.
            embedding_model_name: Name of the embedding model to use.
            use_sparse: Whether to use sparse retrieval (BM25) instead of dense.
            device: Device to use for computation.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sparse = use_sparse
        self.load_knowledge_base(knowledge_base_path)
        
        if not use_sparse:
            self.initialize_dense_retriever(embedding_model_name)
        else:
            self.initialize_sparse_retriever()
        
        logger.info(f"Initialized RetrieverModule with {'sparse' if use_sparse else 'dense'} retrieval")
    
    def load_knowledge_base(self, path: str):
        """
        Load the knowledge base from a JSON file.
        
        Args:
            path: Path to the knowledge base JSON file.
        """
        try:
            with open(path, 'r') as f:
                self.knowledge_base = json.load(f)
            
            self.documents = [item["text"] for item in self.knowledge_base]
            logger.info(f"Loaded knowledge base with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            # Create a minimal dummy knowledge base
            self.knowledge_base = [
                {"id": 0, "text": "The Earth is approximately spherical, not flat."},
                {"id": 1, "text": "Paris is the capital of France."},
                {"id": 2, "text": "Neil Armstrong was the first person to walk on the Moon in 1969."}
            ]
            self.documents = [item["text"] for item in self.knowledge_base]
            logger.warning("Using dummy knowledge base instead")
    
    def initialize_dense_retriever(self, model_name: str):
        """
        Initialize the dense retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model.
        """
        try:
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            
            # Create FAISS index
            self.document_embeddings = self.embedding_model.encode(self.documents)
            dimension = self.document_embeddings.shape[1]
            
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.document_embeddings)
            
            logger.info(f"Initialized dense retriever with {model_name}")
        except Exception as e:
            logger.error(f"Error initializing dense retriever: {e}")
            raise
    
    def initialize_sparse_retriever(self):
        """Initialize a basic sparse retriever (BM25-like)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            
            logger.info("Initialized sparse retriever with TF-IDF")
        except Exception as e:
            logger.error(f"Error initializing sparse retriever: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
        
        Returns:
            List of retrieved documents.
        """
        if self.use_sparse:
            return self._sparse_retrieve(query, top_k)
        else:
            return self._dense_retrieve(query, top_k)
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[str]:
        """
        Retrieve documents using dense embedding similarity.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
        
        Returns:
            List of retrieved documents.
        """
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        return retrieved_docs
    
    def _sparse_retrieve(self, query: str, top_k: int) -> List[str]:
        """
        Retrieve documents using sparse TF-IDF similarity.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
        
        Returns:
            List of retrieved documents.
        """
        query_vector = self.vectorizer.transform([query])
        scores = (self.document_vectors * query_vector.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved_docs = [self.documents[idx] for idx in top_indices]
        return retrieved_docs


class StandardRAGModel:
    """Standard RAG model that always retrieves documents for each input."""
    
    def __init__(
        self,
        base_model: Union[BaseModel, APIBasedModel],
        retriever: RetrieverModule,
        num_documents: int = 3,
        prompt_template: str = None
    ):
        """
        Initialize the standard RAG model.
        
        Args:
            base_model: The base LLM model.
            retriever: The retriever module.
            num_documents: Number of documents to retrieve.
            prompt_template: Template for constructing the prompt with retrieved docs.
        """
        self.base_model = base_model
        self.retriever = retriever
        self.num_documents = num_documents
        
        # Default prompt template if none provided
        self.prompt_template = prompt_template or (
            "Answer the following question based on the information provided.\n\n"
            "Context information:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        logger.info(f"Initialized StandardRAGModel with {num_documents} documents per query")
    
    def format_prompt_with_context(self, question: str, context_docs: List[str]) -> str:
        """
        Format the prompt with the question and retrieved context.
        
        Args:
            question: The input question.
            context_docs: List of retrieved context documents.
        
        Returns:
            The formatted prompt.
        """
        context_str = "\n".join([f"- {doc}" for doc in context_docs])
        return self.prompt_template.format(question=question, context=context_str)
    
    def generate(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """
        Generate answers with retrieval augmentation.
        
        Args:
            inputs: The input question(s).
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional generation parameters.
        
        Returns:
            The generated answers.
        """
        # Handle single input
        if isinstance(inputs, str):
            inputs = [inputs]
        
        augmented_prompts = []
        
        # Retrieve documents for each input and create augmented prompts
        for input_text in inputs:
            retrieved_docs = self.retriever.retrieve(input_text, self.num_documents)
            augmented_prompt = self.format_prompt_with_context(input_text, retrieved_docs)
            augmented_prompts.append(augmented_prompt)
        
        # Generate answers using the base model
        outputs = self.base_model.generate(
            augmented_prompts,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return outputs