"""
Evidence retrieval and alignment module for SCEC.
"""

import os
import json
import logging
import re
import string
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import spacy
import faiss
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DPRQuestionEncoder,
    DPRContextEncoder,
    PreTrainedModel,
    pipeline,
)
from sentence_transformers import SentenceTransformer
import wikipediaapi

logger = logging.getLogger(__name__)

# Load spaCy for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Failed to load spaCy model. Using simple regex for claim extraction.")
    nlp = None


class ClaimExtractor:
    """Extract factual claims from text."""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the claim extractor.
        
        Args:
            use_spacy: Whether to use spaCy for NER and sentence parsing
        """
        self.use_spacy = use_spacy and nlp is not None
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of extracted claim strings
        """
        if self.use_spacy:
            return self._extract_claims_spacy(text)
        else:
            return self._extract_claims_simple(text)
    
    def _extract_claims_spacy(self, text: str) -> List[str]:
        """Use spaCy to extract claims with NER and sentence parsing."""
        doc = nlp(text)
        claims = []
        
        # Consider each sentence as a potential claim
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            
            # Only consider sentences containing entities or numbers
            has_entity = any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL"] 
                            for ent in sentence.ents)
            has_number = any(token.like_num for token in sentence)
            
            if has_entity or has_number:
                claims.append(sentence_text)
        
        return claims
    
    def _extract_claims_simple(self, text: str) -> List[str]:
        """Use simple regex and rules to extract claims."""
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for sentence patterns that often contain factual claims
            has_entity = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence))
            has_number = bool(re.search(r'\b\d+(?:\.\d+)?%?\b', sentence))
            has_date = bool(re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b', sentence, re.IGNORECASE))
            
            # Check for "factual" claim indicators
            claim_indicators = ["is", "was", "are", "were", "has", "have", "had", 
                              "discovered", "invented", "founded", "created", 
                              "born", "died", "established", "developed", "built"]
            has_indicator = any(f" {indicator} " in f" {sentence.lower()} " for indicator in claim_indicators)
            
            if has_entity or has_number or has_date or has_indicator:
                claims.append(sentence)
        
        return claims


class EvidenceRetriever:
    """Base class for evidence retrieval systems."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the evidence retriever.
        
        Args:
            cache_dir: Directory to cache retrieved evidence
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Set up cache
        self.cache = {}
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve evidence for a query.
        
        Args:
            query: Query to retrieve evidence for
            k: Number of evidence passages to retrieve
            
        Returns:
            List of dictionaries containing evidence passages
        """
        raise NotImplementedError("Subclasses must implement retrieve")
    
    def _format_evidence(self, evidence_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format evidence passages consistently.
        
        Args:
            evidence_list: List of evidence dictionaries
            
        Returns:
            Formatted evidence list
        """
        formatted = []
        for i, evidence in enumerate(evidence_list):
            formatted.append({
                "id": evidence.get("id", f"evidence-{i}"),
                "text": evidence["text"],
                "source": evidence.get("source", "unknown"),
                "score": float(evidence.get("score", 0.0)),
            })
        return formatted
    
    def _get_from_cache(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Get cached results for a query if available."""
        if query in self.cache:
            return self.cache[query]
        
        if not self.cache_dir:
            return None
        
        # Try to load from disk cache
        cache_file = os.path.join(self.cache_dir, f"{hash(query):x}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        
        return None
    
    def _save_to_cache(self, query: str, results: List[Dict[str, str]]):
        """Save results to cache."""
        self.cache[query] = results
        
        if not self.cache_dir:
            return
        
        # Save to disk cache
        cache_file = os.path.join(self.cache_dir, f"{hash(query):x}.json")
        with open(cache_file, 'w') as f:
            json.dump(results, f)


class WikipediaRetriever(EvidenceRetriever):
    """Retrieve evidence from Wikipedia."""
    
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        language: str = "en",
        user_agent: str = "SCEC-Experiment/1.0",
    ):
        """
        Initialize the Wikipedia retriever.
        
        Args:
            cache_dir: Directory to cache retrieved evidence
            language: Wikipedia language code
            user_agent: User agent for Wikipedia API
        """
        super().__init__(cache_dir)
        self.language = language
        self.wikipedia = wikipediaapi.Wikipedia(
            language=language,
            user_agent=user_agent
        )
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve evidence from Wikipedia.
        
        Args:
            query: Query to retrieve evidence for
            k: Number of evidence passages to retrieve
            
        Returns:
            List of dictionaries containing evidence passages
        """
        # Check cache first
        cached = self._get_from_cache(query)
        if cached:
            return cached
        
        # Search for Wikipedia pages
        search_results = self.wikipedia.page(query)
        evidence_list = []
        
        if search_results.exists():
            # Extract the main page content
            content = search_results.summary
            if content:
                evidence_list.append({
                    "id": f"wiki-{search_results.pageid}",
                    "text": content,
                    "source": f"Wikipedia: {search_results.title}",
                    "score": 1.0
                })
            
            # Also add sections as separate evidence pieces
            for section_name, section in search_results.sections.items():
                if section.text:
                    evidence_list.append({
                        "id": f"wiki-{search_results.pageid}-{hash(section_name):x}",
                        "text": section.text[:1000],  # Limit length
                        "source": f"Wikipedia: {search_results.title} - {section_name}",
                        "score": 0.9  # Slightly lower than main content
                    })
        
        # If we don't have enough evidence, try disambiguation pages or related
        if len(evidence_list) < k:
            # Implementation would extend search to get more results...
            pass
        
        # Format and limit results
        formatted_evidence = self._format_evidence(evidence_list[:k])
        
        # Cache the results
        self._save_to_cache(query, formatted_evidence)
        
        return formatted_evidence


class BM25Retriever(EvidenceRetriever):
    """BM25-based evidence retriever using a local corpus."""
    
    def __init__(
        self,
        corpus_path: str,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the BM25 retriever.
        
        Args:
            corpus_path: Path to corpus file (JSON list of documents)
            cache_dir: Directory to cache retrieved evidence
        """
        super().__init__(cache_dir)
        self.corpus_path = corpus_path
        
        # Load corpus
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        # Preprocess corpus for BM25
        self.corpus_texts = [doc['text'] for doc in self.corpus]
        tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"Loaded corpus with {len(self.corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Remove punctuation and lowercase
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Split on whitespace and filter empty strings
        return [token for token in text.split() if token.strip()]
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve evidence using BM25.
        
        Args:
            query: Query to retrieve evidence for
            k: Number of evidence passages to retrieve
            
        Returns:
            List of dictionaries containing evidence passages
        """
        # Check cache first
        cached = self._get_from_cache(query)
        if cached:
            return cached
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k documents
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        
        evidence_list = []
        for i, idx in enumerate(top_k_indices):
            doc = self.corpus[idx]
            evidence_list.append({
                "id": doc.get("id", f"doc-{idx}"),
                "text": doc["text"],
                "source": doc.get("source", "corpus"),
                "score": float(doc_scores[idx]),
            })
        
        # Format results
        formatted_evidence = self._format_evidence(evidence_list)
        
        # Cache the results
        self._save_to_cache(query, formatted_evidence)
        
        return formatted_evidence


class DenseRetriever(EvidenceRetriever):
    """Dense retrieval using embedding models."""
    
    def __init__(
        self,
        corpus_path: str,
        model_name: str = "facebook/dpr-question_encoder-single-nq-base",
        context_model_name: Optional[str] = "facebook/dpr-ctx_encoder-single-nq-base",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize the dense retriever.
        
        Args:
            corpus_path: Path to corpus file (JSON list of documents)
            model_name: Query encoder model
            context_model_name: Context encoder model (if different from query encoder)
            cache_dir: Directory to cache retrieved evidence
            device: Device to run models on ('cuda', 'cpu', 'auto')
        """
        super().__init__(cache_dir)
        self.corpus_path = corpus_path
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load corpus
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        # Load query encoder
        logger.info(f"Loading query encoder {model_name}")
        if "dpr" in model_name.lower():
            self.query_encoder = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
            self.query_tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Fallback to sentence transformers for non-DPR models
            self.query_encoder = SentenceTransformer(model_name, device=self.device)
            self.query_tokenizer = None
        
        # Load context encoder (if different from query encoder)
        if context_model_name and context_model_name != model_name:
            logger.info(f"Loading context encoder {context_model_name}")
            if "dpr" in context_model_name.lower():
                self.context_encoder = DPRContextEncoder.from_pretrained(context_model_name).to(self.device)
                self.context_tokenizer = AutoTokenizer.from_pretrained(context_model_name)
            else:
                # Fallback to sentence transformers
                self.context_encoder = SentenceTransformer(context_model_name, device=self.device)
                self.context_tokenizer = None
        else:
            # Use same model for both query and context
            self.context_encoder = self.query_encoder
            self.context_tokenizer = self.query_tokenizer
        
        # Build FAISS index for corpus
        self._build_index()
    
    def _build_index(self):
        """Build a FAISS index for the corpus."""
        logger.info("Building FAISS index for corpus")
        
        # Get all document texts
        corpus_texts = [doc['text'] for doc in self.corpus]
        
        # Encode all documents
        if isinstance(self.context_encoder, SentenceTransformer):
            # Use SentenceTransformer
            corpus_embeddings = self.context_encoder.encode(
                corpus_texts, 
                show_progress_bar=True, 
                convert_to_tensor=True
            )
            corpus_embeddings = corpus_embeddings.cpu().numpy()
        else:
            # Use DPR
            corpus_embeddings = []
            batch_size = 32
            
            for i in tqdm(range(0, len(corpus_texts), batch_size)):
                batch = corpus_texts[i:i+batch_size]
                inputs = self.context_tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.context_encoder(**inputs)
                    embeddings = outputs.pooler_output.cpu().numpy()
                    corpus_embeddings.append(embeddings)
            
            corpus_embeddings = np.vstack(corpus_embeddings)
        
        # Normalize embeddings
        faiss.normalize_L2(corpus_embeddings)
        
        # Build FAISS index
        self.embedding_size = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(corpus_embeddings)
        
        logger.info(f"Built index with {len(corpus_texts)} documents")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query using the query encoder."""
        if isinstance(self.query_encoder, SentenceTransformer):
            # Use SentenceTransformer
            query_embedding = self.query_encoder.encode(
                query, 
                convert_to_tensor=True
            )
            query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
        else:
            # Use DPR
            inputs = self.query_tokenizer(
                query, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.query_encoder(**inputs)
                query_embedding = outputs.pooler_output.cpu().numpy()
        
        # Normalize embedding
        faiss.normalize_L2(query_embedding)
        
        return query_embedding
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Retrieve evidence using dense embeddings.
        
        Args:
            query: Query to retrieve evidence for
            k: Number of evidence passages to retrieve
            
        Returns:
            List of dictionaries containing evidence passages
        """
        # Check cache first
        cached = self._get_from_cache(query)
        if cached:
            return cached
        
        # Encode query
        query_embedding = self._encode_query(query)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Get top-k documents
        evidence_list = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= len(self.corpus):
                continue  # Skip invalid indices
                
            doc = self.corpus[idx]
            evidence_list.append({
                "id": doc.get("id", f"doc-{idx}"),
                "text": doc["text"],
                "source": doc.get("source", "corpus"),
                "score": float(score),
            })
        
        # Format results
        formatted_evidence = self._format_evidence(evidence_list)
        
        # Cache the results
        self._save_to_cache(query, formatted_evidence)
        
        return formatted_evidence


class EntailmentScorer:
    """Score the entailment between claims and evidence."""
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "auto",
    ):
        """
        Initialize the entailment scorer.
        
        Args:
            model_name: Entailment model name
            device: Device to run model on ('cuda', 'cpu', 'auto')
        """
        self.model_name = model_name
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Load model and tokenizer
        logger.info(f"Loading entailment model {model_name}")
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )
    
    def score(
        self,
        claim: str,
        evidence: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """
        Score the entailment between a claim and evidence.
        
        Args:
            claim: Claim text
            evidence: Evidence text or list of evidence texts
            
        Returns:
            Entailment score or list of scores
        """
        # Handle single evidence case
        if isinstance(evidence, str):
            return self._score_single(claim, evidence)
        
        # Handle multiple evidence case
        return [self._score_single(claim, e) for e in evidence]
    
    def _score_single(self, claim: str, evidence: str) -> float:
        """Score entailment for a single claim-evidence pair."""
        # Run zero-shot classification with "entailment" and "contradiction" labels
        result = self.pipeline(
            evidence,
            candidate_labels=["entailment", "contradiction"],
            hypothesis=claim,
        )
        
        # Get the score for entailment
        entailment_idx = result["labels"].index("entailment")
        entailment_score = result["scores"][entailment_idx]
        
        return entailment_score


class EvidenceAligner:
    """Align claims with retrieved evidence and compute agreement scores."""
    
    def __init__(
        self,
        retriever: EvidenceRetriever,
        claim_extractor: Optional[ClaimExtractor] = None,
        entailment_scorer: Optional[EntailmentScorer] = None,
    ):
        """
        Initialize the evidence aligner.
        
        Args:
            retriever: Evidence retriever instance
            claim_extractor: Claim extractor instance (creates new one if None)
            entailment_scorer: Entailment scorer instance (creates new one if None)
        """
        self.retriever = retriever
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.entailment_scorer = entailment_scorer or EntailmentScorer()
    
    def align(
        self, 
        text: str, 
        k_evidence: int = 5,
    ) -> Dict[str, Any]:
        """
        Extract claims from text, retrieve evidence, and compute alignment scores.
        
        Args:
            text: Input text to extract claims from and align with evidence
            k_evidence: Number of evidence passages to retrieve per claim
            
        Returns:
            Dictionary with alignment results
        """
        # Extract claims from text
        claims = self.claim_extractor.extract_claims(text)
        
        if not claims:
            # If no explicit claims found, use the whole text as a single claim
            claims = [text]
        
        # Process each claim
        claim_results = []
        for claim in claims:
            # Retrieve evidence for this claim
            evidence = self.retriever.retrieve(claim, k=k_evidence)
            
            # Score the entailment between claim and evidence
            evidence_texts = [e["text"] for e in evidence]
            entailment_scores = self.entailment_scorer.score(claim, evidence_texts)
            
            # Add scores to evidence
            for i, score in enumerate(entailment_scores):
                evidence[i]["entailment_score"] = float(score)
            
            # Sort evidence by entailment score
            evidence = sorted(evidence, key=lambda e: e["entailment_score"], reverse=True)
            
            # Compute claim agreement score (max entailment score)
            claim_agreement = max(entailment_scores) if entailment_scores else 0.0
            
            claim_results.append({
                "claim": claim,
                "evidence": evidence,
                "agreement_score": claim_agreement,
            })
        
        # Compute overall agreement score (average of claim scores)
        overall_agreement = np.mean([r["agreement_score"] for r in claim_results]) if claim_results else 0.0
        
        return {
            "text": text,
            "claims": claim_results,
            "overall_agreement": float(overall_agreement),
        }


def create_synthetic_corpus(
    output_path: str, 
    num_documents: int = 1000,
    seed: int = 42
) -> None:
    """
    Create a synthetic corpus for testing evidence retrieval.
    
    Args:
        output_path: Path to save the corpus
        num_documents: Number of documents to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Create synthetic topics and facts
    topics = [
        "Paris", "London", "New York", "Tokyo", "Sydney",
        "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Nikola Tesla",
        "COVID-19", "Cancer", "Diabetes", "Heart Disease", "Alzheimer's",
        "Python", "JavaScript", "Java", "C++", "Ruby",
        "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
    ]
    
    facts = {
        "Paris": [
            "Paris is the capital of France",
            "The Eiffel Tower in Paris was completed in 1889",
            "Paris has a population of approximately 2.2 million people",
            "The Louvre Museum in Paris is the world's largest art museum",
            "Paris hosted the Summer Olympics in 1900 and 1924",
        ],
        "London": [
            "London is the capital of England and the United Kingdom",
            "The London Underground was opened in 1863",
            "Buckingham Palace is the residence of the British monarch in London",
            "The River Thames flows through London",
            "The London Eye is a giant Ferris wheel on the South Bank of the River Thames",
        ],
        # ... other topics with facts
    }
    
    # Generate synthetic documents
    documents = []
    
    for i in range(num_documents):
        # Pick a random topic
        topic = np.random.choice(topics)
        
        # If we have facts for this topic, use them
        if topic in facts:
            # Include some facts about the topic
            topic_facts = np.random.choice(facts[topic], size=min(3, len(facts[topic])), replace=False)
            content = f"{topic}. " + " ".join(topic_facts)
        else:
            # Generate generic content
            content = f"{topic}. This is a synthetic document about {topic}."
        
        documents.append({
            "id": f"doc-{i}",
            "text": content,
            "source": f"Synthetic corpus - {topic}",
        })
    
    # Save corpus to file
    with open(output_path, 'w') as f:
        json.dump(documents, f, indent=2)
    
    logger.info(f"Created synthetic corpus with {len(documents)} documents at {output_path}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    # Create a synthetic corpus
    corpus_path = "data/synthetic_corpus.json"
    create_synthetic_corpus(corpus_path, num_documents=1000)
    
    # Initialize retriever
    retriever = BM25Retriever(corpus_path=corpus_path, cache_dir="cache")
    
    # Initialize claim extractor and entailment scorer
    claim_extractor = ClaimExtractor()
    entailment_scorer = EntailmentScorer()
    
    # Initialize evidence aligner
    aligner = EvidenceAligner(retriever, claim_extractor, entailment_scorer)
    
    # Example text to analyze
    text = "Paris is the capital of France. The Eiffel Tower is 330 meters tall."
    
    # Align text with evidence
    alignment_results = aligner.align(text, k_evidence=3)
    
    print(f"Overall agreement score: {alignment_results['overall_agreement']}")
    for i, claim_result in enumerate(alignment_results["claims"]):
        print(f"\nClaim {i+1}: {claim_result['claim']}")
        print(f"Agreement score: {claim_result['agreement_score']}")
        for j, evidence in enumerate(claim_result["evidence"]):
            print(f"  Evidence {j+1}: {evidence['text'][:100]}... (Score: {evidence['entailment_score']})")