"""
Factual Consistency Checker Module for TrustPath.

This module implements the factual consistency component of TrustPath, which verifies
claims made in the LLM output against trusted knowledge sources.
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

import anthropic
from anthropic import Anthropic
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import FACTUAL_CHECKER_CONFIG, LLM_CONFIG
from fix_anthropic import fix_anthropic_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FactualConsistencyChecker:
    """
    Factual consistency checker that verifies claims against trusted knowledge sources.
    
    As described in the TrustPath proposal, this module implements Algorithm 2: Factual Consistency Checking.
    It extracts factual claims from LLM responses, retrieves relevant documents from trusted sources,
    and calculates a verification score based on the support for each claim.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the factual consistency checker.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
        """
        # Initialize the Anthropic client for claim extraction
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        
        # Initialize verification parameters
        self.verification_threshold = FACTUAL_CHECKER_CONFIG["verification_threshold"]
        self.max_documents = FACTUAL_CHECKER_CONFIG["max_documents"]
        self.knowledge_sources = FACTUAL_CHECKER_CONFIG["knowledge_sources"]
        
        # Initialize sentence embedding model for semantic similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"Initialized FactualConsistencyChecker with model: {self.model}")
        logger.info(f"Using knowledge sources: {', '.join(self.knowledge_sources)}")
    
    async def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from the LLM response.
        
        Args:
            response: The LLM response to analyze
            
        Returns:
            A list of extracted factual claims
        """
        logger.info("Extracting factual claims from response...")
        
        # Prompt to extract factual claims
        extraction_prompt = f"""
        Extract all factual claims from the following text. A factual claim is an assertion about the world that can be verified as true or false. Focus on statements that involve facts, dates, numbers, or specific assertions.
        
        Text: {response}
        
        For each factual claim, extract it as a simple, self-contained statement. Return ONLY the list of claims, one per line, without any additional commentary.
        """
        
        try:
            # Get extraction response from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more deterministic extraction
                system="You are an expert at identifying factual claims in text. Be thorough and precise in extracting verifiable assertions.",
                messages=[
                    {"role": "user", "content": extraction_prompt}
                ]
            )
            extraction_response = await fix_anthropic_response(message)
            
            # Split the response into individual claims
            claims = [claim.strip() for claim in extraction_response.split('\n') if claim.strip()]
            
            # Filter out non-claims (e.g., if the model adds commentary)
            claims = [claim for claim in claims if len(claim) > 10 and not claim.startswith("Claim:")]
            
            logger.info(f"Extracted {len(claims)} claims")
            return claims
            
        except Exception as e:
            logger.error(f"Error during claim extraction: {e}")
            # Fallback: use NLTK to split into sentences
            sentences = nltk.sent_tokenize(response)
            # Heuristic: a claim is likely a declarative sentence that's not too short
            claims = [s for s in sentences if len(s) > 20 and s[-1] == '.']
            logger.info(f"Fallback extracted {len(claims)} claims")
            return claims
    
    async def retrieve_documents(self, claim: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a claim from trusted knowledge sources.
        
        In a full implementation, this would connect to actual knowledge bases.
        For this prototype, we simulate document retrieval using the LLM itself.
        
        Args:
            claim: The factual claim to verify
            
        Returns:
            A list of relevant documents, each with content and metadata
        """
        logger.info(f"Retrieving documents for claim: {claim[:50]}...")
        
        # Construct a search query based on the claim
        search_query = self._formulate_search_query(claim)
        
        # Prompt to simulate document retrieval
        retrieval_prompt = f"""
        Simulate retrieving {self.max_documents} documents from trusted knowledge sources ({', '.join(self.knowledge_sources)}) relevant to this claim:
        
        Claim: "{claim}"
        Search query: "{search_query}"
        
        For each document, provide:
        1. Title
        2. Source (e.g., Wikipedia, academic paper, etc.)
        3. Content (a paragraph or two of factual information related to the claim)
        4. Relevance score (0-10)
        
        Format each document as a JSON object. Return a JSON array containing these document objects. Include ONLY the JSON array, no additional text.
        """
        
        try:
            # Get retrieval response from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens * 2,  # More tokens to allow for document content
                temperature=0.5,  # Some variation in documents
                system="You are a search engine with access to factual information from trusted sources. Provide accurate information relevant to the query.",
                messages=[
                    {"role": "user", "content": retrieval_prompt}
                ]
            )
            retrieval_response = await fix_anthropic_response(message)
            
            # Extract the JSON array from the response
            # Sometimes the model includes extra text, so we need to find the JSON array
            json_match = re.search(r'\[[\s\S]*\]', retrieval_response)
            if json_match:
                json_str = json_match.group(0)
                documents = json.loads(json_str)
            else:
                # Fallback if we can't extract a valid JSON array
                logger.warning("Could not extract JSON from retrieval response. Creating fallback documents.")
                documents = [
                    {
                        "title": "Fallback Document",
                        "source": "Error in document retrieval",
                        "content": retrieval_response[:500],
                        "relevance": 5
                    }
                ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []
    
    def _formulate_search_query(self, claim: str) -> str:
        """
        Transform a claim into an effective search query.
        
        Args:
            claim: The factual claim
            
        Returns:
            A search query based on the claim
        """
        # Remove question marks and common filler words
        query = claim.replace("?", "").replace(".", "")
        filler_words = ["a", "an", "the", "is", "are", "was", "were", "to", "in", "on", "at", "by", "for", "with", "about"]
        
        # Split into words, filter out filler words, and rejoin
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in filler_words]
        
        # Prioritize proper nouns (approximated by capitalized words)
        important_words = [word for word in filtered_words if word[0].isupper()] + [word for word in filtered_words if word[0].islower()]
        
        # Take the top words to form a concise query
        top_words = important_words[:8]
        return " ".join(top_words)
    
    async def calculate_verification_score(self, claim: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate a verification score for a claim based on retrieved documents.
        
        Args:
            claim: The factual claim
            documents: Retrieved documents relevant to the claim
            
        Returns:
            A dictionary with the verification score and supporting/contradicting evidence
        """
        if not documents:
            return {
                "verification_score": 0.0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
            }
        
        # Calculate relevance scores based on cosine similarity
        claim_embedding = self.embedding_model.encode([claim])[0]
        relevance_scores = []
        
        for doc in documents:
            doc_content = doc.get("content", "")
            doc_embedding = self.embedding_model.encode([doc_content])[0]
            similarity = cosine_similarity([claim_embedding], [doc_embedding])[0][0]
            # Scale to 0-1 range
            relevance_scores.append(max(0, min(1, similarity)))
        
        # Prompt to determine if each document supports or contradicts the claim
        support_prompt = f"""
        Analyze whether each document supports or contradicts this claim:
        
        Claim: "{claim}"
        
        Documents:
        {json.dumps(documents, indent=2)}
        
        For each document, provide:
        1. Document index (0-based)
        2. Support score (-1 to 1, where -1 means strong contradiction, 0 means neutral, 1 means strong support)
        3. Brief explanation of why it supports or contradicts the claim
        
        Format as a JSON array. Include ONLY the JSON array, no additional text.
        """
        
        try:
            # Get support analysis from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more consistent analysis
                system="You are an expert at analyzing whether evidence supports or contradicts claims. Be objective and thorough in your analysis.",
                messages=[
                    {"role": "user", "content": support_prompt}
                ]
            )
            support_response = await fix_anthropic_response(message)
            
            # Extract the JSON array from the response
            json_match = re.search(r'\[[\s\S]*\]', support_response)
            if json_match:
                json_str = json_match.group(0)
                support_analyses = json.loads(json_str)
            else:
                # Fallback if we can't extract a valid JSON array
                logger.warning("Could not extract JSON from support analysis. Creating fallback analysis.")
                support_analyses = [
                    {
                        "index": i,
                        "support_score": 0.0,
                        "explanation": "Could not analyze support."
                    } for i in range(len(documents))
                ]
            
            # Calculate the verification score as a weighted sum of relevance and support
            verification_score = 0.0
            supporting_evidence = []
            contradicting_evidence = []
            
            for analysis in support_analyses:
                doc_index = analysis.get("index", 0)
                if doc_index < len(documents):
                    doc = documents[doc_index]
                    relevance = relevance_scores[doc_index]
                    support = analysis.get("support_score", 0.0)
                    
                    # Contribute to the verification score
                    verification_score += relevance * support
                    
                    # Collect evidence
                    evidence_item = {
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                        "relevance": relevance,
                        "support": support,
                        "explanation": analysis.get("explanation", "")
                    }
                    
                    if support > 0:
                        supporting_evidence.append(evidence_item)
                    elif support < 0:
                        contradicting_evidence.append(evidence_item)
            
            # Normalize the verification score
            if support_analyses:
                verification_score /= len(support_analyses)
            
            # Transform to 0-1 range (0 = likely false, 1 = likely true)
            verification_score = (verification_score + 1) / 2
            
            return {
                "verification_score": verification_score,
                "supporting_evidence": supporting_evidence,
                "contradicting_evidence": contradicting_evidence,
            }
            
        except Exception as e:
            logger.error(f"Error during verification score calculation: {e}")
            return {
                "verification_score": 0.5,  # Neutral score on error
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "error": str(e)
            }
    
    async def formulate_correction(self, claim: str, verification_result: Dict[str, Any]) -> str:
        """
        Formulate a correction suggestion based on the verification result.
        
        Args:
            claim: The original claim
            verification_result: The verification result with evidence
            
        Returns:
            A suggested correction
        """
        if verification_result.get("verification_score", 0.0) >= self.verification_threshold:
            # Claim is likely true, no correction needed
            return ""
        
        # Construct a prompt with the claim and evidence
        supporting_evidence = verification_result.get("supporting_evidence", [])
        contradicting_evidence = verification_result.get("contradicting_evidence", [])
        
        correction_prompt = f"""
        Formulate a correction for this potentially erroneous claim:
        
        Original claim: "{claim}"
        
        Supporting evidence:
        {json.dumps(supporting_evidence, indent=2)}
        
        Contradicting evidence:
        {json.dumps(contradicting_evidence, indent=2)}
        
        Provide a corrected version of the claim that is accurate based on the evidence.
        Return ONLY the corrected claim, no additional text.
        """
        
        try:
            # Get correction from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more accurate correction
                system="You are an expert at correcting factual errors based on evidence. Be accurate and precise in your corrections.",
                messages=[
                    {"role": "user", "content": correction_prompt}
                ]
            )
            correction_response = await fix_anthropic_response(message)
            correction = correction_response.strip()
            return correction
            
        except Exception as e:
            logger.error(f"Error during correction formulation: {e}")
            return ""
    
    async def check_claim(self, claim: str) -> Dict[str, Any]:
        """
        Check a single factual claim and formulate a correction if needed.
        
        Args:
            claim: The factual claim to check
            
        Returns:
            A dictionary with the verification result and correction suggestion
        """
        logger.info(f"Checking claim: {claim[:50]}...")
        
        # Retrieve relevant documents
        documents = await self.retrieve_documents(claim)
        
        # Calculate verification score
        verification_result = await self.calculate_verification_score(claim, documents)
        
        # Formulate correction if needed
        correction = ""
        if verification_result.get("verification_score", 0.0) < self.verification_threshold:
            correction = await self.formulate_correction(claim, verification_result)
        
        return {
            "claim": claim,
            "verification_score": verification_result.get("verification_score", 0.0),
            "supporting_evidence": verification_result.get("supporting_evidence", []),
            "contradicting_evidence": verification_result.get("contradicting_evidence", []),
            "correction_suggestion": correction,
            "is_erroneous": verification_result.get("verification_score", 0.0) < self.verification_threshold
        }
    
    async def check_response(self, response: str) -> Dict[str, Any]:
        """
        Check an entire LLM response for factual consistency.
        
        Args:
            response: The LLM response to check
            
        Returns:
            A dictionary with the verification results for all claims
        """
        logger.info(f"Checking response for factual consistency...")
        
        # Extract factual claims
        claims = await self.extract_claims(response)
        
        # Check each claim
        claim_results = []
        for claim in claims:
            result = await self.check_claim(claim)
            claim_results.append(result)
        
        # Calculate overall verification score
        if claim_results:
            overall_score = sum(r.get("verification_score", 0.0) for r in claim_results) / len(claim_results)
        else:
            overall_score = 1.0  # No claims means no errors
        
        # Identify erroneous claims
        erroneous_claims = [r for r in claim_results if r.get("is_erroneous", False)]
        
        return {
            "overall_verification_score": overall_score,
            "claim_results": claim_results,
            "erroneous_claims": erroneous_claims,
            "total_claims": len(claims),
            "total_erroneous_claims": len(erroneous_claims)
        }

# Synchronous version of check_response for easier testing
def check_response(response: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around FactualConsistencyChecker.check_response.
    
    Args:
        response: The LLM response to check
        api_key: The API key for the LLM service
        
    Returns:
        A dictionary with the verification results
    """
    import asyncio
    
    checker = FactualConsistencyChecker(api_key=api_key)
    return asyncio.run(checker.check_response(response))

if __name__ == "__main__":
    # Simple test of the factual consistency checker
    test_response = """
    The Eiffel Tower was built in 1878 and is located in Lyon, France. It was designed by Gustave Eiffel and is made entirely of copper. The tower is 124 meters tall and weighs approximately 7,300 tons. It has become one of the most recognizable landmarks in the world.
    """
    
    print("Testing factual consistency checker...")
    check_results = check_response(test_response)
    print(json.dumps(check_results, indent=2))