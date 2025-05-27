"""
Baseline Methods for Comparison with TrustPath.

This module implements baseline methods for error detection and correction:
1. Simple fact-checking
2. Uncertainty estimation
3. Standard post-hoc correction
"""

import json
import logging
import re
from typing import Dict, List, Tuple, Any, Optional

import anthropic
from anthropic import Anthropic
import nltk
from sentence_transformers import SentenceTransformer

from config import BASELINE_CONFIG, LLM_CONFIG
from fix_anthropic import fix_anthropic_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SimpleFactChecker:
    """
    Simple fact-checking baseline that checks facts against a knowledge source.
    
    This baseline directly compares claims in the LLM response against trusted sources
    without the transparency features of TrustPath.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the simple fact checker.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
        """
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        
        logger.info(f"Initialized SimpleFactChecker with model: {self.model}")
    
    async def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from the response.
        
        Args:
            response: The LLM response to analyze
            
        Returns:
            A list of extracted factual claims
        """
        # Use NLTK to split into sentences as a simple claim extraction method
        sentences = nltk.sent_tokenize(response)
        
        # Filter out questions and short sentences
        claims = [s for s in sentences if len(s) > 20 and s[-1] == '.']
        
        logger.info(f"Extracted {len(claims)} claims using simple method")
        return claims
    
    async def check_response(self, response: str) -> Dict[str, Any]:
        """
        Check the LLM response for factual errors.
        
        Args:
            response: The LLM response to check
            
        Returns:
            A dictionary with the checking results
        """
        logger.info(f"Checking response with simple fact checker...")
        
        # Extract claims
        claims = await self.extract_claims(response)
        
        # Prepare the fact checking prompt
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        
        checking_prompt = f"""
        Fact-check the following statements. For each statement, indicate whether it is TRUE or FALSE based on factual knowledge, and provide a brief explanation.
        
        Statements:
        {claims_text}
        
        Return a JSON array with this format: 
        [
          {{
            "claim": "The statement to check",
            "is_true": true/false,
            "explanation": "Brief explanation of the verdict"
          }}
        ]
        
        Return ONLY the JSON array, no additional text.
        """
        
        try:
            # Get fact checking response from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more consistent fact checking
                system="You are a fact-checking expert with access to accurate information about the world. Verify each statement carefully based on factual knowledge.",
                messages=[
                    {"role": "user", "content": checking_prompt}
                ]
            )
            checking_response = await fix_anthropic_response(message)
            
            # Extract the JSON array from the response
            json_match = re.search(r'\[[\s\S]*\]', checking_response)
            if json_match:
                json_str = json_match.group(0)
                checking_results = json.loads(json_str)
            else:
                logger.warning("Could not extract JSON from checking response. Creating fallback results.")
                checking_results = [
                    {
                        "claim": claim,
                        "is_true": True,  # Default to true
                        "explanation": "Unable to verify."
                    } for claim in claims
                ]
            
            # Extract erroneous claims
            erroneous_claims = [
                {
                    "content": result.get("claim", ""),
                    "explanation": result.get("explanation", ""),
                    "source": "simple_fact_checking"
                }
                for result in checking_results if not result.get("is_true", True)
            ]
            
            return {
                "total_claims": len(claims),
                "checked_claims": checking_results,
                "erroneous_claims": erroneous_claims,
                "total_erroneous_claims": len(erroneous_claims)
            }
            
        except Exception as e:
            logger.error(f"Error during simple fact checking: {e}")
            return {
                "total_claims": len(claims),
                "checked_claims": [],
                "erroneous_claims": [],
                "total_erroneous_claims": 0,
                "error": str(e)
            }

class UncertaintyEstimator:
    """
    Uncertainty estimation baseline that estimates the LLM's uncertainty.
    
    This baseline focuses on detecting uncertainty in the LLM's response
    without the transparency and explanation features of TrustPath.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the uncertainty estimator.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
        """
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        
        # List of uncertainty markers (words and phrases indicating uncertainty)
        self.uncertainty_markers = [
            "probably", "likely", "might", "may", "could", "perhaps", 
            "possibly", "potentially", "uncertain", "not sure", "estimate",
            "approximately", "about", "around", "roughly", "I think",
            "seems", "appears", "believed to", "reportedly"
        ]
        
        logger.info(f"Initialized UncertaintyEstimator with model: {self.model}")
    
    async def estimate_uncertainty(self, response: str) -> Dict[str, Any]:
        """
        Estimate uncertainty in the LLM response.
        
        Args:
            response: The LLM response to analyze
            
        Returns:
            A dictionary with the uncertainty estimation results
        """
        logger.info(f"Estimating uncertainty in response...")
        
        # Split the response into sentences
        sentences = nltk.sent_tokenize(response)
        
        uncertain_statements = []
        
        for sentence in sentences:
            # Check for uncertainty markers
            has_marker = any(marker.lower() in sentence.lower() for marker in self.uncertainty_markers)
            
            if has_marker:
                uncertain_statements.append({
                    "content": sentence,
                    "markers": [marker for marker in self.uncertainty_markers if marker.lower() in sentence.lower()],
                    "source": "uncertainty_estimation"
                })
        
        # Prepare the uncertainty estimation prompt
        uncertainty_prompt = f"""
        Rate the certainty level of the following statements from 0 (completely uncertain) to 10 (completely certain).

        Statements:
        {response}
        
        For each statement with a certainty level below 7, provide:
        1. The statement
        2. The certainty score (0-10)
        3. The reason for the uncertainty
        
        Return a JSON array with this format:
        [
          {{
            "statement": "The uncertain statement",
            "certainty_score": 5,
            "reason": "Reason for the uncertainty"
          }}
        ]
        
        Return ONLY the JSON array, no additional text.
        """
        
        try:
            # Get uncertainty estimation from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more consistent estimation
                system="You are an expert at evaluating the certainty level of statements. Identify statements that express uncertainty, hedging, or lack of confidence.",
                messages=[
                    {"role": "user", "content": uncertainty_prompt}
                ]
            )
            uncertainty_response = await fix_anthropic_response(message)
            
            # Extract the JSON array from the response
            json_match = re.search(r'\[[\s\S]*\]', uncertainty_response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    llm_uncertainty = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON from uncertainty response.")
                    llm_uncertainty = []
            else:
                logger.warning("Could not extract JSON from uncertainty response.")
                llm_uncertainty = []
            
            # Combine both methods
            for uncertain in llm_uncertainty:
                statement = uncertain.get("statement", "")
                
                # Check if this statement is already in the list
                if not any(statement in us["content"] for us in uncertain_statements):
                    uncertain_statements.append({
                        "content": statement,
                        "certainty_score": uncertain.get("certainty_score", 5),
                        "reason": uncertain.get("reason", "Low certainty detected by LLM"),
                        "source": "uncertainty_estimation_llm"
                    })
            
            return {
                "uncertain_statements": uncertain_statements,
                "total_uncertain_statements": len(uncertain_statements)
            }
            
        except Exception as e:
            logger.error(f"Error during uncertainty estimation: {e}")
            return {
                "uncertain_statements": uncertain_statements,
                "total_uncertain_statements": len(uncertain_statements),
                "error": str(e)
            }

class StandardCorrector:
    """
    Standard post-hoc correction baseline that corrects errors without explanations.
    
    This baseline focuses on correcting detected errors without the
    transparency and explanation features of TrustPath.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the standard corrector.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
        """
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        
        # Initialize fact checker for finding errors
        self.fact_checker = SimpleFactChecker(api_key=api_key)
        
        logger.info(f"Initialized StandardCorrector with model: {self.model}")
    
    async def correct_response(self, question: str, response: str) -> Dict[str, Any]:
        """
        Correct errors in the LLM response without explanations.
        
        Args:
            question: The original question
            response: The LLM response to correct
            
        Returns:
            A dictionary with the correction results
        """
        logger.info(f"Correcting response with standard corrector...")
        
        # Find errors using the fact checker
        fact_check_results = await self.fact_checker.check_response(response)
        erroneous_claims = fact_check_results.get("erroneous_claims", [])
        
        if not erroneous_claims:
            return {
                "original_response": response,
                "corrected_response": response,
                "has_corrections": False,
                "num_corrections": 0
            }
        
        # Prepare the correction prompt
        errors_text = "\n".join([f"- {err['content']}" for err in erroneous_claims])
        
        correction_prompt = f"""
        The following response contains factual errors:
        
        Original response to the question "{question}":
        {response}
        
        Errors identified:
        {errors_text}
        
        Provide a corrected version of the entire response that fixes these errors.
        Maintain the same style and format as the original response, but with accurate information.
        """
        
        try:
            # Get corrected response from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more accurate correction
                system="You are an expert at correcting factual errors in text. Provide accurate corrections while maintaining the original style and format.",
                messages=[
                    {"role": "user", "content": correction_prompt}
                ]
            )
            corrected_response = await fix_anthropic_response(message)
            
            return {
                "original_response": response,
                "corrected_response": corrected_response,
                "has_corrections": True,
                "num_corrections": len(erroneous_claims)
            }
            
        except Exception as e:
            logger.error(f"Error during standard correction: {e}")
            return {
                "original_response": response,
                "corrected_response": response,
                "has_corrections": False,
                "num_corrections": 0,
                "error": str(e)
            }

# Synchronous wrappers for easier testing
def simple_fact_check(response: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around SimpleFactChecker.check_response.
    
    Args:
        response: The LLM response to check
        api_key: The API key for the LLM service
        
    Returns:
        The checking results
    """
    import asyncio
    
    checker = SimpleFactChecker(api_key=api_key)
    return asyncio.run(checker.check_response(response))

def estimate_uncertainty(response: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around UncertaintyEstimator.estimate_uncertainty.
    
    Args:
        response: The LLM response to analyze
        api_key: The API key for the LLM service
        
    Returns:
        The uncertainty estimation results
    """
    import asyncio
    
    estimator = UncertaintyEstimator(api_key=api_key)
    return asyncio.run(estimator.estimate_uncertainty(response))

def standard_correction(question: str, response: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around StandardCorrector.correct_response.
    
    Args:
        question: The original question
        response: The LLM response to correct
        api_key: The API key for the LLM service
        
    Returns:
        The correction results
    """
    import asyncio
    
    corrector = StandardCorrector(api_key=api_key)
    return asyncio.run(corrector.correct_response(question, response))

if __name__ == "__main__":
    # Simple test of the baseline methods
    test_response = """
    The Eiffel Tower was built in 1878 and is located in Lyon, France. It was designed by Gustave Eiffel and is made entirely of copper. The tower is 124 meters tall and weighs approximately 7,300 tons. It has become one of the most recognizable landmarks in the world.
    """
    
    test_question = "When was the Eiffel Tower built and where is it located?"
    
    print("Testing simple fact checker...")
    fact_check_results = simple_fact_check(test_response)
    print(json.dumps(fact_check_results, indent=2))
    
    print("\nTesting uncertainty estimator...")
    uncertainty_results = estimate_uncertainty(test_response)
    print(json.dumps(uncertainty_results, indent=2))
    
    print("\nTesting standard corrector...")
    correction_results = standard_correction(test_question, test_response)
    print(json.dumps(correction_results, indent=2))