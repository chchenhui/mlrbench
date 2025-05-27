"""
Self-Verification Module for TrustPath.

This module implements the self-verification component of TrustPath, which prompts
the LLM to evaluate its own outputs for potential errors or uncertainties.
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional

import anthropic
from anthropic import Anthropic

from config import SELF_VERIFICATION_CONFIG, LLM_CONFIG
from fix_anthropic import fix_anthropic_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelfVerificationModule:
    """
    Self-verification module that prompts the LLM to evaluate its own outputs.
    
    As described in the TrustPath proposal, this module implements Algorithm 1: Self-Verification Process.
    It asks the LLM to identify statements that might be uncertain or require verification,
    assign confidence scores, and generate alternative formulations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the self-verification module.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
        """
        # Initialize the Anthropic client
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.confidence_threshold = SELF_VERIFICATION_CONFIG["confidence_threshold"]
        self.verification_prompt_template = SELF_VERIFICATION_CONFIG["verification_prompt_template"]
        
        logger.info(f"Initialized SelfVerificationModule with model: {self.model}")
    
    def _generate_verification_prompt(self, question: str, response: str) -> str:
        """
        Generate a verification prompt for the LLM.
        
        Args:
            question: The original question asked by the user
            response: The original response from the LLM
            
        Returns:
            A verification prompt string
        """
        return self.verification_prompt_template.format(
            question=question,
            response=response
        )
    
    async def verify(self, question: str, response: str) -> Dict[str, Any]:
        """
        Verify the LLM's response by prompting it to evaluate its own output.
        
        Args:
            question: The original question asked by the user
            response: The original response from the LLM
            
        Returns:
            A dictionary containing the verification results, including:
            - problematic_statements: List of statements with low confidence
            - confidence_scores: Confidence scores for each problematic statement
            - alternative_formulations: Alternative formulations for each problematic statement
            - explanations: Explanations for why each statement might be problematic
        """
        logger.info(f"Verifying response to question: {question[:50]}...")
        
        # Generate the verification prompt
        verification_prompt = self._generate_verification_prompt(question, response)
        
        try:
            # Get verification response from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an expert at critically analyzing text for accuracy and identifying potential errors or uncertainties. Be thorough and specific in your analysis.",
                messages=[
                    {"role": "user", "content": verification_prompt}
                ]
            )
            verification_response = await fix_anthropic_response(message)
            
            # Parse the verification response
            return self._parse_verification_response(verification_response, response)
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {
                "problematic_statements": [],
                "confidence_scores": [],
                "alternative_formulations": [],
                "explanations": [],
                "error": str(e)
            }
    
    def _parse_verification_response(self, verification_response: str, original_response: str) -> Dict[str, Any]:
        """
        Parse the verification response to extract problematic statements and their metadata.
        
        Args:
            verification_response: The response from the verification prompt
            original_response: The original LLM response being verified
            
        Returns:
            A dictionary containing the parsed verification results
        """
        # Regular expression to find confidence scores
        # This regex looks for patterns like "confidence: 75%" or "confidence score: 75%"
        confidence_pattern = r"confidence(?:\s+score)?:?\s*(\d+)%"
        
        # Split the verification response into sections based on statement analysis
        sections = re.split(r"(?:\n\n|\n\*\*|\n\d+\.)", verification_response)
        
        problematic_statements = []
        confidence_scores = []
        alternative_formulations = []
        explanations = []
        
        for section in sections:
            if not section.strip():
                continue
                
            # Extract confidence score
            confidence_matches = re.search(confidence_pattern, section, re.IGNORECASE)
            if confidence_matches:
                confidence = int(confidence_matches.group(1)) / 100.0
                
                # If below threshold, extract the rest of the data
                if confidence < self.confidence_threshold:
                    # Try to extract the statement being evaluated
                    statement = self._extract_statement(section, original_response)
                    
                    # Extract explanation
                    explanation = self._extract_explanation(section)
                    
                    # Extract alternative formulation
                    alternative = self._extract_alternative(section)
                    
                    # Add to results
                    problematic_statements.append(statement)
                    confidence_scores.append(confidence)
                    alternative_formulations.append(alternative)
                    explanations.append(explanation)
        
        return {
            "problematic_statements": problematic_statements,
            "confidence_scores": confidence_scores,
            "alternative_formulations": alternative_formulations,
            "explanations": explanations,
        }
    
    def _extract_statement(self, section: str, original_response: str) -> str:
        """
        Extract the statement being evaluated from the section.
        
        Args:
            section: A section of the verification response
            original_response: The original LLM response
            
        Returns:
            The extracted statement
        """
        # Try to find quoted text which is likely the statement
        statement_match = re.search(r'"([^"]+)"', section)
        if statement_match:
            return statement_match.group(1)
        
        # If no quotes, look for the beginning of the section up to a common separator
        first_line = section.split('\n')[0].strip()
        if first_line and len(first_line) < 200:  # Reasonable statement length
            return first_line
        
        # If all else fails, return the section
        return section[:200] + "..." if len(section) > 200 else section
    
    def _extract_explanation(self, section: str) -> str:
        """
        Extract the explanation for why the statement might be problematic.
        
        Args:
            section: A section of the verification response
            
        Returns:
            The extracted explanation
        """
        # Look for explanation patterns
        explanation_patterns = [
            r"(?:because|reason|explanation):\s*(.+?)(?:\n\n|\n(?:[A-Z]|\d+\.)|$)",
            r"(?:unsure|uncertain|low confidence)(?:[.:]\s*)(.+?)(?:\n\n|\n(?:[A-Z]|\d+\.)|$)"
        ]
        
        for pattern in explanation_patterns:
            explanation_match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
            if explanation_match:
                return explanation_match.group(1).strip()
        
        # If no explanation found, return empty string
        return ""
    
    def _extract_alternative(self, section: str) -> str:
        """
        Extract the alternative formulation from the section.
        
        Args:
            section: A section of the verification response
            
        Returns:
            The extracted alternative formulation
        """
        # Look for alternative patterns
        alternative_patterns = [
            r"(?:alternative|better formulation|more accurate|instead):\s*(.+?)(?:\n\n|\n(?:[A-Z]|\d+\.)|$)",
            r"(?:could be|should be|might be):\s*(.+?)(?:\n\n|\n(?:[A-Z]|\d+\.)|$)"
        ]
        
        for pattern in alternative_patterns:
            alternative_match = re.search(pattern, section, re.IGNORECASE | re.DOTALL)
            if alternative_match:
                return alternative_match.group(1).strip()
        
        # If no alternative found, return empty string
        return ""

# Synchronous version of verify for easier testing
def verify(question: str, response: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around SelfVerificationModule.verify.
    
    Args:
        question: The original question asked by the user
        response: The original response from the LLM
        api_key: The API key for the LLM service
        
    Returns:
        A dictionary containing the verification results
    """
    import asyncio
    
    module = SelfVerificationModule(api_key=api_key)
    return asyncio.run(module.verify(question, response))

if __name__ == "__main__":
    # Simple test of the self-verification module
    test_question = "What's the capital of France and when was the Eiffel Tower built?"
    test_response = """
    The capital of France is Paris. The Eiffel Tower was built in 1889 as the entrance arch for the 1889 World's Fair. It was designed by engineer Gustave Eiffel and completed in just over two years, having started construction in January 1887. The tower stands 324 meters tall and was the tallest man-made structure in the world until the completion of the Chrysler Building in New York City in 1930.
    """
    
    print("Testing self-verification module...")
    verification_results = verify(test_question, test_response)
    print(json.dumps(verification_results, indent=2))