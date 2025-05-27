"""
TrustPath: A Framework for Transparent Error Detection and Correction in LLMs

This module implements the TrustPath framework that integrates three components:
1. Self-verification module
2. Factual consistency checker
3. Human-in-the-loop feedback system
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

import anthropic
from anthropic import Anthropic

from config import LLM_CONFIG, ROOT_DIR, RESULTS_DIR
from self_verification import SelfVerificationModule
from factual_checker import FactualConsistencyChecker
from human_feedback import HumanFeedbackSimulator
from fix_anthropic import fix_anthropic_response

# Set up logging to file and console
log_file = RESULTS_DIR / "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrustPath:
    """
    TrustPath framework for transparent error detection and correction in LLMs.
    
    This class integrates the three main components of TrustPath:
    1. Self-verification module
    2. Factual consistency checker
    3. Human-in-the-loop feedback system
    
    It provides methods to analyze LLM responses, detect potential errors,
    suggest corrections, and collect human feedback.
    """
    
    def __init__(self, api_key: Optional[str] = None, ground_truth: Dict[str, Any] = None):
        """
        Initialize the TrustPath framework.
        
        Args:
            api_key: The API key for the LLM service. If None, uses environment variable.
            ground_truth: Optional ground truth data for simulating human feedback
        """
        # Initialize the Anthropic client for generating responses
        self.client = Anthropic(api_key=api_key)
        self.model = LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        
        # Initialize the components
        self.self_verification = SelfVerificationModule(api_key=api_key)
        self.factual_checker = FactualConsistencyChecker(api_key=api_key)
        self.human_feedback = HumanFeedbackSimulator(ground_truth=ground_truth)
        
        logger.info(f"Initialized TrustPath framework with model: {self.model}")
    
    async def generate_response(self, question: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            question: The question to ask the LLM
            
        Returns:
            The LLM's response
        """
        logger.info(f"Generating response to question: {question[:50]}...")
        
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
            
            logger.info(f"Generated response of length {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return f"Error generating response: {e}"
    
    async def analyze_response(self, question: str, response: str) -> Dict[str, Any]:
        """
        Analyze a response using both self-verification and factual checking.
        
        Args:
            question: The original question
            response: The LLM's response
            
        Returns:
            A dictionary containing the analysis results from both components
        """
        logger.info(f"Analyzing response with TrustPath...")
        
        # Run both components in parallel for efficiency
        self_verification_task = self.self_verification.verify(question, response)
        factual_checking_task = self.factual_checker.check_response(response)
        
        # Wait for both tasks to complete
        self_verification_result, factual_checking_result = await asyncio.gather(
            self_verification_task, factual_checking_task
        )
        
        # Combine the results
        combined_results = {
            "original_question": question,
            "original_response": response,
            "self_verification": self_verification_result,
            "factual_checking": factual_checking_result,
            "detected_errors": self._combine_detected_errors(
                self_verification_result, factual_checking_result
            ),
        }
        
        # Add suggested corrections
        combined_results["suggested_corrections"] = await self._generate_corrections(
            combined_results["detected_errors"],
            self_verification_result,
            factual_checking_result
        )
        
        return combined_results
    
    def _combine_detected_errors(self, 
                                self_verification_result: Dict[str, Any], 
                                factual_checking_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Combine detected errors from both components.
        
        Args:
            self_verification_result: Result from the self-verification module
            factual_checking_result: Result from the factual consistency checker
            
        Returns:
            A unified list of detected errors
        """
        errors = []
        
        # Add errors from self-verification
        for i, statement in enumerate(self_verification_result.get("problematic_statements", [])):
            if i < len(self_verification_result.get("confidence_scores", [])):
                confidence = self_verification_result["confidence_scores"][i]
                explanation = self_verification_result.get("explanations", [""])[i] if i < len(self_verification_result.get("explanations", [])) else ""
                
                errors.append({
                    "source": "self_verification",
                    "content": statement,
                    "confidence_score": confidence,
                    "explanation": explanation,
                    "error_id": f"sv_{i}"
                })
        
        # Add errors from factual checking
        for i, claim_result in enumerate(factual_checking_result.get("claim_results", [])):
            if claim_result.get("is_erroneous", False):
                claim = claim_result.get("claim", "")
                score = claim_result.get("verification_score", 0.0)
                evidence = claim_result.get("contradicting_evidence", [])
                
                # Extract explanation from evidence if available
                explanation = ""
                if evidence and "explanation" in evidence[0]:
                    explanation = evidence[0]["explanation"]
                
                errors.append({
                    "source": "factual_checking",
                    "content": claim,
                    "confidence_score": 1 - score,  # Invert score (lower score = higher confidence of error)
                    "explanation": explanation,
                    "error_id": f"fc_{i}"
                })
        
        return errors
    
    async def _generate_corrections(self, 
                                   detected_errors: List[Dict[str, Any]],
                                   self_verification_result: Dict[str, Any],
                                   factual_checking_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate correction suggestions for detected errors.
        
        Args:
            detected_errors: Combined list of detected errors
            self_verification_result: Result from the self-verification module
            factual_checking_result: Result from the factual consistency checker
            
        Returns:
            A list of suggested corrections
        """
        corrections = []
        
        for error in detected_errors:
            error_id = error.get("error_id", "")
            source = error.get("source", "")
            content = error.get("content", "")
            
            if source == "self_verification":
                # Extract index from error_id (sv_0, sv_1, etc.)
                try:
                    index = int(error_id.split("_")[1])
                    alternatives = self_verification_result.get("alternative_formulations", [])
                    if index < len(alternatives) and alternatives[index]:
                        corrections.append({
                            "error_id": error_id,
                            "content": alternatives[index],
                            "confidence_score": 0.7,  # Default confidence for self-verification corrections
                            "source": "self_verification"
                        })
                        continue
                except (ValueError, IndexError):
                    pass
            
            elif source == "factual_checking":
                # Extract index from error_id (fc_0, fc_1, etc.)
                try:
                    index = int(error_id.split("_")[1])
                    claim_results = factual_checking_result.get("claim_results", [])
                    if index < len(claim_results) and claim_results[index].get("correction_suggestion"):
                        corrections.append({
                            "error_id": error_id,
                            "content": claim_results[index]["correction_suggestion"],
                            "confidence_score": 0.8,  # Default confidence for factual checking corrections
                            "source": "factual_checking"
                        })
                        continue
                except (ValueError, IndexError):
                    pass
            
            # If we couldn't find a correction from the components, generate one
            correction = await self._generate_fallback_correction(content, error.get("explanation", ""))
            corrections.append({
                "error_id": error_id,
                "content": correction,
                "confidence_score": 0.5,  # Lower confidence for fallback corrections
                "source": "fallback"
            })
        
        return corrections
    
    async def _generate_fallback_correction(self, error_content: str, explanation: str) -> str:
        """
        Generate a fallback correction when component-specific corrections are unavailable.
        
        Args:
            error_content: The content of the detected error
            explanation: Explanation of why it's an error
            
        Returns:
            A suggested correction
        """
        logger.info(f"Generating fallback correction for: {error_content[:50]}...")
        
        correction_prompt = f"""
        Correct the following statement that contains an error:
        
        Statement: "{error_content}"
        
        Error explanation: "{explanation}"
        
        Provide a corrected version of the statement that addresses the error.
        Make minimal changes necessary to fix the issue.
        Return ONLY the corrected statement, no additional text.
        """
        
        try:
            # Get correction from the LLM
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,  # Lower temperature for more accurate correction
                system="You are an expert at correcting factual errors. Be accurate and precise in your corrections.",
                messages=[
                    {"role": "user", "content": correction_prompt}
                ]
            )
            correction_response = await fix_anthropic_response(message)
            correction = correction_response.strip()
            return correction
            
        except Exception as e:
            logger.error(f"Error during fallback correction generation: {e}")
            return f"[Unable to generate correction: {e}]"
    
    async def collect_feedback(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect simulated human feedback on the analysis results.
        
        Args:
            analysis_results: The analysis results containing detected errors and corrections
            
        Returns:
            The feedback results
        """
        logger.info("Collecting simulated human feedback...")
        
        original_response = analysis_results.get("original_response", "")
        detected_errors = analysis_results.get("detected_errors", [])
        suggested_corrections = analysis_results.get("suggested_corrections", [])
        
        feedback = self.human_feedback.simulate_feedback(
            original_response, detected_errors, suggested_corrections
        )
        
        # Add the feedback to the analysis results
        analysis_results["human_feedback"] = feedback
        
        return feedback
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the entire TrustPath pipeline.
        
        Args:
            question: The question to process
            
        Returns:
            The complete analysis results
        """
        logger.info(f"Processing question: {question[:50]}...")
        
        # Generate response
        response = await self.generate_response(question)
        
        # Analyze response
        analysis_results = await self.analyze_response(question, response)
        
        # Collect feedback
        feedback = await self.collect_feedback(analysis_results)
        
        return analysis_results
    
    def get_visual_representation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate data for visualizing the analysis results.
        
        This would feed into the visual interface described in the proposal.
        For the experiment, we return structured data that can be visualized.
        
        Args:
            analysis_results: The analysis results
            
        Returns:
            Data for visualization
        """
        original_response = analysis_results.get("original_response", "")
        detected_errors = analysis_results.get("detected_errors", [])
        suggested_corrections = analysis_results.get("suggested_corrections", [])
        
        # Create a list of spans to highlight
        spans = []
        
        for i, error in enumerate(detected_errors):
            content = error.get("content", "")
            confidence = error.get("confidence_score", 0.5)
            explanation = error.get("explanation", "")
            error_id = error.get("error_id", f"error_{i}")
            
            # Find the error in the original response
            start_idx = original_response.find(content)
            if start_idx != -1:
                end_idx = start_idx + len(content)
                
                # Get the correction if available
                correction = ""
                for corr in suggested_corrections:
                    if corr.get("error_id") == error_id:
                        correction = corr.get("content", "")
                        break
                
                # Determine confidence level color
                if confidence > 0.8:
                    confidence_level = "low"  # High confidence that it's an error
                elif confidence > 0.6:
                    confidence_level = "medium"
                else:
                    confidence_level = "high"  # Low confidence that it's an error
                
                spans.append({
                    "start": start_idx,
                    "end": end_idx,
                    "text": content,
                    "confidence_level": confidence_level,
                    "confidence_score": confidence,
                    "explanation": explanation,
                    "correction": correction,
                    "error_id": error_id,
                    "source": error.get("source", "")
                })
        
        # Create the visualization data
        visualization_data = {
            "original_text": original_response,
            "spans": spans,
            "corrections": suggested_corrections
        }
        
        return visualization_data
    
    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save analysis results to a JSON file.
        
        Args:
            results: The analysis results
            filename: The filename to save to
            
        Returns:
            The path to the saved file
        """
        file_path = RESULTS_DIR / filename
        
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {file_path}")
        return str(file_path)

# Synchronous version of process_question for easier testing
def process_question(question: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous wrapper around TrustPath.process_question.
    
    Args:
        question: The question to process
        api_key: The API key for the LLM service
        
    Returns:
        The complete analysis results
    """
    import asyncio
    
    trustpath = TrustPath(api_key=api_key)
    return asyncio.run(trustpath.process_question(question))

if __name__ == "__main__":
    # Simple test of the TrustPath framework
    test_question = "What's the capital of France and when was the Eiffel Tower built?"
    
    print("Testing TrustPath framework...")
    results = process_question(test_question)
    
    # Save results
    trustpath = TrustPath()
    filepath = trustpath.save_results(results, "test_results.json")
    print(f"Results saved to {filepath}")