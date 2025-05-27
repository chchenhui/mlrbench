"""
Human-in-the-Loop Feedback System for TrustPath.

This module simulates the human feedback component of TrustPath, which
enables users to provide feedback on detected errors and suggested corrections.
"""

import json
import logging
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from config import HUMAN_FEEDBACK_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HumanFeedbackSimulator:
    """
    Simulates the human-in-the-loop feedback system described in the TrustPath proposal.
    
    This system collects feedback on error detections and correction suggestions,
    which can then be used to improve the system's performance over time.
    """
    
    def __init__(self, ground_truth: Dict[str, Any] = None):
        """
        Initialize the human feedback simulator.
        
        Args:
            ground_truth: Optional ground truth data for more realistic simulation
        """
        self.feedback_probability = HUMAN_FEEDBACK_CONFIG["feedback_probability"]
        self.feedback_accuracy = HUMAN_FEEDBACK_CONFIG["feedback_accuracy"]
        self.ground_truth = ground_truth or {}
        self.feedback_history = []
        
        logger.info("Initialized HumanFeedbackSimulator")
    
    def simulate_feedback(self, 
                          original_response: str, 
                          detected_errors: List[Dict[str, Any]], 
                          suggested_corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate human feedback on detected errors and suggested corrections.
        
        Args:
            original_response: The original LLM response
            detected_errors: List of detected errors with metadata
            suggested_corrections: List of suggested corrections with metadata
            
        Returns:
            A dictionary containing the simulated feedback
        """
        logger.info(f"Simulating human feedback on {len(detected_errors)} detected errors")
        
        # Initialize feedback
        feedback = {
            "validations": [],
            "correction_assessments": [],
            "user_corrections": [],
            "feedback_context": {"user_expertise": self._simulate_user_expertise()}
        }
        
        # Process each detected error
        for i, error in enumerate(detected_errors):
            # Only provide feedback with a certain probability
            if random.random() > self.feedback_probability:
                continue
            
            # Get corresponding correction if available
            correction = suggested_corrections[i] if i < len(suggested_corrections) else None
            
            # Validate the error detection (true/false positive)
            is_true_positive = self._validate_error_detection(error)
            
            # Simulate feedback on the validation with some noise based on accuracy
            if random.random() < self.feedback_accuracy:
                validation_feedback = is_true_positive
            else:
                validation_feedback = not is_true_positive
            
            feedback["validations"].append({
                "error_id": i,
                "error_content": error.get("content", ""),
                "is_valid_detection": validation_feedback
            })
            
            # If there's a corresponding correction and it's a true positive, assess the correction
            if correction and validation_feedback:
                correction_quality = self._assess_correction_quality(error, correction)
                
                feedback["correction_assessments"].append({
                    "error_id": i,
                    "correction_content": correction.get("content", ""),
                    "quality_rating": correction_quality,
                    "notes": self._generate_feedback_notes(correction_quality)
                })
                
                # Occasionally provide a user-generated correction
                if correction_quality < 4 and random.random() < 0.3:
                    feedback["user_corrections"].append({
                        "error_id": i,
                        "user_correction": self._generate_user_correction(error)
                    })
        
        # Store feedback in history
        self.feedback_history.append({
            "original_response": original_response,
            "detected_errors": detected_errors,
            "suggested_corrections": suggested_corrections,
            "feedback": feedback
        })
        
        return feedback
    
    def _validate_error_detection(self, error: Dict[str, Any]) -> bool:
        """
        Determine if an error detection is a true positive.
        
        In a real system, this would be based on user input. In this simulation,
        we use a combination of ground truth (if available) and heuristics.
        
        Args:
            error: The detected error with metadata
            
        Returns:
            Boolean indicating if the detection is valid (true positive)
        """
        # If we have ground truth data, use it
        if self.ground_truth:
            error_content = error.get("content", "")
            for true_error in self.ground_truth.get("errors", []):
                # Check if the detected error matches a known error
                if error_content in true_error or true_error in error_content:
                    return True
            
            # If we checked all known errors and found no match, it's likely a false positive
            # But we still allow for some to be true positives that weren't in our ground truth
            return random.random() < 0.2
        
        # Without ground truth, use confidence scores and randomness
        confidence = error.get("confidence_score", 0.5)
        
        # Higher confidence errors are more likely to be true positives
        true_positive_probability = 1 - confidence
        true_positive_probability = max(0.1, min(0.9, true_positive_probability))
        
        return random.random() < true_positive_probability
    
    def _assess_correction_quality(self, error: Dict[str, Any], correction: Dict[str, Any]) -> int:
        """
        Assess the quality of a suggested correction on a scale from 1 to 5.
        
        Args:
            error: The detected error
            correction: The suggested correction
            
        Returns:
            Quality rating from 1 (poor) to 5 (excellent)
        """
        # In a real system, this would be based on user input
        # In this simulation, we use heuristics and randomness
        
        # If we have ground truth data, use it
        if self.ground_truth:
            correction_content = correction.get("content", "")
            for true_correction in self.ground_truth.get("corrections", []):
                # Check if the suggested correction matches a known good correction
                if correction_content in true_correction or true_correction in correction_content:
                    # Even good corrections have a range of quality
                    return random.randint(4, 5)
            
            # If no match, it's an average to poor correction
            return random.randint(1, 3)
        
        # Without ground truth, use confidence and randomness
        confidence = correction.get("confidence_score", 0.5)
        
        # More confident corrections tend to be better
        mean_quality = 1 + 4 * confidence
        
        # Add randomness
        quality = int(np.random.normal(mean_quality, 1))
        
        # Ensure it's in range 1-5
        return max(1, min(5, quality))
    
    def _generate_feedback_notes(self, quality_rating: int) -> str:
        """
        Generate simulated feedback notes based on quality rating.
        
        Args:
            quality_rating: The quality rating from 1 to 5
            
        Returns:
            Simulated feedback notes
        """
        if quality_rating >= 4:
            notes = [
                "Good correction, accurate and clear.",
                "This correction is accurate and helpful.",
                "Well worded and factually correct.",
                "The correction addresses the issue well."
            ]
        elif quality_rating == 3:
            notes = [
                "The correction is okay but could be clearer.",
                "Somewhat helpful but not perfect.",
                "Correct information but awkwardly phrased.",
                "The correction is accurate but incomplete."
            ]
        else:
            notes = [
                "This correction is not accurate.",
                "The suggestion doesn't address the real issue.",
                "The correction contains new errors.",
                "This doesn't help much and might confuse users."
            ]
        
        return random.choice(notes)
    
    def _generate_user_correction(self, error: Dict[str, Any]) -> str:
        """
        Generate a simulated user-provided correction.
        
        Args:
            error: The detected error
            
        Returns:
            Simulated user correction
        """
        # In a real system, this would be user input
        # In this simulation, we use the ground truth if available, or generate something plausible
        
        if self.ground_truth:
            error_content = error.get("content", "")
            for i, true_error in enumerate(self.ground_truth.get("errors", [])):
                if error_content in true_error or true_error in error_content:
                    # If we have a matching correction, use it
                    if i < len(self.ground_truth.get("corrections", [])):
                        return self.ground_truth["corrections"][i]
        
        # Generate a placeholder correction
        return f"[Simulated user correction for: {error.get('content', '')[:30]}...]"
    
    def _simulate_user_expertise(self) -> str:
        """
        Simulate the expertise level of the user providing feedback.
        
        Returns:
            A string indicating the user's expertise level
        """
        expertise_levels = ["novice", "intermediate", "expert"]
        weights = [0.3, 0.5, 0.2]  # More intermediate users than novices or experts
        
        return random.choices(expertise_levels, weights=weights, k=1)[0]
    
    def get_feedback_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of feedback collected.
        
        Returns:
            The feedback history
        """
        return self.feedback_history
    
    def clear_feedback_history(self) -> None:
        """
        Clear the feedback history.
        """
        self.feedback_history = []
        logger.info("Feedback history cleared")

# Helper function to create a ground truth dataset for testing
def create_test_ground_truth() -> Dict[str, Any]:
    """
    Create a test ground truth dataset for simulating more realistic feedback.
    
    Returns:
        A dictionary with ground truth data
    """
    return {
        "errors": [
            "The Eiffel Tower was built in 1878",
            "The Eiffel Tower is located in Lyon, France",
            "The Eiffel Tower is made entirely of copper",
            "The Eiffel Tower is 124 meters tall"
        ],
        "corrections": [
            "The Eiffel Tower was built in 1889",
            "The Eiffel Tower is located in Paris, France",
            "The Eiffel Tower is made primarily of wrought iron",
            "The Eiffel Tower is 324 meters tall"
        ]
    }

if __name__ == "__main__":
    # Simple test of the human feedback simulator
    test_ground_truth = create_test_ground_truth()
    
    simulator = HumanFeedbackSimulator(ground_truth=test_ground_truth)
    
    test_response = """
    The Eiffel Tower was built in 1878 and is located in Lyon, France. It was designed by Gustave Eiffel and is made entirely of copper. The tower is 124 meters tall and weighs approximately 7,300 tons.
    """
    
    test_errors = [
        {"content": "The Eiffel Tower was built in 1878", "confidence_score": 0.3},
        {"content": "The Eiffel Tower is located in Lyon, France", "confidence_score": 0.2},
        {"content": "The Eiffel Tower is made entirely of copper", "confidence_score": 0.1}
    ]
    
    test_corrections = [
        {"content": "The Eiffel Tower was built in 1889", "confidence_score": 0.8},
        {"content": "The Eiffel Tower is located in Paris, France", "confidence_score": 0.9},
        {"content": "The Eiffel Tower is made primarily of iron", "confidence_score": 0.7}
    ]
    
    print("Testing human feedback simulator...")
    feedback = simulator.simulate_feedback(test_response, test_errors, test_corrections)
    print(json.dumps(feedback, indent=2))