"""
Baseline Methods Module

This module implements various baseline explanation methods for comparison:
1. Standard Explanation: Typical AI explanation without adaptive tutoring
2. No Explanation: Just the diagnosis without explanations
3. Static Tutorial: A fixed tutorial about the AI at the beginning
"""

import random
from typing import Dict, List, Tuple, Any, Optional
import logging

class ExplanationBase:
    """Base class for all explanation methods"""
    
    def __init__(self, ai_diagnostic):
        """
        Initialize the explanation method.
        
        Args:
            ai_diagnostic: The AI diagnostic system to explain
        """
        self.ai_diagnostic = ai_diagnostic
        self.logger = logging.getLogger(__name__)
    
    def detect_misunderstanding(self, user_id: str, user_behavior: Dict[str, Any], diagnosis: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Base implementation for misunderstanding detection.
        
        Args:
            user_id: Unique identifier for the user
            user_behavior: Dictionary containing user interaction data
            diagnosis: The diagnosis produced by the AI
            
        Returns:
            Tuple containing:
            - Whether a misunderstanding was detected (bool)
            - The type of misunderstanding detected (str)
            - The probability of misunderstanding (float)
        """
        # By default, no misunderstanding detection
        return False, "", 0.0
    
    def generate_intervention(self, user_id: str, misunderstanding_type: str, diagnosis: Dict[str, Any], user_behavior: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base implementation for generating interventions.
        
        Args:
            user_id: Unique identifier for the user
            misunderstanding_type: Type of misunderstanding detected
            diagnosis: The diagnosis produced by the AI
            user_behavior: Dictionary containing user interaction data
            
        Returns:
            Dictionary containing the intervention details
        """
        # By default, no intervention
        return {"type": "none", "content": ""}
    
    def process_feedback(self, user_id: str, intervention_id: int, feedback: Dict[str, Any]):
        """
        Base implementation for processing feedback.
        
        Args:
            user_id: Unique identifier for the user
            intervention_id: ID of the intervention receiving feedback
            feedback: Dictionary containing feedback data
        """
        # By default, no feedback processing
        pass

class StandardExplanation(ExplanationBase):
    """
    Standard explanation method without adaptive tutoring.
    
    This baseline provides standard explanations for AI diagnoses
    without adapting to user misunderstandings.
    """
    
    def __init__(self, ai_diagnostic):
        super().__init__(ai_diagnostic)
        self.logger.info("Initialized Standard Explanation baseline")
    
    def generate_explanation(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a standard explanation for a diagnosis.
        
        Args:
            diagnosis: The diagnosis produced by the AI
            
        Returns:
            Dictionary containing the explanation
        """
        predicted_condition = diagnosis.get("predicted_condition", "")
        confidence = diagnosis.get("confidence", 0.0)
        uncertainty = diagnosis.get("uncertainty", {}).get("level", "unknown")
        
        # Get feature importance if available
        feature_importance = diagnosis.get("explanations", {}).get("feature_importance", {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Format the explanation text
        explanation_text = f"The AI system has diagnosed {predicted_condition} with {confidence*100:.1f}% confidence.\n"
        explanation_text += f"The uncertainty level is {uncertainty}.\n\n"
        
        if sorted_features:
            explanation_text += "Key factors influencing this diagnosis:\n"
            for feature, importance in sorted_features[:3]:  # Show top 3 features
                explanation_text += f"- {feature}: {importance*100:.1f}%\n"
        
        return {
            "type": "standard",
            "content": explanation_text,
            "condition": predicted_condition,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "feature_importance": dict(sorted_features[:3]) if sorted_features else {}
        }

class NoExplanation(ExplanationBase):
    """
    No explanation baseline.
    
    This baseline provides only the diagnosis without any explanations.
    """
    
    def __init__(self, ai_diagnostic):
        super().__init__(ai_diagnostic)
        self.logger.info("Initialized No Explanation baseline")
    
    def generate_explanation(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a minimal explanation (just the diagnosis).
        
        Args:
            diagnosis: The diagnosis produced by the AI
            
        Returns:
            Dictionary containing the minimal explanation
        """
        predicted_condition = diagnosis.get("predicted_condition", "")
        
        # Return just the diagnosis
        return {
            "type": "none",
            "content": f"The AI system has diagnosed {predicted_condition}.",
            "condition": predicted_condition
        }

class StaticTutorial(ExplanationBase):
    """
    Static tutorial baseline.
    
    This baseline provides a fixed tutorial about the AI system at the beginning,
    followed by standard explanations for each diagnosis.
    """
    
    def __init__(self, ai_diagnostic):
        super().__init__(ai_diagnostic)
        self.logger.info("Initialized Static Tutorial baseline")
        self.user_tutorials = {}  # Track which users have seen the tutorial
    
    def get_tutorial(self) -> str:
        """
        Get the static tutorial text.
        
        Returns:
            Tutorial text as a string
        """
        tutorial = """
        UNDERSTANDING THE AI DIAGNOSTIC SYSTEM
        
        This AI system analyzes patient symptoms and medical history to suggest potential diagnoses. 
        Here's how to interpret the AI's output:
        
        1. DIAGNOSIS: 
           The AI will suggest a diagnosis based on the symptoms provided.
        
        2. CONFIDENCE SCORE:
           - 85-100%: High confidence in the diagnosis
           - 65-85%: Moderate confidence
           - Below 65%: Low confidence
           
           Remember, high confidence doesn't guarantee correctness, especially for complex cases.
        
        3. UNCERTAINTY LEVEL:
           - Low: The AI is fairly certain of its diagnosis
           - Medium: The AI has some reservations about its diagnosis
           - High: The AI is quite uncertain and the diagnosis should be treated with caution
        
        4. FEATURE IMPORTANCE:
           The AI will show which symptoms most strongly influenced its diagnosis. 
           Higher percentages indicate more significant influence.
           
        LIMITATIONS:
        - The AI can only consider the symptoms provided
        - It may struggle with rare conditions or unusual symptom presentations
        - It doesn't replace the expertise of a human medical professional
        
        Please use this information as one input among many for your decision-making process.
        """
        
        return tutorial
    
    def generate_explanation(self, diagnosis: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Generate an explanation with or without the tutorial.
        
        Args:
            diagnosis: The diagnosis produced by the AI
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing the explanation and possibly tutorial
        """
        predicted_condition = diagnosis.get("predicted_condition", "")
        confidence = diagnosis.get("confidence", 0.0)
        uncertainty = diagnosis.get("uncertainty", {}).get("level", "unknown")
        
        # Get feature importance if available
        feature_importance = diagnosis.get("explanations", {}).get("feature_importance", {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Format the explanation text (same as StandardExplanation)
        explanation_text = f"The AI system has diagnosed {predicted_condition} with {confidence*100:.1f}% confidence.\n"
        explanation_text += f"The uncertainty level is {uncertainty}.\n\n"
        
        if sorted_features:
            explanation_text += "Key factors influencing this diagnosis:\n"
            for feature, importance in sorted_features[:3]:  # Show top 3 features
                explanation_text += f"- {feature}: {importance*100:.1f}%\n"
        
        # Check if the user has seen the tutorial
        show_tutorial = user_id not in self.user_tutorials
        if show_tutorial:
            self.user_tutorials[user_id] = True
            full_content = self.get_tutorial() + "\n\n" + explanation_text
            content_type = "tutorial_and_explanation"
        else:
            full_content = explanation_text
            content_type = "standard"
        
        return {
            "type": content_type,
            "content": full_content,
            "condition": predicted_condition,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "feature_importance": dict(sorted_features[:3]) if sorted_features else {},
            "tutorial_shown": show_tutorial
        }