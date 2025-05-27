"""
AI Cognitive Tutor Module

This module implements the AI Cognitive Tutor that detects user misunderstanding
and provides adaptive tutoring interventions to improve user understanding.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import logging

class AICognitiveTutor:
    """
    The AI Cognitive Tutor that monitors user interaction, detects misunderstandings,
    and provides adaptive tutoring interventions.
    """
    
    def __init__(
        self,
        ai_diagnostic,
        activation_threshold: float = 0.7,
        strategies: Dict[str, bool] = None,
        triggers: Dict[str, bool] = None
    ):
        """
        Initialize the AI Cognitive Tutor.
        
        Args:
            ai_diagnostic: The AI diagnostic system to explain
            activation_threshold: Threshold for activating the tutor (0.0-1.0)
            strategies: Dictionary of tutoring strategies to use
            triggers: Dictionary of misunderstanding triggers to detect
        """
        self.ai_diagnostic = ai_diagnostic
        self.activation_threshold = activation_threshold
        self.strategies = strategies or {
            "simplified_explanation": True,
            "analogies": True,
            "interactive_qa": True,
            "visualizations": True,
            "micro_learning": True,
            "contrastive_explanation": True
        }
        self.triggers = triggers or {
            "repeated_queries": True,
            "inconsistent_actions": True,
            "ignoring_uncertainty": True,
            "confusion_signals": True,
            "prolonged_hesitation": True
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized AI Cognitive Tutor")
        
        # Initialize user model to track user's understanding
        self.user_models = {}
        
        # Load knowledge base for tutoring strategies
        self._load_tutoring_knowledge()
    
    def _load_tutoring_knowledge(self):
        """Load knowledge for tutoring strategies"""
        # Simplified explanations for medical concepts
        self.simplified_explanations = {
            "feature_importance": (
                "Feature importance shows which symptoms most strongly influenced the AI's diagnosis. "
                "Larger values (closer to 1.0) mean that symptom was more important for the diagnosis."
            ),
            "confidence_score": (
                "The confidence score shows how certain the AI is about its diagnosis. "
                "A higher score (closer to 100%) means the AI is more certain, while a lower score means it's less sure."
            ),
            "uncertainty_estimate": (
                "The uncertainty estimate tells you how much the AI is unsure about its diagnosis. "
                "Low uncertainty means the AI is quite confident, medium uncertainty means some doubt, "
                "and high uncertainty means the AI has significant doubts about its diagnosis."
            ),
            "model_mechanism": (
                "This AI analyzes patterns in symptoms and compares them to known medical conditions. "
                "It uses statistical patterns from thousands of cases to make predictions, but doesn't 'think' like a human doctor."
            )
        }
        
        # Analogies to explain AI concepts
        self.analogies = {
            "feature_importance": (
                "Think of feature importance like detective clues. Some clues (like fingerprints) "
                "are stronger evidence than others (like a witness seeing someone 'similar'). "
                "The AI ranks symptoms by how strongly they point to a specific diagnosis."
            ),
            "confidence_score": (
                "The AI's confidence is like a weather forecast. A 90% chance of rain means you should "
                "definitely bring an umbrella, while a 60% chance means it could go either way. "
                "Similarly, higher confidence in a diagnosis means it's more likely to be correct, "
                "but isn't a guarantee."
            ),
            "uncertainty_estimate": (
                "Think of AI uncertainty like blurriness in a photo. When the image is clear (low uncertainty), "
                "you can easily identify what you're seeing. When it's blurry (high uncertainty), "
                "you might need more information or a second opinion."
            ),
            "model_limitations": (
                "The AI is like a cookbook that's learned many recipes but doesn't understand the chemistry of cooking. "
                "It recognizes patterns but doesn't truly understand underlying medical mechanisms the way a doctor would."
            )
        }
        
        # Micro-learning snippets for key concepts
        self.micro_learning = {
            "ai_confidence": (
                "AI Confidence vs. Uncertainty: An AI can be confident but wrong, like a student who "
                "studied the wrong material for an exam. High confidence means the AI thinks it knows "
                "the answer, but doesn't guarantee correctness. Always consider both the confidence score "
                "AND the uncertainty estimate when evaluating the AI's diagnosis."
            ),
            "rare_conditions": (
                "AI and Rare Conditions: AI systems are trained primarily on common cases. For rare conditions, "
                "the AI may have seen fewer examples, making it less reliable. When symptoms don't clearly "
                "match common conditions, consider the possibility of rare conditions even if the AI doesn't suggest them."
            ),
            "feature_correlations": (
                "Symptoms often correlate with each other. The AI might give high importance to a symptom "
                "not because it directly causes the condition, but because it frequently appears alongside "
                "other symptoms of that condition. This is correlation, not necessarily causation."
            ),
            "data_limitations": (
                "AI training data may have demographic biases or gaps. Conditions might present differently "
                "across age, sex, race, or other factors. If your patient has characteristics underrepresented "
                "in training data, the AI may be less accurate."
            )
        }
        
        # Interactive Q&A templates
        self.qa_templates = {
            "feature_importance": [
                "What specific symptom most influenced this diagnosis?",
                "How would the diagnosis change if symptom X was not present?",
                "Why is symptom X considered more important than symptom Y?"
            ],
            "confidence": [
                "Why is the AI uncertain about this diagnosis?",
                "What additional information would increase the AI's confidence?",
                "What alternative diagnoses is the AI considering?"
            ],
            "general_understanding": [
                "What type of data was the AI trained on?",
                "How does the AI handle rare or unusual cases?",
                "What are the limitations of this AI system?"
            ]
        }
        
        # Contrastive explanation templates
        self.contrastive_templates = {
            "similar_conditions": (
                "The AI diagnosed {condition_1} instead of {condition_2} primarily because of "
                "the presence of {key_feature_1} and the absence of {key_feature_2}, which would "
                "be more indicative of {condition_2}."
            ),
            "confidence_differences": (
                "While both {condition_1} and {condition_2} are possibilities, the AI is more confident "
                "in {condition_1} ({confidence_1}%) than {condition_2} ({confidence_2}%) because "
                "the pattern of symptoms more consistently matches known cases of {condition_1}."
            )
        }
        
        self.logger.info("Loaded tutoring knowledge base")
    
    def detect_misunderstanding(
        self, 
        user_id: str,
        user_behavior: Dict[str, Any], 
        diagnosis: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        """
        Detect if the user is misunderstanding the AI system based on their behavior.
        
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
        # Initialize or retrieve user model
        if user_id not in self.user_models:
            self.user_models[user_id] = {
                "expertise_level": user_behavior.get("expertise_level", "novice"),
                "misunderstanding_history": [],
                "successful_interventions": [],
                "interaction_count": 0
            }
        
        user_model = self.user_models[user_id]
        user_model["interaction_count"] += 1
        
        # Calculate probability of misunderstanding for different trigger types
        misunderstanding_probs = {}
        
        # 1. Check for repeated queries about the same information
        if self.triggers.get("repeated_queries", False):
            recent_queries = user_behavior.get("recent_queries", [])
            unique_queries = len(set(recent_queries))
            if len(recent_queries) > 0:
                repetition_ratio = 1 - (unique_queries / len(recent_queries))
                if repetition_ratio > 0.3:  # More than 30% repeated queries
                    misunderstanding_probs["repeated_queries"] = min(1.0, repetition_ratio * 1.5)
        
        # 2. Check for inconsistent actions with AI recommendation
        if self.triggers.get("inconsistent_actions", False):
            ai_recommendation = diagnosis.get("predicted_condition")
            user_decision = user_behavior.get("user_decision")
            confidence = diagnosis.get("confidence", 0.5)
            
            if user_decision and user_decision != ai_recommendation:
                # If AI is highly confident but user disagrees
                if confidence > 0.85:
                    misunderstanding_probs["inconsistent_actions"] = 0.8
                # If AI has medium confidence and user disagrees
                elif confidence > 0.7:
                    misunderstanding_probs["inconsistent_actions"] = 0.5
                # Low confidence disagreements are reasonable
        
        # 3. Check if user ignores high uncertainty warnings
        if self.triggers.get("ignoring_uncertainty", False):
            uncertainty = diagnosis.get("uncertainty", {}).get("level")
            acknowledged_uncertainty = user_behavior.get("acknowledged_uncertainty", False)
            
            if uncertainty == "high" and not acknowledged_uncertainty:
                misunderstanding_probs["ignoring_uncertainty"] = 0.9
            elif uncertainty == "medium" and not acknowledged_uncertainty:
                misunderstanding_probs["ignoring_uncertainty"] = 0.6
        
        # 4. Check for explicit confusion signals
        if self.triggers.get("confusion_signals", False):
            confusion_level = user_behavior.get("confusion_level", 0)
            if confusion_level > 7:  # 0-10 scale
                misunderstanding_probs["confusion_signals"] = 0.9
            elif confusion_level > 4:
                misunderstanding_probs["confusion_signals"] = 0.6
        
        # 5. Check for prolonged hesitation
        if self.triggers.get("prolonged_hesitation", False):
            decision_time = user_behavior.get("decision_time", 0)
            average_time = user_behavior.get("average_decision_time", 30)
            
            if decision_time > 2 * average_time:
                misunderstanding_probs["prolonged_hesitation"] = 0.7
        
        # Determine if any trigger exceeds the activation threshold
        if misunderstanding_probs:
            max_trigger = max(misunderstanding_probs.items(), key=lambda x: x[1])
            trigger_type, trigger_prob = max_trigger
            
            if trigger_prob >= self.activation_threshold:
                # Record the misunderstanding in user model
                user_model["misunderstanding_history"].append({
                    "type": trigger_type,
                    "probability": trigger_prob,
                    "interaction_id": user_model["interaction_count"]
                })
                
                self.logger.info(f"Detected misunderstanding: {trigger_type} with probability {trigger_prob:.2f}")
                return True, trigger_type, trigger_prob
        
        return False, "", 0.0
    
    def generate_intervention(
        self,
        user_id: str,
        misunderstanding_type: str,
        diagnosis: Dict[str, Any],
        user_behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an appropriate tutoring intervention based on the detected misunderstanding.
        
        Args:
            user_id: Unique identifier for the user
            misunderstanding_type: Type of misunderstanding detected
            diagnosis: The diagnosis produced by the AI
            user_behavior: Dictionary containing user interaction data
            
        Returns:
            Dictionary containing the tutoring intervention
        """
        user_model = self.user_models.get(user_id, {
            "expertise_level": "novice",
            "misunderstanding_history": [],
            "successful_interventions": [],
            "interaction_count": 0
        })
        
        expertise_level = user_model.get("expertise_level", "novice")
        
        # Map misunderstanding types to relevant concepts that need explanation
        concept_mapping = {
            "repeated_queries": ["model_mechanism", "feature_importance"],
            "inconsistent_actions": ["confidence_score", "model_limitations"],
            "ignoring_uncertainty": ["uncertainty_estimate", "ai_confidence"],
            "confusion_signals": ["feature_importance", "model_mechanism"],
            "prolonged_hesitation": ["confidence_score", "feature_correlations"]
        }
        
        # Select relevant concepts based on misunderstanding type
        relevant_concepts = concept_mapping.get(misunderstanding_type, ["model_mechanism"])
        primary_concept = relevant_concepts[0] if relevant_concepts else "model_mechanism"
        
        # Choose an appropriate tutoring strategy based on user expertise and history
        available_strategies = [s for s, enabled in self.strategies.items() if enabled]
        
        # Weight strategies based on user expertise and misunderstanding type
        strategy_weights = {
            "novice": {
                "simplified_explanation": 0.35,
                "analogies": 0.30,
                "micro_learning": 0.20,
                "visualizations": 0.10,
                "interactive_qa": 0.03,
                "contrastive_explanation": 0.02
            },
            "intermediate": {
                "simplified_explanation": 0.20,
                "analogies": 0.20,
                "micro_learning": 0.15,
                "visualizations": 0.15,
                "interactive_qa": 0.15,
                "contrastive_explanation": 0.15
            },
            "expert": {
                "simplified_explanation": 0.10,
                "analogies": 0.15,
                "micro_learning": 0.20,
                "visualizations": 0.15,
                "interactive_qa": 0.20,
                "contrastive_explanation": 0.20
            }
        }
        
        # Filter to only include available strategies
        expertise_weights = strategy_weights.get(expertise_level, strategy_weights["novice"])
        available_weights = {s: w for s, w in expertise_weights.items() if s in available_strategies}
        
        # Normalize weights
        total_weight = sum(available_weights.values())
        normalized_weights = {s: w/total_weight for s, w in available_weights.items()}
        
        # Select strategy based on weights
        strategies = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        selected_strategy = random.choices(strategies, weights=weights, k=1)[0]
        
        # Generate the intervention content based on the selected strategy and concept
        intervention_content = self._generate_strategy_content(
            strategy=selected_strategy,
            concept=primary_concept,
            diagnosis=diagnosis,
            user_behavior=user_behavior
        )
        
        # Construct the complete intervention
        intervention = {
            "type": selected_strategy,
            "concept": primary_concept,
            "content": intervention_content,
            "misunderstanding_type": misunderstanding_type,
            "timestamp": user_model["interaction_count"]
        }
        
        self.logger.info(f"Generated {selected_strategy} intervention for {primary_concept}")
        
        return intervention
    
    def _generate_strategy_content(
        self,
        strategy: str,
        concept: str,
        diagnosis: Dict[str, Any],
        user_behavior: Dict[str, Any]
    ) -> str:
        """
        Generate the specific content for a tutoring strategy.
        
        Args:
            strategy: The tutoring strategy to use
            concept: The concept to explain
            diagnosis: The diagnosis produced by the AI
            user_behavior: Dictionary containing user interaction data
            
        Returns:
            The generated content as a string
        """
        if strategy == "simplified_explanation":
            # Return a simplified explanation for the concept
            if concept in self.simplified_explanations:
                return self.simplified_explanations[concept]
            else:
                return self.simplified_explanations.get("model_mechanism", "")
        
        elif strategy == "analogies":
            # Return an analogy for the concept
            if concept in self.analogies:
                return self.analogies[concept]
            else:
                return self.analogies.get("model_limitations", "")
        
        elif strategy == "micro_learning":
            # Return a micro-learning snippet
            if concept in self.micro_learning:
                return self.micro_learning[concept]
            else:
                # Fallback to a general micro-learning snippet
                concepts = list(self.micro_learning.keys())
                return self.micro_learning[random.choice(concepts)]
        
        elif strategy == "interactive_qa":
            # Return Q&A options relevant to the concept
            if concept == "feature_importance":
                return "\n".join(self.qa_templates["feature_importance"])
            elif concept in ["confidence_score", "uncertainty_estimate", "ai_confidence"]:
                return "\n".join(self.qa_templates["confidence"])
            else:
                return "\n".join(self.qa_templates["general_understanding"])
        
        elif strategy == "contrastive_explanation":
            # Generate a contrastive explanation comparing the predicted condition with an alternative
            predicted_condition = diagnosis.get("predicted_condition")
            
            # Find an alternative condition (e.g., from overlapping conditions)
            alternative_conditions = self.ai_diagnostic.condition_overlaps.get(predicted_condition, [])
            if alternative_conditions:
                alternative = random.choice(alternative_conditions)
                
                # Find distinguishing features
                pred_features = self.ai_diagnostic.condition_features.get(predicted_condition, [])
                alt_features = self.ai_diagnostic.condition_features.get(alternative, [])
                
                unique_to_pred = [f for f in pred_features if f not in alt_features]
                unique_to_alt = [f for f in alt_features if f not in pred_features]
                
                if unique_to_pred and unique_to_alt:
                    key_feature_1 = random.choice(unique_to_pred)
                    key_feature_2 = random.choice(unique_to_alt)
                    
                    return self.contrastive_templates["similar_conditions"].format(
                        condition_1=predicted_condition,
                        condition_2=alternative,
                        key_feature_1=key_feature_1,
                        key_feature_2=key_feature_2
                    )
                else:
                    # If no unique features, use confidence difference template
                    return self.contrastive_templates["confidence_differences"].format(
                        condition_1=predicted_condition,
                        condition_2=alternative,
                        confidence_1=int(diagnosis.get("confidence", 0.7) * 100),
                        confidence_2=int(diagnosis.get("confidence", 0.7) * 70)  # Simulate lower confidence for alternative
                    )
            else:
                # Fallback to a simplified explanation if no alternative condition
                return self.simplified_explanations.get(concept, self.simplified_explanations["model_mechanism"])
        
        elif strategy == "visualizations":
            # In a real system, this would generate a visualization
            # For simulation, we'll just return a text description
            return f"[Visualization for {concept}]"
        
        else:
            # Fallback to simplified explanation
            return self.simplified_explanations.get(concept, self.simplified_explanations["model_mechanism"])
    
    def process_feedback(self, user_id: str, intervention_id: int, feedback: Dict[str, Any]):
        """
        Process user feedback on a tutoring intervention.
        
        Args:
            user_id: Unique identifier for the user
            intervention_id: ID of the intervention receiving feedback
            feedback: Dictionary containing feedback data
        """
        user_model = self.user_models.get(user_id)
        if not user_model:
            self.logger.warning(f"No user model found for user {user_id}")
            return
        
        # Record feedback in user model
        helpfulness = feedback.get("helpfulness", 0)  # 0-10 scale
        understanding_improved = feedback.get("understanding_improved", False)
        
        if helpfulness > 7 or understanding_improved:
            user_model["successful_interventions"].append({
                "intervention_id": intervention_id,
                "helpfulness": helpfulness,
                "understanding_improved": understanding_improved
            })
        
        # Update user expertise level based on accumulated feedback
        successful_count = len(user_model["successful_interventions"])
        total_interventions = len(user_model["misunderstanding_history"])
        
        if total_interventions > 10:
            # Adjust expertise level based on intervention success
            if successful_count / total_interventions > 0.8:
                # User seems to be learning quickly
                if user_model["expertise_level"] == "novice":
                    user_model["expertise_level"] = "intermediate"
                elif user_model["expertise_level"] == "intermediate":
                    user_model["expertise_level"] = "expert"
        
        self.logger.info(f"Processed feedback for user {user_id}, intervention {intervention_id}")
    
    def get_user_model(self, user_id: str) -> Dict[str, Any]:
        """
        Get the current model for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            The user model dictionary
        """
        return self.user_models.get(user_id, {
            "expertise_level": "novice",
            "misunderstanding_history": [],
            "successful_interventions": [],
            "interaction_count": 0
        })