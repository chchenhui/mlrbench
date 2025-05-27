"""
Simulated Participant Module

This module simulates user interaction with the AI diagnostic system,
providing a controlled environment for testing the AI Cognitive Tutor.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import logging

class SimulatedParticipant:
    """
    Simulates a human participant interacting with the AI diagnostic system.
    
    Each participant has different characteristics (e.g., expertise level,
    trust in AI, learning rate) that influence their behavior.
    """
    
    def __init__(
        self,
        participant_id: str,
        expertise_level: str = "novice",
        trust_in_ai: float = 0.7,
        learning_rate: float = 0.1,
        attention_span: float = 0.8,
        group: str = "control"
    ):
        """
        Initialize a simulated participant.
        
        Args:
            participant_id: Unique identifier for the participant
            expertise_level: Initial expertise level ("novice", "intermediate", "expert")
            trust_in_ai: Initial trust in AI systems (0.0-1.0)
            learning_rate: How quickly the participant learns (0.0-1.0)
            attention_span: How attentive the participant is to explanations (0.0-1.0)
            group: Experimental group ("control" or "treatment")
        """
        self.participant_id = participant_id
        self.expertise_level = expertise_level
        self.trust_in_ai = trust_in_ai
        self.learning_rate = learning_rate
        self.attention_span = attention_span
        self.group = group
        
        self.logger = logging.getLogger(__name__)
        
        # Track participant's knowledge and history
        self.knowledge = self._initialize_knowledge()
        self.decision_history = []
        self.intervention_history = []
        self.interaction_count = 0
        self.average_decision_time = random.uniform(20, 40)  # seconds
        
        self.logger.info(f"Initialized participant {participant_id} ({expertise_level}, {group} group)")
    
    def _initialize_knowledge(self) -> Dict[str, float]:
        """
        Initialize the participant's knowledge of AI concepts.
        
        Returns:
            Dictionary mapping concept names to knowledge levels (0.0-1.0)
        """
        # Base knowledge levels depending on expertise
        expertise_modifiers = {
            "novice": 0.2,
            "intermediate": 0.5,
            "expert": 0.8
        }
        
        base_level = expertise_modifiers.get(self.expertise_level, 0.2)
        
        # Add some random variation
        knowledge = {
            "feature_importance": min(1.0, max(0.0, base_level + random.uniform(-0.1, 0.1))),
            "confidence_score": min(1.0, max(0.0, base_level + random.uniform(-0.1, 0.1))),
            "uncertainty_estimate": min(1.0, max(0.0, base_level + random.uniform(-0.1, 0.1))),
            "model_mechanism": min(1.0, max(0.0, base_level + random.uniform(-0.1, 0.1))),
            "model_limitations": min(1.0, max(0.0, base_level + random.uniform(-0.1, 0.1))),
            "feature_correlations": min(1.0, max(0.0, base_level - 0.1 + random.uniform(-0.1, 0.1)))
        }
        
        return knowledge
    
    def make_decision(self, diagnosis: Dict[str, Any], patient_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate the participant making a diagnostic decision based on AI diagnosis.
        
        Args:
            diagnosis: The diagnosis produced by the AI
            patient_case: The patient case information
            
        Returns:
            Dictionary containing the participant's decision and behavior
        """
        self.interaction_count += 1
        
        # Extract key information from the diagnosis
        predicted_condition = diagnosis.get("predicted_condition")
        true_condition = patient_case.get("true_condition")
        ai_is_correct = predicted_condition == true_condition
        confidence = diagnosis.get("confidence", 0.7)
        uncertainty_level = diagnosis.get("uncertainty", {}).get("level", "medium")
        uncertainty_value = diagnosis.get("uncertainty", {}).get("value", 0.5)
        
        # Simulate decision-making based on participant characteristics
        
        # 1. Trust Factor: How much weight to give the AI's recommendation
        trust_factor = self.trust_in_ai
        
        # Expertise affects how critically they assess the AI
        expertise_modifier = {
            "novice": 0.1,      # Novices tend to rely more on AI
            "intermediate": 0,  # Neutral
            "expert": -0.1      # Experts tend to rely less on AI
        }.get(self.expertise_level, 0)
        
        trust_factor += expertise_modifier
        
        # 2. Understanding Factor: How well they understand the AI's rationale
        # This is based on their knowledge of relevant concepts
        relevant_concepts = ["feature_importance", "confidence_score", "uncertainty_estimate"]
        understanding_factor = sum(self.knowledge.get(c, 0) for c in relevant_concepts) / len(relevant_concepts)
        
        # 3. Decision making process
        
        # Chance of agreeing with AI (influenced by trust, understanding, and AI confidence)
        # Higher values make agreement more likely
        agreement_bias = (0.5 * trust_factor) + (0.3 * understanding_factor) + (0.2 * confidence)
        
        # Attention to uncertainty
        understands_uncertainty = self.knowledge.get("uncertainty_estimate", 0) > 0.5
        acknowledges_uncertainty = random.random() < (0.3 + 0.7 * understands_uncertainty)
        
        # Adjust agreement bias based on uncertainty (if they understand it)
        if understands_uncertainty and uncertainty_level == "high":
            agreement_bias *= 0.7  # Less likely to agree if high uncertainty
        
        # Make decision
        agree_with_ai = random.random() < agreement_bias
        
        # If they agree, take AI's prediction, otherwise make their own diagnosis
        if agree_with_ai:
            user_decision = predicted_condition
        else:
            # Simulate making an alternative diagnosis
            if random.random() < understanding_factor and ai_is_correct:
                # If they have good understanding and AI is right, more likely to also be right
                user_decision = true_condition
            else:
                # Otherwise, choose randomly (possibly the right answer)
                potential_conditions = [c for c in patient_case.get("all_conditions", []) if c != predicted_condition]
                if not potential_conditions:  # Fallback if no alternatives provided
                    user_decision = predicted_condition  # Default to agreeing
                else:
                    user_decision = random.choice(potential_conditions)
        
        # Simulate decision time (longer for complex cases or when uncertain)
        base_time = self.average_decision_time
        complexity_modifier = {"simple": 0.7, "medium": 1.0, "complex": 1.3}.get(patient_case.get("complexity", "medium"), 1.0)
        uncertainty_modifier = 1.0 + (uncertainty_value * 0.5)
        confusion_modifier = 1.0 + (1.0 - understanding_factor) * 0.5
        
        decision_time = base_time * complexity_modifier * uncertainty_modifier * confusion_modifier
        
        # Add some random variation to decision time
        decision_time *= random.uniform(0.8, 1.2)
        
        # Simulate confusion level (higher with lower understanding and higher uncertainty)
        confusion_level = max(0, min(10, 10 * (1 - understanding_factor) * (0.5 + 0.5 * uncertainty_value)))
        confusion_level += random.uniform(-1, 1)  # Add some noise
        confusion_level = max(0, min(10, confusion_level))
        
        # Simulate query behavior
        num_queries = 1 + int(confusion_level / 3)  # More confused = more queries
        # Simulate some repeated queries to indicate confusion
        if confusion_level > 5:
            repeated_ratio = 0.4  # 40% of queries are repeated when confused
        else:
            repeated_ratio = 0.1  # 10% of queries are repeated otherwise
        
        queries = []
        for _ in range(num_queries):
            if random.random() < repeated_ratio and queries:
                # Repeat a previous query
                queries.append(random.choice(queries))
            else:
                # Generate a new query
                query_options = [
                    "What does this feature mean?",
                    "Why is the confidence so low/high?",
                    "What other conditions were considered?",
                    "How reliable is this diagnosis?",
                    "What is the uncertainty level based on?",
                    "How does this AI work?",
                    "What symptoms support this diagnosis?"
                ]
                queries.append(random.choice(query_options))
        
        # Record decision
        decision = {
            "participant_id": self.participant_id,
            "interaction_id": self.interaction_count,
            "ai_diagnosis": predicted_condition,
            "user_decision": user_decision,
            "decision_time": decision_time,
            "agrees_with_ai": agree_with_ai,
            "acknowledged_uncertainty": acknowledges_uncertainty,
            "confusion_level": confusion_level,
            "recent_queries": queries,
            "expertise_level": self.expertise_level,
            "average_decision_time": self.average_decision_time
        }
        
        self.decision_history.append(decision)
        
        # Update trust based on outcome (if ground truth is known)
        if true_condition:
            self._update_trust(ai_is_correct, user_decision == true_condition)
        
        self.logger.debug(f"Participant {self.participant_id} made decision: {user_decision} (AI: {predicted_condition})")
        
        return decision
    
    def _update_trust(self, ai_correct: bool, user_correct: bool):
        """
        Update trust level based on decision outcomes.
        
        Args:
            ai_correct: Whether the AI's prediction was correct
            user_correct: Whether the user's decision was correct
        """
        # Trust increases if AI is correct, decreases if wrong
        if ai_correct:
            trust_change = 0.05  # Small increase for correct AI
        else:
            trust_change = -0.1  # Larger decrease for incorrect AI
        
        # Expertise affects how much trust changes
        if self.expertise_level == "expert":
            trust_change *= 0.7  # Experts change trust more gradually
        elif self.expertise_level == "novice":
            trust_change *= 1.3  # Novices change trust more dramatically
        
        # Update trust (bounded between 0.2 and 1.0)
        self.trust_in_ai = min(1.0, max(0.2, self.trust_in_ai + trust_change))
    
    def process_intervention(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and respond to a tutoring intervention.
        
        Args:
            intervention: The tutoring intervention to process
            
        Returns:
            Dictionary containing the participant's feedback on the intervention
        """
        if intervention.get("type") == "none":
            # No intervention, no learning
            return {"helpfulness": 0, "understanding_improved": False}
        
        # Extract relevant information
        intervention_type = intervention.get("type")
        concept = intervention.get("concept")
        content = intervention.get("content")
        
        # Check if participant attended to the intervention
        attended = random.random() < self.attention_span
        
        if not attended:
            # If they didn't pay attention, minimal learning occurs
            learning_amount = 0.01
            helpfulness = random.uniform(1, 4)  # Not very helpful if not paying attention
            understanding_improved = False
        else:
            # Base learning amount depends on intervention type and expertise
            type_effectiveness = {
                "simplified_explanation": 0.8 if self.expertise_level == "novice" else 0.5,
                "analogies": 0.7,
                "micro_learning": 0.6,
                "visualizations": 0.7,
                "interactive_qa": 0.8 if self.expertise_level == "expert" else 0.6,
                "contrastive_explanation": 0.7 if self.expertise_level == "expert" else 0.5,
                "standard": 0.5,  # For StandardExplanation baseline
                "tutorial_and_explanation": 0.6  # For StaticTutorial baseline
            }.get(intervention_type, 0.5)
            
            # Calculate learning amount
            learning_amount = type_effectiveness * self.learning_rate
            
            # Add some random variation
            learning_amount *= random.uniform(0.8, 1.2)
            
            # Calculate perceived helpfulness
            helpfulness = 5 + (type_effectiveness * 5 * random.uniform(0.8, 1.2))
            
            # Determine if understanding improved
            current_knowledge = self.knowledge.get(concept, 0)
            understanding_improved = (current_knowledge + learning_amount) - current_knowledge > 0.05
        
        # Update knowledge of the concept
        if concept in self.knowledge:
            self.knowledge[concept] = min(1.0, self.knowledge[concept] + learning_amount)
        
        # Record intervention response
        intervention_response = {
            "participant_id": self.participant_id,
            "intervention_id": intervention.get("timestamp", self.interaction_count),
            "intervention_type": intervention_type,
            "concept": concept,
            "attended": attended,
            "helpfulness": helpfulness,
            "understanding_improved": understanding_improved,
            "knowledge_before": self.knowledge.get(concept, 0) - learning_amount if concept in self.knowledge else 0,
            "knowledge_after": self.knowledge.get(concept, 0) if concept in self.knowledge else 0
        }
        
        self.intervention_history.append(intervention_response)
        
        # Update average decision time (people tend to get faster with experience)
        self.average_decision_time *= 0.98
        
        self.logger.debug(f"Participant {self.participant_id} processed intervention for concept {concept}")
        
        return {
            "helpfulness": helpfulness,
            "understanding_improved": understanding_improved
        }
    
    def get_mental_model_accuracy(self) -> float:
        """
        Calculate the accuracy of the participant's mental model of the AI.
        
        Returns:
            Mental model accuracy score (0.0-1.0)
        """
        # Calculate a weighted average of knowledge across all concepts
        concept_weights = {
            "feature_importance": 0.2,
            "confidence_score": 0.2,
            "uncertainty_estimate": 0.2,
            "model_mechanism": 0.15,
            "model_limitations": 0.15,
            "feature_correlations": 0.1
        }
        
        weighted_sum = sum(self.knowledge.get(concept, 0) * weight 
                          for concept, weight in concept_weights.items())
        total_weight = sum(concept_weights.values())
        
        return weighted_sum / total_weight
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the participant's characteristics and performance.
        
        Returns:
            Dictionary containing the summary data
        """
        return {
            "participant_id": self.participant_id,
            "group": self.group,
            "expertise_level": self.expertise_level,
            "trust_in_ai": self.trust_in_ai,
            "learning_rate": self.learning_rate,
            "attention_span": self.attention_span,
            "mental_model_accuracy": self.get_mental_model_accuracy(),
            "knowledge": self.knowledge,
            "num_interactions": self.interaction_count,
            "intervention_count": len(self.intervention_history)
        }