"""
AI Diagnostic System Module

This module simulates a complex AI diagnostic system that medical professionals
would interact with. It provides diagnoses with varying accuracy and uncertainty,
along with different types of explanations.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import logging

class AIDiagnosticSystem:
    """
    A simulated AI system for medical diagnosis that provides predictions
    with varying levels of uncertainty and different types of explanations.
    """
    
    def __init__(
        self,
        model_type: str = "medical_diagnosis",
        accuracy: float = 0.85,
        uncertainty_levels: int = 3,
        explanation_types: List[str] = ["feature_importance", "confidence_score", "uncertainty_estimate"]
    ):
        """
        Initialize the AI Diagnostic System.
        
        Args:
            model_type: Type of diagnostic model (e.g., "medical_diagnosis")
            accuracy: Base accuracy of the diagnostic model (0.0-1.0)
            uncertainty_levels: Number of uncertainty levels (e.g., 3 for low/medium/high)
            explanation_types: Types of explanations the system can provide
        """
        self.model_type = model_type
        self.accuracy = accuracy
        self.uncertainty_levels = uncertainty_levels
        self.explanation_types = explanation_types
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized {model_type} with accuracy {accuracy}")
        
        # Load simulated medical knowledge base (conditions, symptoms, treatments)
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load a simulated knowledge base of medical conditions"""
        # In a real system, this would load from a database or external source
        # For simulation, we'll create a simplified medical knowledge base
        
        self.conditions = [
            "Pneumonia",
            "Heart Failure",
            "Diabetes Type 2",
            "Rheumatoid Arthritis",
            "Parkinson's Disease",
            "Multiple Sclerosis",
            "Chronic Kidney Disease",
            "Asthma",
            "Chronic Obstructive Pulmonary Disease",
            "Osteoporosis"
        ]
        
        # Map conditions to their primary features/symptoms (simplified)
        self.condition_features = {
            "Pneumonia": ["Cough", "Fever", "Shortness of Breath", "Chest Pain", "Sputum Production"],
            "Heart Failure": ["Shortness of Breath", "Fatigue", "Edema", "Irregular Heartbeat", "Reduced Exercise Capacity"],
            "Diabetes Type 2": ["Increased Thirst", "Frequent Urination", "Increased Hunger", "Weight Loss", "Fatigue"],
            "Rheumatoid Arthritis": ["Joint Pain", "Joint Swelling", "Joint Stiffness", "Fatigue", "Fever"],
            "Parkinson's Disease": ["Tremor", "Bradykinesia", "Rigid Muscles", "Impaired Posture", "Loss of Automatic Movements"],
            "Multiple Sclerosis": ["Fatigue", "Vision Problems", "Numbness", "Balance Problems", "Cognitive Issues"],
            "Chronic Kidney Disease": ["High Blood Pressure", "Edema", "Fatigue", "Urination Changes", "Nausea"],
            "Asthma": ["Shortness of Breath", "Chest Tightness", "Wheezing", "Coughing", "Sleep Difficulties"],
            "Chronic Obstructive Pulmonary Disease": ["Shortness of Breath", "Chronic Cough", "Sputum Production", "Wheezing", "Fatigue"],
            "Osteoporosis": ["Back Pain", "Height Loss", "Bone Fractures", "Stooped Posture", "Bone Pain"]
        }
        
        # Map features to their relative importance for each condition (0-10 scale)
        self.feature_importances = {
            "Pneumonia": {"Cough": 8, "Fever": 7, "Shortness of Breath": 9, "Chest Pain": 6, "Sputum Production": 7},
            "Heart Failure": {"Shortness of Breath": 9, "Fatigue": 7, "Edema": 8, "Irregular Heartbeat": 8, "Reduced Exercise Capacity": 7},
            "Diabetes Type 2": {"Increased Thirst": 9, "Frequent Urination": 9, "Increased Hunger": 7, "Weight Loss": 6, "Fatigue": 5},
            "Rheumatoid Arthritis": {"Joint Pain": 9, "Joint Swelling": 8, "Joint Stiffness": 8, "Fatigue": 6, "Fever": 5},
            "Parkinson's Disease": {"Tremor": 9, "Bradykinesia": 9, "Rigid Muscles": 8, "Impaired Posture": 7, "Loss of Automatic Movements": 7},
            "Multiple Sclerosis": {"Fatigue": 7, "Vision Problems": 8, "Numbness": 8, "Balance Problems": 7, "Cognitive Issues": 6},
            "Chronic Kidney Disease": {"High Blood Pressure": 8, "Edema": 7, "Fatigue": 6, "Urination Changes": 9, "Nausea": 5},
            "Asthma": {"Shortness of Breath": 9, "Chest Tightness": 8, "Wheezing": 9, "Coughing": 7, "Sleep Difficulties": 6},
            "Chronic Obstructive Pulmonary Disease": {"Shortness of Breath": 9, "Chronic Cough": 8, "Sputum Production": 7, "Wheezing": 8, "Fatigue": 6},
            "Osteoporosis": {"Back Pain": 7, "Height Loss": 6, "Bone Fractures": 9, "Stooped Posture": 7, "Bone Pain": 6}
        }
        
        # Overlap matrix showing conditions that share features
        # Used to create ambiguous cases for testing the tutor
        self.condition_overlaps = {
            "Pneumonia": ["Asthma", "Chronic Obstructive Pulmonary Disease", "Heart Failure"],
            "Heart Failure": ["Pneumonia", "Chronic Kidney Disease"],
            "Diabetes Type 2": ["Chronic Kidney Disease"],
            "Rheumatoid Arthritis": ["Osteoporosis"],
            "Parkinson's Disease": ["Multiple Sclerosis"],
            "Multiple Sclerosis": ["Parkinson's Disease"],
            "Chronic Kidney Disease": ["Heart Failure", "Diabetes Type 2"],
            "Asthma": ["Chronic Obstructive Pulmonary Disease", "Pneumonia"],
            "Chronic Obstructive Pulmonary Disease": ["Asthma", "Pneumonia"],
            "Osteoporosis": ["Rheumatoid Arthritis"]
        }
        
        self.logger.info(f"Loaded knowledge base with {len(self.conditions)} conditions")
    
    def diagnose(self, patient_data: Dict[str, Any], complexity: str = "medium") -> Dict[str, Any]:
        """
        Generate a diagnosis based on patient data.
        
        Args:
            patient_data: Dictionary containing patient symptoms and medical history
            complexity: Complexity of the case ("simple", "medium", "complex")
            
        Returns:
            Dictionary containing the diagnosis, confidence, uncertainty, and explanations
        """
        # Extract symptoms from patient data
        symptoms = patient_data.get("symptoms", [])
        
        # Determine the true condition (for simulation purposes)
        true_condition = patient_data.get("true_condition", random.choice(self.conditions))
        
        # Determine if the AI will be correct (based on accuracy and complexity)
        complexity_modifier = {"simple": 0.1, "medium": 0, "complex": -0.1}
        effective_accuracy = min(1.0, max(0.0, self.accuracy + complexity_modifier[complexity]))
        is_correct = random.random() < effective_accuracy
        
        # Set predicted condition
        if is_correct:
            predicted_condition = true_condition
        else:
            # If incorrect, choose a condition that shares some symptoms (if available)
            overlapping_conditions = self.condition_overlaps.get(true_condition, [])
            if overlapping_conditions:
                predicted_condition = random.choice(overlapping_conditions)
            else:
                # Otherwise, just pick a random different condition
                options = [c for c in self.conditions if c != true_condition]
                predicted_condition = random.choice(options)
        
        # Calculate confidence score (higher for correct predictions, but not perfect)
        base_confidence = 0.7 if is_correct else 0.6
        confidence_noise = random.uniform(-0.15, 0.15)
        confidence = min(0.99, max(0.5, base_confidence + confidence_noise))
        
        # Determine uncertainty level
        if confidence > 0.8:
            uncertainty = "low"
            uncertainty_value = random.uniform(0.1, 0.3)
        elif confidence > 0.65:
            uncertainty = "medium"
            uncertainty_value = random.uniform(0.3, 0.6)
        else:
            uncertainty = "high"
            uncertainty_value = random.uniform(0.6, 0.9)
        
        # Generate feature importance explanation
        feature_importance = {}
        if "feature_importance" in self.explanation_types:
            # Get the model's understanding of feature importance for the predicted condition
            model_importances = self.feature_importances.get(predicted_condition, {})
            
            # Add some noise to simulate model imperfection
            for feature, importance in model_importances.items():
                if feature in symptoms:
                    # Add noise to importance values
                    noise = random.uniform(-1.0, 1.0)
                    normalized_importance = (importance + noise) / 10.0  # Scale to 0-1
                    feature_importance[feature] = min(1.0, max(0.0, normalized_importance))
        
        # Construct the diagnosis object
        diagnosis = {
            "predicted_condition": predicted_condition,
            "confidence": confidence,
            "uncertainty": {
                "level": uncertainty,
                "value": uncertainty_value
            },
            "is_correct": is_correct,  # This would not be known in a real system, used for evaluation
            "explanations": {}
        }
        
        # Add explanations based on configured types
        if "feature_importance" in self.explanation_types:
            diagnosis["explanations"]["feature_importance"] = feature_importance
        
        if "confidence_score" in self.explanation_types:
            diagnosis["explanations"]["confidence_score"] = {
                "value": confidence,
                "explanation": f"The model is {confidence*100:.1f}% confident in this diagnosis."
            }
        
        if "uncertainty_estimate" in self.explanation_types:
            diagnosis["explanations"]["uncertainty_estimate"] = {
                "level": uncertainty,
                "value": uncertainty_value,
                "explanation": f"The model has {uncertainty} uncertainty about this diagnosis."
            }
        
        self.logger.info(f"Generated diagnosis: {predicted_condition} (Correct: {is_correct}, Confidence: {confidence:.2f})")
        
        return diagnosis
    
    def generate_explanation(self, diagnosis: Dict[str, Any], explanation_type: str) -> Dict[str, Any]:
        """
        Generate a specific type of explanation for a given diagnosis.
        
        Args:
            diagnosis: The diagnosis object returned by the diagnose method
            explanation_type: Type of explanation to generate
            
        Returns:
            Dictionary containing the explanation
        """
        if explanation_type not in self.explanation_types:
            self.logger.warning(f"Explanation type '{explanation_type}' not supported")
            return {"error": f"Explanation type '{explanation_type}' not supported"}
        
        predicted_condition = diagnosis.get("predicted_condition")
        
        if explanation_type == "feature_importance":
            # Return feature importance already in the diagnosis
            return diagnosis.get("explanations", {}).get("feature_importance", {})
        
        elif explanation_type == "confidence_score":
            return diagnosis.get("explanations", {}).get("confidence_score", {})
        
        elif explanation_type == "uncertainty_estimate":
            return diagnosis.get("explanations", {}).get("uncertainty_estimate", {})
        
        # Could add more explanation types here
        
        return {"error": "Unknown explanation type"}
    
    def generate_patient_case(self, complexity: str = "medium", true_condition: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a simulated patient case for testing.
        
        Args:
            complexity: Complexity of the case ("simple", "medium", "complex")
            true_condition: Optional specific condition to create a case for
            
        Returns:
            Dictionary containing patient data
        """
        if true_condition is None:
            true_condition = random.choice(self.conditions)
        
        # Get primary features for this condition
        primary_features = self.condition_features.get(true_condition, [])
        
        # Determine how many primary features to include based on complexity
        # More features = easier diagnosis
        if complexity == "simple":
            num_primary = min(len(primary_features), max(3, int(len(primary_features) * 0.8)))
        elif complexity == "medium":
            num_primary = min(len(primary_features), max(2, int(len(primary_features) * 0.6)))
        else:  # complex
            num_primary = min(len(primary_features), max(2, int(len(primary_features) * 0.4)))
        
        # Randomly select primary features
        selected_primary = random.sample(primary_features, num_primary)
        
        # Potentially add some overlapping features from other conditions
        overlapping_conditions = self.condition_overlaps.get(true_condition, [])
        overlapping_features = []
        
        for condition in overlapping_conditions:
            other_features = self.condition_features.get(condition, [])
            # Filter to features not already in our condition
            other_unique = [f for f in other_features if f not in primary_features]
            if other_unique and random.random() < 0.5:  # 50% chance to add confounding features
                # Add 1-2 confounding features
                num_confounding = random.randint(1, min(2, len(other_unique)))
                overlapping_features.extend(random.sample(other_unique, num_confounding))
        
        # Combine all symptoms
        all_symptoms = selected_primary + overlapping_features
        
        # Create patient case
        patient_case = {
            "true_condition": true_condition,
            "symptoms": all_symptoms,
            "complexity": complexity,
            "age": random.randint(25, 75),
            "sex": random.choice(["Male", "Female"]),
            # Could add more patient attributes here
        }
        
        return patient_case