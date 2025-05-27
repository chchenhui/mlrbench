"""
Experiment Module

This module orchestrates the experimental workflow, including:
1. Setting up participants in control and treatment groups
2. Running diagnostic trials
3. Collecting and organizing results
"""

import numpy as np
import random
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import os
from pathlib import Path

from simulation.participant import SimulatedParticipant

class Experiment:
    """
    Manages the experimental workflow for evaluating the AI Cognitive Tutor.
    """
    
    def __init__(
        self,
        ai_diagnostic,
        cognitive_tutor,
        baselines,
        config,
        logger=None
    ):
        """
        Initialize the experiment.
        
        Args:
            ai_diagnostic: The AI diagnostic system
            cognitive_tutor: The AI Cognitive Tutor
            baselines: Dictionary of baseline explanation methods
            config: Configuration dictionary
            logger: Logger instance
        """
        self.ai_diagnostic = ai_diagnostic
        self.cognitive_tutor = cognitive_tutor
        self.baselines = baselines
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize result containers
        self.results = {
            "summary_metrics": {},
            "participant_data": [],
            "trial_data": []
        }
        
        # Initialize participant groups
        self.treatment_group = []
        self.control_group = []
        
        # Random seed for reproducibility
        random.seed(config['experiment']['random_seed'])
        np.random.seed(config['experiment']['random_seed'])
        
        self.logger.info("Initialized experiment")
    
    def setup_participants(self):
        """Set up simulated participants for control and treatment groups"""
        num_participants = self.config['experiment']['num_simulated_participants']
        participants_per_group = num_participants // 2
        
        self.logger.info(f"Setting up {num_participants} simulated participants")
        
        # Define expertise distribution
        expertise_levels = self.config['participants']['expertise_levels']
        expertise_distribution = self.config['participants']['expertise_distribution']
        
        # Generate participants for treatment group
        self.treatment_group = self._generate_group_participants(
            count=participants_per_group,
            expertise_levels=expertise_levels,
            expertise_distribution=expertise_distribution,
            group="treatment"
        )
        
        # Generate participants for control group
        self.control_group = self._generate_group_participants(
            count=participants_per_group,
            expertise_levels=expertise_levels,
            expertise_distribution=expertise_distribution,
            group="control"
        )
        
        self.logger.info(f"Created {len(self.treatment_group)} treatment group participants and {len(self.control_group)} control group participants")
    
    def _generate_group_participants(self, count, expertise_levels, expertise_distribution, group):
        """Generate participants for a group with specified distribution"""
        participants = []
        
        # Calculate how many participants of each expertise level
        counts = []
        remaining = count
        for i, prob in enumerate(expertise_distribution[:-1]):
            level_count = int(count * prob)
            counts.append(level_count)
            remaining -= level_count
        counts.append(remaining)  # Add remaining to last category
        
        # Generate participants
        for i, (level, level_count) in enumerate(zip(expertise_levels, counts)):
            for j in range(level_count):
                participant_id = f"{group}_{level}_{j+1}"
                
                # Randomize trust in AI, learning rate, and attention span
                trust_in_ai = random.uniform(0.5, 0.9)
                learning_rate = random.uniform(0.05, 0.2)
                attention_span = random.uniform(0.7, 0.95)
                
                # Create participant
                participant = SimulatedParticipant(
                    participant_id=participant_id,
                    expertise_level=level,
                    trust_in_ai=trust_in_ai,
                    learning_rate=learning_rate,
                    attention_span=attention_span,
                    group=group
                )
                
                participants.append(participant)
        
        return participants
    
    def generate_diagnostic_trials(self, num_trials):
        """Generate a set of diagnostic trials with varying complexity"""
        self.logger.info(f"Generating {num_trials} diagnostic trials")
        
        trials = []
        
        # Balance complexity
        complexities = ["simple", "medium", "complex"]
        complexity_counts = {
            "simple": int(num_trials * 0.3),
            "medium": int(num_trials * 0.4),
            "complex": int(num_trials * 0.3) + (num_trials - int(num_trials * 0.3) - int(num_trials * 0.4))
        }
        
        # Generate trials
        for complexity in complexities:
            for i in range(complexity_counts[complexity]):
                trial_id = len(trials) + 1
                
                # Randomly select a condition from the AI diagnostic's knowledge base
                condition = random.choice(self.ai_diagnostic.conditions)
                
                # Generate a patient case
                patient_case = self.ai_diagnostic.generate_patient_case(
                    complexity=complexity,
                    true_condition=condition
                )
                
                # Add overlapping conditions to the patient case
                overlapping_conditions = self.ai_diagnostic.condition_overlaps.get(condition, [])
                all_conditions = [condition] + overlapping_conditions
                patient_case["all_conditions"] = all_conditions
                
                # Create trial
                trial = {
                    "trial_id": trial_id,
                    "patient_case": patient_case,
                    "complexity": complexity
                }
                
                trials.append(trial)
        
        # Shuffle trials to randomize order
        random.shuffle(trials)
        
        return trials
    
    def run_treatment_group(self, trials):
        """Run the experiment for the treatment group (with AI Cognitive Tutor)"""
        self.logger.info("Running treatment group trials")
        
        results = []
        
        for participant in tqdm(self.treatment_group, desc="Treatment Group"):
            participant_results = []
            
            for trial in trials:
                trial_id = trial["trial_id"]
                patient_case = trial["patient_case"]
                
                # Generate AI diagnosis
                diagnosis = self.ai_diagnostic.diagnose(
                    patient_data=patient_case,
                    complexity=trial["complexity"]
                )
                
                # Simulate participant decision
                decision = participant.make_decision(diagnosis, patient_case)
                
                # Check for misunderstanding
                misunderstanding_detected, misunderstanding_type, misunderstanding_prob = \
                    self.cognitive_tutor.detect_misunderstanding(
                        user_id=participant.participant_id,
                        user_behavior=decision,
                        diagnosis=diagnosis
                    )
                
                # Generate intervention if misunderstanding detected
                if misunderstanding_detected:
                    intervention = self.cognitive_tutor.generate_intervention(
                        user_id=participant.participant_id,
                        misunderstanding_type=misunderstanding_type,
                        diagnosis=diagnosis,
                        user_behavior=decision
                    )
                else:
                    intervention = {"type": "none", "content": ""}
                
                # Process intervention feedback
                feedback = participant.process_intervention(intervention)
                
                # If intervention was generated, process the feedback
                if misunderstanding_detected:
                    self.cognitive_tutor.process_feedback(
                        user_id=participant.participant_id,
                        intervention_id=intervention.get("timestamp", 0),
                        feedback=feedback
                    )
                
                # Record trial results
                trial_result = {
                    "participant_id": participant.participant_id,
                    "trial_id": trial_id,
                    "group": "treatment",
                    "complexity": trial["complexity"],
                    "true_condition": patient_case["true_condition"],
                    "ai_diagnosis": diagnosis["predicted_condition"],
                    "user_decision": decision["user_decision"],
                    "ai_correct": diagnosis["predicted_condition"] == patient_case["true_condition"],
                    "user_correct": decision["user_decision"] == patient_case["true_condition"],
                    "confidence": diagnosis["confidence"],
                    "uncertainty": diagnosis["uncertainty"]["level"],
                    "misunderstanding_detected": misunderstanding_detected,
                    "misunderstanding_type": misunderstanding_type if misunderstanding_detected else "",
                    "intervention_type": intervention["type"],
                    "helpfulness": feedback.get("helpfulness", 0),
                    "understanding_improved": feedback.get("understanding_improved", False),
                    "mental_model_accuracy": participant.get_mental_model_accuracy(),
                    "decision_time": decision["decision_time"],
                    "confusion_level": decision["confusion_level"]
                }
                
                participant_results.append(trial_result)
            
            # Collect participant summary
            participant_summary = participant.get_summary()
            self.results["participant_data"].append(participant_summary)
            
            # Add all trial results
            results.extend(participant_results)
        
        return results
    
    def run_control_group(self, trials):
        """Run the experiment for the control group (with baseline explanations)"""
        self.logger.info("Running control group trials")
        
        results = []
        
        # Get the baseline method
        baseline_method = None
        if "standard_explanation" in self.baselines:
            baseline_method = self.baselines["standard_explanation"]
        elif "static_tutorial" in self.baselines:
            baseline_method = self.baselines["static_tutorial"]
        elif "no_explanation" in self.baselines:
            baseline_method = self.baselines["no_explanation"]
        else:
            # Fallback to standard explanation if no baseline specified
            from models.baselines import StandardExplanation
            baseline_method = StandardExplanation(self.ai_diagnostic)
        
        for participant in tqdm(self.control_group, desc="Control Group"):
            participant_results = []
            
            for trial in trials:
                trial_id = trial["trial_id"]
                patient_case = trial["patient_case"]
                
                # Generate AI diagnosis
                diagnosis = self.ai_diagnostic.diagnose(
                    patient_data=patient_case,
                    complexity=trial["complexity"]
                )
                
                # Generate baseline explanation
                if isinstance(baseline_method, type(self.baselines.get("static_tutorial", None))):
                    explanation = baseline_method.generate_explanation(diagnosis, participant.participant_id)
                else:
                    explanation = baseline_method.generate_explanation(diagnosis)
                
                # Simulate participant decision
                decision = participant.make_decision(diagnosis, patient_case)
                
                # Process explanation as an intervention
                feedback = participant.process_intervention(explanation)
                
                # Record trial results
                trial_result = {
                    "participant_id": participant.participant_id,
                    "trial_id": trial_id,
                    "group": "control",
                    "complexity": trial["complexity"],
                    "true_condition": patient_case["true_condition"],
                    "ai_diagnosis": diagnosis["predicted_condition"],
                    "user_decision": decision["user_decision"],
                    "ai_correct": diagnosis["predicted_condition"] == patient_case["true_condition"],
                    "user_correct": decision["user_decision"] == patient_case["true_condition"],
                    "confidence": diagnosis["confidence"],
                    "uncertainty": diagnosis["uncertainty"]["level"],
                    "misunderstanding_detected": False,
                    "misunderstanding_type": "",
                    "intervention_type": explanation["type"],
                    "helpfulness": feedback.get("helpfulness", 0),
                    "understanding_improved": feedback.get("understanding_improved", False),
                    "mental_model_accuracy": participant.get_mental_model_accuracy(),
                    "decision_time": decision["decision_time"],
                    "confusion_level": decision["confusion_level"]
                }
                
                participant_results.append(trial_result)
            
            # Collect participant summary
            participant_summary = participant.get_summary()
            self.results["participant_data"].append(participant_summary)
            
            # Add all trial results
            results.extend(participant_results)
        
        return results
    
    def run(self):
        """Run the complete experiment and return results"""
        start_time = time.time()
        self.logger.info("Starting experiment")
        
        # Set up participants
        self.setup_participants()
        
        # Generate diagnostic trials
        num_trials = self.config['experiment']['num_trials']
        trials = self.generate_diagnostic_trials(num_trials)
        
        # Run treatment group (with Cognitive Tutor)
        treatment_results = self.run_treatment_group(trials)
        
        # Run control group (with baseline explanation)
        control_results = self.run_control_group(trials)
        
        # Combine results
        all_trial_results = treatment_results + control_results
        self.results["trial_data"] = all_trial_results
        
        # Calculate summary metrics
        self._calculate_summary_metrics()
        
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"Experiment completed in {duration:.2f} seconds")
        
        return self.results
    
    def _calculate_summary_metrics(self):
        """Calculate summary metrics from the experiment results"""
        self.logger.info("Calculating summary metrics")
        
        trial_data = self.results["trial_data"]
        participant_data = self.results["participant_data"]
        
        # Group trials by group
        treatment_trials = [t for t in trial_data if t["group"] == "treatment"]
        control_trials = [t for t in trial_data if t["group"] == "control"]
        
        # Group participants by group
        treatment_participants = [p for p in participant_data if p["group"] == "treatment"]
        control_participants = [p for p in participant_data if p["group"] == "control"]
        
        # Calculate mental model accuracy
        treatment_mma = np.mean([p["mental_model_accuracy"] for p in treatment_participants])
        control_mma = np.mean([p["mental_model_accuracy"] for p in control_participants])
        
        # Calculate diagnostic performance (percent correct decisions)
        treatment_accuracy = np.mean([1 if t["user_correct"] else 0 for t in treatment_trials])
        control_accuracy = np.mean([1 if t["user_correct"] else 0 for t in control_trials])
        
        # Calculate appropriate reliance
        def calculate_appropriate_reliance(trials):
            total_appropriate = 0
            total_trials = len(trials)
            
            for trial in trials:
                ai_correct = trial["ai_correct"]
                agrees_with_ai = trial["user_decision"] == trial["ai_diagnosis"]
                
                # Appropriate reliance = agreeing when AI is right, disagreeing when AI is wrong
                if (ai_correct and agrees_with_ai) or (not ai_correct and not agrees_with_ai):
                    total_appropriate += 1
            
            return total_appropriate / total_trials if total_trials > 0 else 0
        
        treatment_reliance = calculate_appropriate_reliance(treatment_trials)
        control_reliance = calculate_appropriate_reliance(control_trials)
        
        # Calculate user-AI misalignment incidents (using confusion level as proxy)
        treatment_confusion = np.mean([t["confusion_level"] for t in treatment_trials])
        control_confusion = np.mean([t["confusion_level"] for t in control_trials])
        
        # Calculate cognitive load (using decision time as proxy)
        treatment_decision_time = np.mean([t["decision_time"] for t in treatment_trials])
        control_decision_time = np.mean([t["decision_time"] for t in control_trials])
        
        # Calculate trust calibration (correlation between appropriateness of trust and AI reliability)
        # This is a simplified metric
        treatment_trust_calibration = treatment_reliance
        control_trust_calibration = control_reliance
        
        # Calculate tutor effectiveness (for treatment group)
        treatment_helpfulness = np.mean([t["helpfulness"] for t in treatment_trials if t["intervention_type"] != "none"])
        treatment_improvement = np.mean([1 if t["understanding_improved"] else 0 for t in treatment_trials if t["intervention_type"] != "none"])
        
        # Organize metrics by category
        self.results["summary_metrics"] = {
            "mental_model_accuracy": {
                "treatment": treatment_mma,
                "control": control_mma,
                "difference": treatment_mma - control_mma,
                "percent_improvement": (treatment_mma - control_mma) / control_mma * 100 if control_mma > 0 else 0
            },
            "diagnostic_performance": {
                "treatment": treatment_accuracy,
                "control": control_accuracy,
                "difference": treatment_accuracy - control_accuracy,
                "percent_improvement": (treatment_accuracy - control_accuracy) / control_accuracy * 100 if control_accuracy > 0 else 0
            },
            "appropriate_reliance": {
                "treatment": treatment_reliance,
                "control": control_reliance,
                "difference": treatment_reliance - control_reliance,
                "percent_improvement": (treatment_reliance - control_reliance) / control_reliance * 100 if control_reliance > 0 else 0
            },
            "user_ai_misalignment": {
                "treatment": treatment_confusion,
                "control": control_confusion,
                "difference": control_confusion - treatment_confusion,  # Reversed because lower is better
                "percent_improvement": (control_confusion - treatment_confusion) / control_confusion * 100 if control_confusion > 0 else 0
            },
            "cognitive_load": {
                "treatment": treatment_decision_time,
                "control": control_decision_time,
                "difference": control_decision_time - treatment_decision_time,  # Reversed because lower is better
                "percent_improvement": (control_decision_time - treatment_decision_time) / control_decision_time * 100 if control_decision_time > 0 else 0
            },
            "trust_calibration": {
                "treatment": treatment_trust_calibration,
                "control": control_trust_calibration,
                "difference": treatment_trust_calibration - control_trust_calibration,
                "percent_improvement": (treatment_trust_calibration - control_trust_calibration) / control_trust_calibration * 100 if control_trust_calibration > 0 else 0
            },
            "tutor_effectiveness": {
                "intervention_helpfulness": treatment_helpfulness,
                "understanding_improvement_rate": treatment_improvement
            }
        }
        
        # Add by-complexity metrics for diagnostic performance
        complexity_levels = ["simple", "medium", "complex"]
        
        for complexity in complexity_levels:
            treatment_by_complexity = [t for t in treatment_trials if t["complexity"] == complexity]
            control_by_complexity = [t for t in control_trials if t["complexity"] == complexity]
            
            treatment_accuracy_by_complexity = np.mean([1 if t["user_correct"] else 0 for t in treatment_by_complexity])
            control_accuracy_by_complexity = np.mean([1 if t["user_correct"] else 0 for t in control_by_complexity])
            
            self.results["summary_metrics"][f"diagnostic_performance_{complexity}"] = {
                "treatment": treatment_accuracy_by_complexity,
                "control": control_accuracy_by_complexity,
                "difference": treatment_accuracy_by_complexity - control_accuracy_by_complexity,
                "percent_improvement": (treatment_accuracy_by_complexity - control_accuracy_by_complexity) / control_accuracy_by_complexity * 100 if control_accuracy_by_complexity > 0 else 0
            }
        
        self.logger.info("Summary metrics calculated")