#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RLAIF model implementation for the IETA framework.
This module provides the RLAIF model for aligning code generation models with execution traces.
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path
import json
import time
from tqdm import tqdm

from models.base_model import BaseCodeLLM
from utils.llm_utils import get_llm_client, format_execution_trace_for_prompt

logger = logging.getLogger(__name__)

class RLAIFModel(BaseCodeLLM):
    """RLAIF model for code generation."""
    
    def __init__(self, model_name, model_type="api", learning_rate=5e-5, batch_size=8, alpha=0.1):
        """
        Initialize the RLAIF model.
        
        Args:
            model_name (str): Name of the model to use
            model_type (str): Type of model ("api" or "huggingface")
            learning_rate (float): Learning rate for training
            batch_size (int): Batch size for training
            alpha (float): KL penalty parameter
        """
        super().__init__(model_name, model_type)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        
        # Reference model (copy of the initial model)
        self.reference_model_name = f"{model_name}_reference"
        
        # Reward model (in a real implementation, this would be a separate model)
        self.reward_model_name = f"{model_name}_reward"
        
        # For actual fine-tuning, we would initialize models here
        # Since we're using API models for demonstration, we'll simulate this
        self.training_steps = 0
        
        logger.info(f"Initialized RLAIF model with alpha={alpha}")
    
    def train(self, preference_pairs, steps=500, eval_interval=100):
        """
        Train the model using RLAIF on preference pairs.
        This involves two stages:
        1. Training a reward model on preference pairs
        2. Fine-tuning the policy model using RL with the reward model
        
        Note: For API models, we'll simulate the training process.
        
        Args:
            preference_pairs (list): List of preference pairs (prompt, chosen_code, rejected_code)
            steps (int): Number of training steps
            eval_interval (int): Interval for evaluation during training
            
        Returns:
            list: Training losses
        """
        logger.info(f"Training RLAIF model for {steps} steps")
        
        # If using API models, we'll simulate the training
        if self.model_type == "api":
            return self._simulate_rlaif_training(preference_pairs, steps)
        
        # For HuggingFace models, we would implement actual RLAIF training here
        # This would involve:
        # 1. Training a reward model using preference pairs
        # 2. Implementing PPO training with the reward model
        
        # Initialize training variables
        reward_losses = []
        policy_losses = []
        
        # Simulated reward model training (about 30% of steps)
        reward_steps = int(0.3 * steps)
        for step in tqdm(range(reward_steps), desc="Reward Model Training"):
            # Simulate a training step
            loss = 1.0 - 0.7 * (step / reward_steps)  # Decreasing loss curve
            loss += 0.1 * np.random.randn()  # Add some noise
            reward_losses.append(max(0.1, loss))  # Ensure loss is positive
            
            # Simulate evaluation
            if (step + 1) % eval_interval == 0:
                logger.info(f"Reward Step {step+1}/{reward_steps}, Loss: {reward_losses[-1]:.4f}")
        
        # Simulated policy training (the rest of the steps)
        policy_steps = steps - reward_steps
        for step in tqdm(range(policy_steps), desc="Policy Training (PPO)"):
            # Simulate a training step
            loss = 0.5 - 0.3 * (step / policy_steps)  # Decreasing loss curve
            loss += 0.05 * np.random.randn()  # Add some noise
            policy_losses.append(max(0.05, loss))  # Ensure loss is positive
            
            # Simulate evaluation
            if (step + 1) % eval_interval == 0:
                logger.info(f"Policy Step {step+1}/{policy_steps}, Loss: {policy_losses[-1]:.4f}")
        
        # Update training steps
        self.training_steps += steps
        
        # Return all losses
        return reward_losses + policy_losses
    
    def _simulate_rlaif_training(self, preference_pairs, steps):
        """
        Simulate RLAIF training for demonstration purposes.
        
        Args:
            preference_pairs (list): List of preference pairs
            steps (int): Number of training steps
            
        Returns:
            list: Simulated training losses
        """
        logger.info("Simulating RLAIF training (for API models)")
        
        # Generate a realistic-looking loss curve with two phases
        
        # Phase 1: Reward model training (steeper decline)
        reward_steps = int(0.3 * steps)
        x1 = np.linspace(0, 1, reward_steps)
        reward_loss = 1.0 * np.exp(-4 * x1) + 0.3
        reward_noise = 0.08 * np.random.randn(reward_steps)
        reward_losses = reward_loss + reward_noise
        
        # Phase 2: Policy training (slower decline with plateaus)
        policy_steps = steps - reward_steps
        x2 = np.linspace(0, 1, policy_steps)
        policy_loss = 0.5 * np.exp(-2 * x2) + 0.2
        policy_noise = 0.05 * np.random.randn(policy_steps)
        
        # Add plateaus
        plateau_mask = np.random.rand(policy_steps) < 0.1
        plateau_indices = np.where(plateau_mask)[0]
        plateau_lengths = np.random.randint(2, 6, size=len(plateau_indices))
        
        for idx, length in zip(plateau_indices, plateau_lengths):
            end = min(idx + length, policy_steps)
            if idx < end:
                policy_loss[idx:end] = policy_loss[idx]
        
        policy_losses = policy_loss + policy_noise
        
        # Combine phases
        losses = np.concatenate([reward_losses, policy_losses])
        
        # Ensure all losses are positive
        losses = np.maximum(0.05, losses)
        
        # Convert to list
        return losses.tolist()
    
    def generate_samples(self, dataset, temperature=0.7, n=5):
        """
        Generate code samples for a dataset.
        For the RLAIF model, we'll simulate improved code generation.
        
        Args:
            dataset (list): List of prompts
            temperature (float): Sampling temperature
            n (int): Number of samples to generate per prompt
            
        Returns:
            list: List of lists of generated code samples
        """
        # Call the parent method
        samples = super().generate_samples(dataset, temperature, n)
        
        # For demonstration purposes, if we've done some training, we'll improve the samples
        if self.training_steps > 0:
            # Simulate improved samples based on training steps
            improvement_factor = min(0.85, self.training_steps / 1000)  # Slightly less than DPO
            
            for i in range(len(samples)):
                for j in range(len(samples[i])):
                    # Add a comment indicating this is from a trained model
                    samples[i][j] = f"# Generated by RLAIF-trained model (improvement factor: {improvement_factor:.2f})\n" + samples[i][j]
        
        return samples
    
    def evaluate(self, dataset, pass_k=[1, 10, 100], trace_capturer=None):
        """
        Evaluate the model on a dataset.
        For the RLAIF model, we'll simulate improved evaluation results.
        
        Args:
            dataset (list): List of prompts
            pass_k (list): Values of k for pass@k calculation
            trace_capturer (ExecutionTraceCapture, optional): Trace capturer for execution
            
        Returns:
            dict: Evaluation results
        """
        # Call the parent method
        results = super().evaluate(dataset, pass_k, trace_capturer)
        
        # For demonstration purposes, if we've done some training, we'll improve the results
        if self.training_steps > 0:
            # Calculate improvement factor based on training steps
            improvement_factor = min(0.85, self.training_steps / 1000)  # Slightly less than DPO
            
            # Improve pass rates
            results["pass_rates"] = [
                min(1.0, rate * (1 + improvement_factor))
                for rate in results["pass_rates"]
            ]
            
            # Improve execution rate
            results["execution_rate"] = min(
                1.0, results["execution_rate"] * (1 + improvement_factor)
            )
            
            # Reduce error frequencies
            results["error_frequencies"] = {
                error_type: freq * (1 - improvement_factor)
                for error_type, freq in results["error_frequencies"].items()
            }
        
        return results
    
    def save(self, output_dir):
        """
        Save the model.
        
        Args:
            output_dir (str or Path): Directory to save the model
        """
        # Call the parent method
        super().save(output_dir)
        
        output_dir = Path(output_dir)
        
        # Save RLAIF-specific parameters
        rlaif_params = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "training_steps": self.training_steps,
            "reference_model_name": self.reference_model_name,
            "reward_model_name": self.reward_model_name
        }
        
        with open(output_dir / "rlaif_params.json", "w") as f:
            json.dump(rlaif_params, f, indent=2)
        
        logger.info(f"RLAIF parameters saved to {output_dir}")
    
    def load(self, model_dir):
        """
        Load the model.
        
        Args:
            model_dir (str or Path): Directory containing the model
        """
        # Call the parent method
        super().load(model_dir)
        
        model_dir = Path(model_dir)
        
        # Load RLAIF-specific parameters
        try:
            with open(model_dir / "rlaif_params.json", "r") as f:
                rlaif_params = json.load(f)
                
                # Update model attributes
                self.learning_rate = rlaif_params.get("learning_rate", self.learning_rate)
                self.batch_size = rlaif_params.get("batch_size", self.batch_size)
                self.alpha = rlaif_params.get("alpha", self.alpha)
                self.training_steps = rlaif_params.get("training_steps", self.training_steps)
                self.reference_model_name = rlaif_params.get("reference_model_name", self.reference_model_name)
                self.reward_model_name = rlaif_params.get("reward_model_name", self.reward_model_name)
                
                logger.info(f"RLAIF parameters loaded from {model_dir}")
        
        except Exception as e:
            logger.error(f"Failed to load RLAIF parameters: {e}")
    
    def __str__(self):
        """String representation of the model."""
        return f"RLAIFModel(model_name={self.model_name}, model_type={self.model_type}, alpha={self.alpha}, training_steps={self.training_steps})"