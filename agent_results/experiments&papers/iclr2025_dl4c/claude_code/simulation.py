"""
Simulation module for the adaptive code assistant experiment.
This module simulates interactions between developers and code assistants.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from utils import set_seed, save_json, load_json, ensure_dir, timer
from data import DeveloperProfile, generate_developer_profiles, CodeTaskDataset
from models import get_model, CodeAssistantModel
from evaluation import CodeEvaluator, ExperimentEvaluator

logger = logging.getLogger("adaptive_code_assistant.simulation")

class CodingSimulation:
    """
    Class to simulate coding interactions between a developer and a code assistant.
    """
    
    def __init__(
        self,
        developer_profile: DeveloperProfile,
        code_assistant: CodeAssistantModel,
        max_iterations: int = 5,
        output_dir: str = "results"
    ):
        """
        Initialize the coding simulation.
        
        Args:
            developer_profile: Developer profile for the simulation
            code_assistant: Code assistant model for the simulation
            max_iterations: Maximum number of iterations for each task
            output_dir: Directory to save simulation results
        """
        self.developer_profile = developer_profile
        self.code_assistant = code_assistant
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        self.evaluator = CodeEvaluator()
        
        # Set up rule-based model with developer preferences if applicable
        if hasattr(self.code_assistant, "set_developer_preferences"):
            self.code_assistant.set_developer_preferences(developer_profile)
        
        logger.info(f"Initialized coding simulation for developer {developer_profile.dev_id} with model {code_assistant.model_name}")
    
    @timer
    def run_task(self, task: Dict[str, str], task_type: str = None) -> Dict[str, Any]:
        """
        Run a coding task simulation.
        
        Args:
            task: Coding task
            task_type: Type of coding task for task-specific adaptation
            
        Returns:
            Dictionary containing simulation results
        """
        task_id = task["task_id"]
        prompt = task["prompt"]
        solution = task["canonical_solution"]
        test_cases = task["test_cases"]
        
        logger.info(f"Starting task {task_id}")
        
        # Initialize session data
        session_data = {
            "task_id": task_id,
            "developer_id": self.developer_profile.dev_id,
            "model_name": self.code_assistant.model_name,
            "iterations": 0,
            "completion_time": 0.0,
            "feedback_history": [],
            "code_history": [],
            "satisfaction_history": []
        }
        
        current_code = ""
        satisfaction = 0.0
        
        # Simulate iterations
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration+1}/{self.max_iterations}")
            
            # Generate code completion
            start_time = time.time()
            completion = self.code_assistant.complete_code(prompt, task_type=task_type)
            completion_time = time.time() - start_time
            
            # Update current code
            current_code = completion
            
            # Add to history
            session_data["code_history"].append(current_code)
            
            # Evaluate code
            correctness_results = self.evaluator.evaluate_functional_correctness(current_code, test_cases)
            style_results = self.evaluator.evaluate_code_style(current_code, self.developer_profile.formatting_preferences)
            
            # Generate developer feedback
            feedback = self.developer_profile.generate_feedback(current_code)
            session_data["feedback_history"].append(feedback)
            
            # Calculate satisfaction
            correctness_score = 1.0 if correctness_results.get("passed", False) else 0.0
            style_score = style_results.get("overall_style_score", 0.0)
            
            satisfaction = 0.6 * correctness_score + 0.4 * style_score
            session_data["satisfaction_history"].append(satisfaction)
            
            # Update model based on feedback
            self.code_assistant.update(prompt, feedback, task_type=task_type)
            
            # Update iteration count and completion time
            session_data["iterations"] += 1
            session_data["completion_time"] += completion_time
            
            # Break if satisfaction is high enough or code is correct
            if satisfaction > 0.8 or correctness_results.get("passed", False):
                logger.info(f"Stopping iterations early due to high satisfaction: {satisfaction:.2f}")
                break
        
        # Save final results
        session_data["final_code"] = current_code
        session_data["final_satisfaction"] = satisfaction
        session_data["test_cases"] = test_cases
        
        # Save session data
        session_file = os.path.join(self.output_dir, f"session_{task_id}_{self.developer_profile.dev_id}_{self.code_assistant.model_name}.json")
        save_json(session_data, session_file)
        
        logger.info(f"Task {task_id} completed with final satisfaction: {satisfaction:.2f}")
        
        return session_data

class ExperimentRunner:
    """
    Class to run the entire experiment.
    """
    
    def __init__(
        self,
        num_developers: int = 5,
        num_tasks: int = 10,
        max_iterations: int = 5,
        output_dir: str = "results",
        use_small_models: bool = True
    ):
        """
        Initialize the experiment runner.
        
        Args:
            num_developers: Number of developer profiles to generate
            num_tasks: Number of tasks to run for each model
            max_iterations: Maximum number of iterations for each task
            output_dir: Directory to save experiment results
            use_small_models: Whether to use small models for faster experimentation
        """
        self.num_developers = num_developers
        self.num_tasks = num_tasks
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.use_small_models = use_small_models
        ensure_dir(output_dir)
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Generate developer profiles
        self.developer_profiles = generate_developer_profiles(
            num_developers,
            os.path.join(self.output_dir, "developer_profiles.json")
        )
        
        # Load code tasks
        self.code_tasks = CodeTaskDataset()
        self.tasks = self.code_tasks.get_n_tasks(num_tasks)
        
        # Define models to evaluate
        self.model_names = [
            "static",            # Baseline 1: Static LLM
            "fine_tuned",        # Baseline 2: Fine-tuned LLM
            "rule_based",        # Baseline 3: Rule-based Personalization
            "online",            # Proposed 1: Online Learning
            "maml",              # Proposed 2: MAML-based Adaptation
            "hybrid"             # Proposed 3: Hybrid Approach
        ]
        
        logger.info(f"Initialized experiment with {num_developers} developers and {num_tasks} tasks")
    
    @timer
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the entire experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        experiment_data = {
            "config": {
                "num_developers": self.num_developers,
                "num_tasks": self.num_tasks,
                "max_iterations": self.max_iterations,
                "use_small_models": self.use_small_models
            },
            "sessions": {}
        }
        
        # Run simulations for each model and developer
        for model_name in self.model_names:
            logger.info(f"Running simulations for model: {model_name}")
            
            for developer_profile in self.developer_profiles:
                logger.info(f"Simulating developer: {developer_profile.dev_id}")
                
                # Initialize model
                code_assistant = get_model(model_name, self.use_small_models)
                
                # Initialize simulation
                simulation = CodingSimulation(
                    developer_profile,
                    code_assistant,
                    max_iterations=self.max_iterations,
                    output_dir=self.output_dir
                )
                
                # Run tasks
                for task_idx, task in enumerate(self.tasks):
                    if task_idx >= self.num_tasks:
                        break
                    
                    logger.info(f"Running task {task_idx+1}/{self.num_tasks}: {task['task_id']}")
                    
                    # Determine task type based on prompt
                    task_type = self._extract_task_type(task["prompt"])
                    
                    # Run task simulation
                    session_data = simulation.run_task(task, task_type)
                    
                    # Generate session ID
                    session_id = f"{model_name}_{developer_profile.dev_id}_{task['task_id']}"
                    
                    # Store session data
                    experiment_data["sessions"][session_id] = session_data
        
        # Save experiment data
        save_json(experiment_data, os.path.join(self.output_dir, "experiment_data.json"))
        
        logger.info("Experiment completed")
        
        return experiment_data
    
    def _extract_task_type(self, prompt: str) -> str:
        """
        Extract the task type from a prompt.
        
        Args:
            prompt: Task prompt
            
        Returns:
            Task type
        """
        # Simple keyword-based extraction
        if "sort" in prompt.lower() or "order" in prompt.lower():
            return "sorting"
        elif "search" in prompt.lower() or "find" in prompt.lower():
            return "searching"
        elif "calculate" in prompt.lower() or "compute" in prompt.lower():
            return "calculation"
        elif "convert" in prompt.lower() or "transform" in prompt.lower():
            return "conversion"
        elif "check" in prompt.lower() or "validate" in prompt.lower():
            return "validation"
        else:
            return "general"
    
    def evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate the experiment results.
        
        Returns:
            Evaluation results
        """
        # Load experiment data
        experiment_data = load_json(os.path.join(self.output_dir, "experiment_data.json"))
        
        # Load developer profiles
        developer_profiles = load_json(os.path.join(self.output_dir, "developer_profiles.json"))
        
        # Initialize evaluator
        evaluator = ExperimentEvaluator(self.output_dir)
        
        # Evaluate experiment
        evaluation_results = evaluator.evaluate_experiment(experiment_data, developer_profiles)
        
        # Visualize results
        evaluator.visualize_results()
        
        # Generate markdown report
        markdown_report = evaluator.generate_results_markdown()
        
        # Save markdown report
        with open(os.path.join(self.output_dir, "results.md"), "w") as f:
            f.write(markdown_report)
        
        logger.info("Evaluation completed")
        
        return evaluation_results

def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Run adaptive code assistant experiment")
    parser.add_argument("--developers", type=int, default=3, help="Number of developer profiles")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks per model")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum iterations per task")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--small-models", action="store_true", help="Use small models for faster experimentation")
    
    args = parser.parse_args()
    
    # Configure logging to file
    output_dir = os.path.join("claude_exp2", "iclr2025_dl4c", "claude_code", args.output)
    ensure_dir(output_dir)
    
    log_file = os.path.join(output_dir, "log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Print start message
    logger.info("Starting adaptive code assistant experiment")
    logger.info(f"Configuration: {args}")
    
    # Run experiment
    experiment_runner = ExperimentRunner(
        num_developers=args.developers,
        num_tasks=args.tasks,
        max_iterations=args.iterations,
        output_dir=output_dir,
        use_small_models=args.small_models
    )
    
    # Run experiment
    experiment_data = experiment_runner.run_experiment()
    
    # Evaluate results
    evaluation_results = experiment_runner.evaluate_results()
    
    # Print completion message
    logger.info("Experiment completed successfully")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()