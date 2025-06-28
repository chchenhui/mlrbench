#!/usr/bin/env python3
"""
Main script for running the Dynamic Human-AI Co-Adaptation experiment.
This script orchestrates the entire experimental workflow.
"""

import os
import json
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Import local modules
from environment import RecommendationEnvironment
from models import DynamicAlignmentAgent, StaticRLHFAgent, DirectRLAIFAgent
from utils import plot_metrics, save_results, generate_user_preferences
from explanation import ExplanationGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../results/log.txt"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Configuration
CONFIG = {
    "seed": 42,
    "n_users": 100,
    "n_items": 1000,
    "n_features": 20,
    "n_episodes": 100,
    "preference_shift_interval": 20,
    "preference_shift_magnitude": 0.3,
    "learning_rate": 0.001,
    "discount_factor": 0.95,
    "imitation_weight": 0.3,
    "explanation_threshold": 0.1,
    "batch_size": 64,
    "eval_interval": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def run_experiment():
    """Run the main experiment comparing different alignment approaches."""
    logger.info("Starting the Dynamic Human-AI Co-Adaptation experiment")
    logger.info(f"Using device: {CONFIG['device']}")
    
    # Set random seeds for reproducibility
    set_seeds(CONFIG["seed"])
    
    # Create output directories if they don't exist
    os.makedirs("../results", exist_ok=True)
    
    # Save configuration
    with open("../results/config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    # Generate initial user preferences
    logger.info("Generating user preferences")
    user_preferences = generate_user_preferences(
        n_users=CONFIG["n_users"],
        n_features=CONFIG["n_features"]
    )
    
    # Create environment
    logger.info("Creating recommendation environment")
    env = RecommendationEnvironment(
        n_users=CONFIG["n_users"],
        n_items=CONFIG["n_items"],
        n_features=CONFIG["n_features"],
        user_preferences=user_preferences,
        preference_shift_interval=CONFIG["preference_shift_interval"],
        preference_shift_magnitude=CONFIG["preference_shift_magnitude"]
    )
    
    # Create agents
    logger.info("Creating agents")
    agents = {
        "dynamic_alignment": DynamicAlignmentAgent(
            n_features=CONFIG["n_features"],
            learning_rate=CONFIG["learning_rate"],
            discount_factor=CONFIG["discount_factor"],
            imitation_weight=CONFIG["imitation_weight"],
            device=CONFIG["device"]
        ),
        "static_rlhf": StaticRLHFAgent(
            n_features=CONFIG["n_features"],
            learning_rate=CONFIG["learning_rate"],
            discount_factor=CONFIG["discount_factor"],
            device=CONFIG["device"]
        ),
        "direct_rlaif": DirectRLAIFAgent(
            n_features=CONFIG["n_features"],
            learning_rate=CONFIG["learning_rate"],
            discount_factor=CONFIG["discount_factor"],
            device=CONFIG["device"]
        )
    }
    
    # Create explanation generator
    explanation_generator = ExplanationGenerator(threshold=CONFIG["explanation_threshold"])
    
    # Initialize results dictionary
    results = {agent_name: {
        "rewards": [],
        "alignment_scores": [],
        "trust_scores": [],
        "adaptability_scores": []
    } for agent_name in agents.keys()}
    
    # Run the experiment
    logger.info("Starting training and evaluation")
    
    for episode in tqdm(range(CONFIG["n_episodes"])):
        logger.info(f"Episode {episode+1}/{CONFIG['n_episodes']}")
        
        # Update environment if it's time to shift preferences
        if episode > 0 and episode % CONFIG["preference_shift_interval"] == 0:
            logger.info(f"Shifting user preferences at episode {episode}")
            env.shift_preferences()
        
        # Train and evaluate each agent
        for agent_name, agent in agents.items():
            logger.info(f"Training agent: {agent_name}")
            
            # Reset environment for this agent
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Interaction loop
            while not done:
                # Agent selects action
                action = agent.select_action(state)
                
                # Environment provides next state and reward
                next_state, reward, done, info = env.step(action)
                
                # Generate explanations (only for dynamic alignment agent)
                if agent_name == "dynamic_alignment":
                    explanations = explanation_generator.generate(state, action, agent)
                    agent.update(state, action, reward, next_state, done, explanations)
                else:
                    agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            # Record results
            results[agent_name]["rewards"].append(episode_reward)
            
            # Evaluate alignment, trust, and adaptability every eval_interval episodes
            if episode % CONFIG["eval_interval"] == 0:
                alignment_score = env.evaluate_alignment(agent)
                trust_score = env.evaluate_trust(agent)
                adaptability_score = env.evaluate_adaptability(agent, episode)
                
                results[agent_name]["alignment_scores"].append(alignment_score)
                results[agent_name]["trust_scores"].append(trust_score)
                results[agent_name]["adaptability_scores"].append(adaptability_score)
                
                logger.info(f"{agent_name} - Episode: {episode}, "
                          f"Reward: {episode_reward:.3f}, "
                          f"Alignment: {alignment_score:.3f}, "
                          f"Trust: {trust_score:.3f}, "
                          f"Adaptability: {adaptability_score:.3f}")
    
    # Save results
    logger.info("Saving results and generating plots")
    save_results(results, "../results/experiment_results.json")
    
    # Generate plots
    plot_metrics(results, "../results/reward_curve.png", "Reward", range(CONFIG["n_episodes"]))
    plot_metrics(results, "../results/alignment_curve.png", "Alignment Score", 
                 range(0, CONFIG["n_episodes"], CONFIG["eval_interval"]))
    plot_metrics(results, "../results/trust_curve.png", "Trust Score", 
                 range(0, CONFIG["n_episodes"], CONFIG["eval_interval"]))
    plot_metrics(results, "../results/adaptability_curve.png", "Adaptability Score", 
                 range(0, CONFIG["n_episodes"], CONFIG["eval_interval"]))
    
    # Generate comparison table
    generate_comparison_table(results, "../results/comparison_table.csv")
    
    logger.info("Experiment completed successfully")
    return results

def generate_comparison_table(results, output_file):
    """Generate a comparison table of the final performance metrics."""
    final_results = {}
    
    for agent_name, metrics in results.items():
        final_results[agent_name] = {
            "Average Reward": np.mean(metrics["rewards"]),
            "Final Reward": metrics["rewards"][-1],
            "Average Alignment": np.mean(metrics["alignment_scores"]),
            "Final Alignment": metrics["alignment_scores"][-1],
            "Average Trust": np.mean(metrics["trust_scores"]),
            "Final Trust": metrics["trust_scores"][-1],
            "Average Adaptability": np.mean(metrics["adaptability_scores"]),
            "Final Adaptability": metrics["adaptability_scores"][-1]
        }
    
    df = pd.DataFrame(final_results).T
    df.to_csv(output_file)
    logger.info(f"Comparison table saved to {output_file}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Experiment started at: {start_time}")
    
    try:
        results = run_experiment()
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}", exc_info=True)
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Experiment ended at: {end_time}")
    logger.info(f"Total duration: {duration}")