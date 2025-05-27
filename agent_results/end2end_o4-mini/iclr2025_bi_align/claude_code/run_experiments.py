#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running UDRA experiments.
This script executes both baseline and UDRA algorithms on simulated environments
and records the results for later analysis.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from environments.resource_allocation import ResourceAllocationEnv
from environments.safety_critical import SafetyCriticalEnv
from models.baseline import BaselineRLHF
from models.udra import UDRA
from utils.metrics import compute_alignment_error, compute_task_efficiency, compute_trust_calibration
from utils.simulated_human import SimulatedHuman

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./results/log.txt', mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run UDRA experiments')
    parser.add_argument('--num_episodes', type=int, default=500, 
                        help='Number of episodes to run')
    parser.add_argument('--env_type', type=str, default='both', 
                        choices=['resource', 'safety', 'both'],
                        help='Which environment to use')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--lambda_val', type=float, default=0.5, 
                        help='Lambda value for balancing task performance and alignment')
    return parser.parse_args()

def run_experiment(env, agent, simulated_human, num_episodes):
    """Run a single experiment with the given environment, agent and simulated human."""
    results = {
        'episode_rewards': [],
        'alignment_errors': [],
        'trust_calibration': [],
        'q_uncertainties': [],
        'actions_taken': [],
        'human_corrections': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_alignment_errors = []
        episode_uncertainties = []
        episode_actions = []
        episode_corrections = []
        
        while not done:
            # Agent selects action and estimates uncertainty
            action, q_values, uncertainty = agent.select_action_with_uncertainty(state)
            episode_actions.append(action)
            episode_uncertainties.append(uncertainty)
            
            # Simulated human may provide correction
            human_correction = simulated_human.provide_feedback(state, action, q_values, uncertainty)
            episode_corrections.append(human_correction)
            
            # Update agent based on human feedback if provided
            if human_correction is not None:
                agent.update_from_feedback(state, human_correction)
                # Execute corrected action
                next_state, reward, done, info = env.step(human_correction)
            else:
                # Execute agent's action
                next_state, reward, done, info = env.step(action)
            
            # Update agent based on environment feedback
            agent.update_from_environment(state, action, reward, next_state, done)
            
            # Compute alignment error
            if human_correction is not None:
                alignment_error = compute_alignment_error(action, human_correction)
                episode_alignment_errors.append(alignment_error)
            
            state = next_state
            episode_reward += reward
            
        # Store episode results
        results['episode_rewards'].append(episode_reward)
        if episode_alignment_errors:
            results['alignment_errors'].append(np.mean(episode_alignment_errors))
        else:
            results['alignment_errors'].append(0.0)  # No corrections this episode
        
        # Compute trust calibration at episode level (correlation between uncertainty and corrections)
        if episode % 10 == 0:  # Compute less frequently to accumulate more data points
            trust_cal = compute_trust_calibration(
                episode_uncertainties, 
                [1 if corr is not None else 0 for corr in episode_corrections]
            )
            results['trust_calibration'].append(trust_cal)
        
        # Store detailed data
        results['q_uncertainties'].extend(episode_uncertainties)
        results['actions_taken'].extend(episode_actions)
        results['human_corrections'].extend(episode_corrections)
        
        # Log progress
        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                         f"Alignment Error: {results['alignment_errors'][-1]:.4f}")
                         
    return results

def main():
    args = parse_args()
    start_time = time.time()
    
    logger.info("Starting UDRA experiments")
    logger.info(f"Arguments: {args}")
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    
    # Initialize results container
    all_results = {}
    
    # Define environment parameters
    env_params = {
        'resource': {
            'state_dim': 10,
            'action_dim': 5,
            'feature_dim': 8  # Dimension of feature map φ(s,a)
        },
        'safety': {
            'state_dim': 12,
            'action_dim': 4,
            'feature_dim': 10  # Dimension of feature map φ(s,a)
        }
    }
    
    # Define which environments to run
    envs_to_run = []
    if args.env_type in ['resource', 'both']:
        envs_to_run.append('resource')
    if args.env_type in ['safety', 'both']:
        envs_to_run.append('safety')
    
    # Run experiments for each environment
    for env_name in envs_to_run:
        logger.info(f"\n{'='*50}\nRunning experiments on {env_name} environment\n{'='*50}")
        
        # Get environment parameters
        params = env_params[env_name]
        
        # Initialize environment
        if env_name == 'resource':
            env = ResourceAllocationEnv(
                state_dim=params['state_dim'],
                action_dim=params['action_dim'],
                feature_dim=params['feature_dim']
            )
        else:  # safety
            env = SafetyCriticalEnv(
                state_dim=params['state_dim'],
                action_dim=params['action_dim'],
                feature_dim=params['feature_dim']
            )
            
        # Generate true user preference vector (hidden from agents)
        true_preference = np.random.randn(params['feature_dim'])
        true_preference = true_preference / np.linalg.norm(true_preference)  # Normalize
        
        # Initialize simulated human
        simulated_human = SimulatedHuman(
            true_preference=true_preference,
            feature_dim=params['feature_dim'],
            correction_threshold=0.2,  # Threshold for when to provide corrections
            noise_level=0.05  # Add some noise to human feedback
        )
        
        # Run baseline algorithm (standard RLHF with static alignment)
        logger.info(f"\n{'-'*40}\nRunning baseline RLHF algorithm\n{'-'*40}")
        baseline_agent = BaselineRLHF(
            state_dim=params['state_dim'],
            action_dim=params['action_dim'],
            feature_dim=params['feature_dim'],
            learning_rate=0.001,
            gamma=0.99
        )
        
        baseline_results = run_experiment(
            env=env,
            agent=baseline_agent,
            simulated_human=simulated_human,
            num_episodes=args.num_episodes
        )
        
        # Run UDRA algorithm
        logger.info(f"\n{'-'*40}\nRunning UDRA algorithm\n{'-'*40}")
        udra_agent = UDRA(
            state_dim=params['state_dim'],
            action_dim=params['action_dim'],
            feature_dim=params['feature_dim'],
            learning_rate=0.001,
            gamma=0.99,
            lambda_val=args.lambda_val,
            ensemble_size=5  # Number of models in ensemble for uncertainty estimation
        )
        
        udra_results = run_experiment(
            env=env,
            agent=udra_agent,
            simulated_human=simulated_human,
            num_episodes=args.num_episodes
        )
        
        # Store results
        all_results[env_name] = {
            'baseline': baseline_results,
            'udra': udra_results,
            'env_params': params,
            'true_preference': true_preference.tolist(),
        }
    
    # Save results
    results_file = './claude_code/results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"All experiments completed. Total time: {(time.time() - start_time)/60:.2f} minutes")
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()