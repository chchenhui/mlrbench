#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for UDRA experiments.

This module contains functions for computing key metrics like alignment error,
task efficiency, and trust calibration.
"""

import numpy as np
from scipy import stats

def compute_alignment_error(agent_action, human_action):
    """
    Compute alignment error between agent action and human correction.
    
    For discrete actions, this is 0 for matching actions and 1 for different actions.
    For continuous actions, this is the Euclidean distance between actions.
    
    Args:
        agent_action: Action selected by the agent
        human_action: Action selected by the human
        
    Returns:
        error (float): Alignment error between actions
    """
    # Check action type to handle discrete vs continuous actions
    if isinstance(agent_action, (int, np.integer)) and isinstance(human_action, (int, np.integer)):
        # Discrete actions: 0 if matching, 1 if different
        return 0.0 if agent_action == human_action else 1.0
    else:
        # Continuous actions or mixed types: use Euclidean distance
        agent_action_arr = np.array(agent_action) if not isinstance(agent_action, np.ndarray) else agent_action
        human_action_arr = np.array(human_action) if not isinstance(human_action, np.ndarray) else human_action
        
        # Normalize to unit vectors before computing distance
        agent_norm = np.linalg.norm(agent_action_arr)
        human_norm = np.linalg.norm(human_action_arr)
        
        if agent_norm > 0:
            agent_action_arr = agent_action_arr / agent_norm
        if human_norm > 0:
            human_action_arr = human_action_arr / human_norm
            
        return np.linalg.norm(agent_action_arr - human_action_arr)

def compute_task_efficiency(rewards, window_size=10):
    """
    Compute task efficiency based on rolling average of rewards.
    
    Args:
        rewards (list): List of episode rewards
        window_size (int): Window size for computing moving average
        
    Returns:
        efficiency (list): Moving average of rewards
    """
    if len(rewards) < window_size:
        return np.mean(rewards)
    
    # Compute moving average
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def compute_trust_calibration(uncertainties, corrections):
    """
    Compute trust calibration as Spearman's rank correlation between
    uncertainty estimates and binary correction indicators.
    
    A high positive correlation indicates good calibration:
    the agent is uncertain when corrections are needed.
    
    Args:
        uncertainties (list): List of uncertainty estimates
        corrections (list): Binary indicators of corrections (1 if corrected, 0 if not)
        
    Returns:
        correlation (float): Spearman's rank correlation coefficient
    """
    # Ensure arrays are the same length
    assert len(uncertainties) == len(corrections), "Uncertainty and correction arrays must have the same length"
    
    # Handle empty arrays or single-valued arrays
    if len(uncertainties) <= 1:
        return np.nan
    
    # Check if all values are the same (can't compute correlation)
    if len(set(uncertainties)) <= 1 or len(set(corrections)) <= 1:
        return np.nan
    
    # Compute Spearman's rank correlation
    correlation, p_value = stats.spearmanr(uncertainties, corrections)
    
    return correlation