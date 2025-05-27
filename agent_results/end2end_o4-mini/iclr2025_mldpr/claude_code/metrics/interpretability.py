"""
Interpretability metrics module

This module implements metrics for evaluating the interpretability of machine learning models,
including feature attribution stability, decision boundary clarity, and explainability.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import warnings


def attribute_stability(
    attribution_func: Callable,
    x: np.ndarray,
    perturbation_func: Callable,
    num_perturbations: int = 10
) -> Dict[str, float]:
    """
    Calculate stability of feature attributions under perturbations.
    
    Args:
        attribution_func: Function that returns feature attributions for an input
        x: Input samples
        perturbation_func: Function that perturbs an input
        num_perturbations: Number of perturbations to generate
        
    Returns:
        dict: Dictionary of attribution stability metrics
    """
    try:
        num_samples = min(len(x), 100)  # Limit to 100 samples for efficiency
        x_subset = x[:num_samples]
        
        stabilities = []
        
        for sample in x_subset:
            # Get original attribution
            orig_attribution = attribution_func(sample.reshape(1, -1)).flatten()
            
            # Generate perturbed samples and their attributions
            perturbed_attributions = []
            
            for _ in range(num_perturbations):
                perturbed_sample = perturbation_func(sample.reshape(1, -1)).flatten()
                attribution = attribution_func(perturbed_sample.reshape(1, -1)).flatten()
                perturbed_attributions.append(attribution)
            
            # Calculate stability metrics
            attribution_diffs = []
            
            for perturbed_attr in perturbed_attributions:
                # Handle potential shape mismatches
                min_len = min(len(orig_attribution), len(perturbed_attr))
                orig_attr_trimmed = orig_attribution[:min_len]
                perturbed_attr_trimmed = perturbed_attr[:min_len]
                
                # Calculate L1 difference
                l1_diff = np.sum(np.abs(orig_attr_trimmed - perturbed_attr_trimmed))
                
                # Normalize by the L1 norm of the original attribution
                orig_norm = np.sum(np.abs(orig_attr_trimmed))
                if orig_norm > 0:
                    norm_diff = l1_diff / orig_norm
                else:
                    norm_diff = 0.0
                
                attribution_diffs.append(norm_diff)
            
            # Calculate stability as 1 - average normalized difference
            stability = 1.0 - np.mean(attribution_diffs)
            stabilities.append(stability)
        
        # Calculate overall metrics
        mean_stability = np.mean(stabilities)
        std_stability = np.std(stabilities)
        
        return {
            'mean_stability': float(mean_stability),
            'std_stability': float(std_stability)
        }
    except Exception as e:
        warnings.warn(f"Failed to calculate attribution stability: {str(e)}")
        return {
            'mean_stability': float('nan'),
            'std_stability': float('nan')
        }


def feature_importance_concentration(
    feature_importances: np.ndarray,
    threshold: float = 0.8
) -> Dict[str, float]:
    """
    Calculate metrics related to the concentration of feature importances.
    
    Args:
        feature_importances: Array of feature importance values
        threshold: Threshold for cumulative importance
        
    Returns:
        dict: Dictionary of concentration metrics
    """
    try:
        # Normalize feature importances if not already
        if np.sum(feature_importances) != 1.0:
            feature_importances = feature_importances / np.sum(feature_importances)
        
        # Sort importances in descending order
        sorted_importances = np.sort(feature_importances)[::-1]
        
        # Calculate cumulative importance
        cumulative_importance = np.cumsum(sorted_importances)
        
        # Find the number of features needed to reach the threshold
        features_for_threshold = np.argmax(cumulative_importance >= threshold) + 1
        
        # Calculate Gini coefficient as a measure of importance concentration
        n = len(feature_importances)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_importances)) / (n * np.sum(sorted_importances)) - (n + 1) / n
        
        return {
            'features_for_threshold': int(features_for_threshold),
            'feature_concentration_ratio': float(features_for_threshold / len(feature_importances)),
            'gini_coefficient': float(gini)
        }
    except Exception as e:
        warnings.warn(f"Failed to calculate feature importance concentration: {str(e)}")
        return {
            'features_for_threshold': int(len(feature_importances)),
            'feature_concentration_ratio': 1.0,
            'gini_coefficient': float('nan')
        }


def decision_boundary_clarity(
    decision_function: Callable,
    x: np.ndarray,
    num_samples: int = 1000,
    distance_metric: str = 'euclidean'
) -> Dict[str, float]:
    """
    Estimate the clarity of decision boundaries.
    
    Args:
        decision_function: Function that returns decision scores
        x: Input data
        num_samples: Number of samples to use
        distance_metric: Distance metric to use
        
    Returns:
        dict: Dictionary of boundary clarity metrics
    """
    try:
        # Limit the number of samples for efficiency
        x_subset = x[:min(len(x), num_samples)]
        
        # Get decision scores
        scores = decision_function(x_subset)
        
        # Calculate score gradients (approximate using small perturbations)
        epsilon = 1e-5
        gradients = []
        
        for i, sample in enumerate(x_subset):
            perturbed_samples = []
            
            # Perturb each feature slightly
            for j in range(sample.shape[0]):
                perturbed = sample.copy()
                perturbed[j] += epsilon
                perturbed_samples.append(perturbed)
            
            # Get scores for perturbed samples
            perturbed_scores = decision_function(np.array(perturbed_samples))
            
            # Calculate gradient
            gradient = (perturbed_scores - scores[i]) / epsilon
            gradients.append(gradient)
        
        # Calculate gradient magnitude
        gradient_magnitudes = np.linalg.norm(gradients, axis=1)
        
        # Calculate metrics
        mean_gradient = np.mean(gradient_magnitudes)
        std_gradient = np.std(gradient_magnitudes)
        
        # Higher gradient magnitude indicates sharper decision boundaries
        return {
            'mean_gradient_magnitude': float(mean_gradient),
            'std_gradient_magnitude': float(std_gradient)
        }
    except Exception as e:
        warnings.warn(f"Failed to calculate decision boundary clarity: {str(e)}")
        return {
            'mean_gradient_magnitude': float('nan'),
            'std_gradient_magnitude': float('nan')
        }


def calculate_interpretability_metrics(
    model: Any,
    x: np.ndarray,
    attribution_func: Optional[Callable] = None,
    perturbation_func: Optional[Callable] = None,
    feature_importances: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive interpretability metrics.
    
    Args:
        model: ML model
        x: Input data
        attribution_func: Function that returns feature attributions (optional)
        perturbation_func: Function that perturbs inputs (optional)
        feature_importances: Array of feature importance values (optional)
        
    Returns:
        dict: Dictionary of interpretability metrics
    """
    interpretability_metrics = {}
    
    # Attribution stability if attribution function is provided
    if attribution_func is not None and perturbation_func is not None:
        try:
            stability_metrics = attribute_stability(
                attribution_func, x, perturbation_func
            )
            interpretability_metrics['stability'] = stability_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate attribution stability: {str(e)}")
            interpretability_metrics['stability'] = {
                'mean_stability': float('nan'),
                'std_stability': float('nan')
            }
    
    # Feature importance concentration if importances are provided
    if feature_importances is not None:
        try:
            concentration_metrics = feature_importance_concentration(feature_importances)
            interpretability_metrics['concentration'] = concentration_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate feature importance concentration: {str(e)}")
            interpretability_metrics['concentration'] = {
                'features_for_threshold': int(len(feature_importances)),
                'feature_concentration_ratio': 1.0,
                'gini_coefficient': float('nan')
            }
    
    # Decision boundary clarity if model has a decision_function
    if hasattr(model, 'decision_function'):
        try:
            decision_function = model.decision_function
            clarity_metrics = decision_boundary_clarity(decision_function, x)
            interpretability_metrics['clarity'] = clarity_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate decision boundary clarity: {str(e)}")
            interpretability_metrics['clarity'] = {
                'mean_gradient_magnitude': float('nan'),
                'std_gradient_magnitude': float('nan')
            }
    
    return interpretability_metrics