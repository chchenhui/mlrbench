"""
Robustness metrics module

This module implements robustness metrics for evaluating machine learning models
under adversarial attacks and distribution shifts.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.metrics import accuracy_score
import warnings


def adversarial_accuracy(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack_function: Callable,
    epsilon: float = 0.1,
    norm: str = 'l_inf'
) -> float:
    """
    Calculate adversarial accuracy under attacks.
    
    Args:
        model: ML model with predict method
        x_test: Test data
        y_test: True labels
        attack_function: Function to generate adversarial examples
        epsilon: Maximum perturbation size
        norm: Type of norm ('l_inf', 'l2', or 'l1')
        
    Returns:
        float: Adversarial accuracy
    """
    try:
        # Generate adversarial examples
        x_adv = attack_function(model, x_test, y_test, epsilon=epsilon, norm=norm)
        
        # Get predictions on adversarial examples
        y_pred_adv = model.predict(x_adv)
        
        # Calculate adversarial accuracy
        adv_acc = accuracy_score(y_test, y_pred_adv)
        
        return float(adv_acc)
    except Exception as e:
        warnings.warn(f"Failed to calculate adversarial accuracy: {str(e)}")
        return float('nan')


def shift_robustness(
    model: Any,
    x_orig: np.ndarray,
    y_orig: np.ndarray,
    x_shifted: np.ndarray,
    y_shifted: np.ndarray
) -> Dict[str, float]:
    """
    Calculate robustness metrics under distribution shift.
    
    Args:
        model: ML model with predict method
        x_orig: Original test data
        y_orig: Original labels
        x_shifted: Shifted test data
        y_shifted: Shifted labels
        
    Returns:
        dict: Dictionary of shift robustness metrics
    """
    try:
        # Calculate accuracy on original and shifted distributions
        y_pred_orig = model.predict(x_orig)
        y_pred_shifted = model.predict(x_shifted)
        
        acc_orig = accuracy_score(y_orig, y_pred_orig)
        acc_shifted = accuracy_score(y_shifted, y_pred_shifted)
        
        # Calculate drop in accuracy
        shift_drop = 1.0 - (acc_shifted / acc_orig) if acc_orig > 0 else float('nan')
        
        # Calculate retained accuracy percentage
        retained_acc_pct = (acc_shifted / acc_orig) * 100 if acc_orig > 0 else float('nan')
        
        return {
            'original_accuracy': float(acc_orig),
            'shifted_accuracy': float(acc_shifted),
            'shift_drop': float(shift_drop),
            'retained_accuracy_pct': float(retained_acc_pct)
        }
    except Exception as e:
        warnings.warn(f"Failed to calculate shift robustness: {str(e)}")
        return {
            'original_accuracy': float('nan'),
            'shifted_accuracy': float('nan'),
            'shift_drop': float('nan'),
            'retained_accuracy_pct': float('nan')
        }


def noise_robustness(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    noise_type: str = 'gaussian',
    noise_level: float = 0.1
) -> Dict[str, float]:
    """
    Calculate robustness against random noise.
    
    Args:
        model: ML model with predict method
        x_test: Test data
        y_test: True labels
        noise_type: Type of noise ('gaussian' or 'uniform')
        noise_level: Noise level (standard deviation for Gaussian, range for uniform)
        
    Returns:
        dict: Dictionary of noise robustness metrics
    """
    try:
        # Get original predictions and accuracy
        y_pred_orig = model.predict(x_test)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        
        # Add noise to the test data
        x_noisy = x_test.copy()
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, size=x_test.shape)
            x_noisy = x_test + noise
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, size=x_test.shape)
            x_noisy = x_test + noise
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        # Get predictions on noisy data
        y_pred_noisy = model.predict(x_noisy)
        acc_noisy = accuracy_score(y_test, y_pred_noisy)
        
        # Calculate drop in accuracy
        noise_drop = 1.0 - (acc_noisy / acc_orig) if acc_orig > 0 else float('nan')
        
        # Calculate retained accuracy percentage
        retained_acc_pct = (acc_noisy / acc_orig) * 100 if acc_orig > 0 else float('nan')
        
        return {
            'original_accuracy': float(acc_orig),
            'noisy_accuracy': float(acc_noisy),
            'noise_drop': float(noise_drop),
            'retained_accuracy_pct': float(retained_acc_pct)
        }
    except Exception as e:
        warnings.warn(f"Failed to calculate noise robustness: {str(e)}")
        return {
            'original_accuracy': float('nan'),
            'noisy_accuracy': float('nan'),
            'noise_drop': float('nan'),
            'retained_accuracy_pct': float('nan')
        }


def boundary_attack(
    model: Any,
    x_test: np.ndarray,
    decision_function: Callable,
    n_iters: int = 10,
    step_size: float = 0.01
) -> float:
    """
    Simplified implementation of boundary attack to test decision boundary robustness.
    
    Args:
        model: ML model with predict method
        x_test: Test data
        decision_function: Function that returns the confidence score for a sample
        n_iters: Number of iterations for the attack
        step_size: Step size for the attack
        
    Returns:
        float: Average perturbation magnitude needed to cross decision boundary
    """
    try:
        # Implement a simplified version of boundary attack
        n_samples = min(len(x_test), 100)  # Limit to 100 samples for efficiency
        x_subset = x_test[:n_samples]
        
        perturbation_magnitudes = []
        
        for x in x_subset:
            x_current = x.copy().reshape(1, -1)
            orig_score = decision_function(x_current)
            orig_class = 1 if orig_score > 0.5 else 0
            
            # Initialize perturbation
            delta = np.zeros_like(x_current)
            
            # Iteratively search for decision boundary
            for _ in range(n_iters):
                # Add random perturbation in the direction that decreases confidence
                random_direction = np.random.randn(*x_current.shape)
                random_direction = random_direction / np.linalg.norm(random_direction)
                
                if decision_function(x_current + step_size * random_direction) < orig_score:
                    delta += step_size * random_direction
                else:
                    delta -= step_size * random_direction
                
                # Check if we've crossed the decision boundary
                new_score = decision_function(x_current + delta)
                new_class = 1 if new_score > 0.5 else 0
                
                if new_class != orig_class:
                    break
            
            # Calculate the magnitude of the perturbation
            perturbation_magnitude = np.linalg.norm(delta)
            perturbation_magnitudes.append(perturbation_magnitude)
        
        # Return the average perturbation magnitude
        avg_perturbation = np.mean(perturbation_magnitudes) if perturbation_magnitudes else float('nan')
        return float(avg_perturbation)
    except Exception as e:
        warnings.warn(f"Failed to calculate boundary attack: {str(e)}")
        return float('nan')


def calculate_robustness_metrics(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_shifted: Optional[np.ndarray] = None,
    y_shifted: Optional[np.ndarray] = None,
    attack_function: Optional[Callable] = None,
    epsilon: float = 0.1,
    domain: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive robustness metrics.
    
    Args:
        model: ML model with predict method
        x_test: Test data
        y_test: True labels
        x_shifted: Shifted test data (optional)
        y_shifted: Shifted labels (optional)
        attack_function: Function to generate adversarial examples (optional)
        epsilon: Maximum perturbation size for adversarial examples
        domain: Domain of the task (optional)
        
    Returns:
        dict: Dictionary of robustness metrics
    """
    robustness_metrics = {}
    
    # Noise robustness (always calculated)
    noise_metrics = noise_robustness(model, x_test, y_test)
    robustness_metrics['noise'] = noise_metrics
    
    # Shift robustness (if shifted data is provided)
    if x_shifted is not None and y_shifted is not None:
        shift_metrics = shift_robustness(model, x_test, y_test, x_shifted, y_shifted)
        robustness_metrics['shift'] = shift_metrics
    
    # Adversarial robustness (if attack function is provided)
    if attack_function is not None:
        try:
            adv_acc = adversarial_accuracy(model, x_test, y_test, attack_function, epsilon)
            robustness_metrics['adversarial'] = {'adversarial_accuracy': float(adv_acc)}
        except Exception as e:
            warnings.warn(f"Failed to calculate adversarial robustness: {str(e)}")
            robustness_metrics['adversarial'] = {'adversarial_accuracy': float('nan')}
    
    # Calculate domain-specific robustness metrics if domain is provided
    if domain is not None:
        # Add domain-specific robustness metrics here if needed
        pass
    
    return robustness_metrics