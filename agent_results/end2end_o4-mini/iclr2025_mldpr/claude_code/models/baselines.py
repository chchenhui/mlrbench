"""
Baseline methods module

This module implements baseline methods for comparison in the ContextBench experiments.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_classification_baselines() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Get baseline classification models for comparison.
    
    Returns:
        dict: Dictionary mapping model names to tuples of (model_class, hyperparameters)
    """
    baselines = {
        'LogisticRegression': (
            LogisticRegression,
            {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'multi_class': 'auto',
                'random_state': 42
            }
        ),
        
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        ),
        
        'SVM': (
            SVC,
            {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            }
        ),
        
        'MLP': (
            MLPClassifier,
            {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'max_iter': 200,
                'early_stopping': True,
                'random_state': 42
            }
        ),
        
        'XGBoost': (
            XGBClassifier,
            {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            }
        )
    }
    
    return baselines


def get_regression_baselines() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Get baseline regression models for comparison.
    
    Returns:
        dict: Dictionary mapping model names to tuples of (model_class, hyperparameters)
    """
    baselines = {
        'Ridge': (
            Ridge,
            {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            }
        ),
        
        'RandomForest': (
            RandomForestRegressor,
            {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        ),
        
        'SVR': (
            SVR,
            {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        ),
        
        'MLP': (
            MLPRegressor,
            {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'max_iter': 200,
                'early_stopping': True,
                'random_state': 42
            }
        ),
        
        'XGBoost': (
            XGBRegressor,
            {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        )
    }
    
    return baselines


def get_vision_baselines() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Get baseline vision models for comparison.
    
    Returns:
        dict: Dictionary mapping model names to tuples of (model_class, hyperparameters)
    """
    # For vision, we'll use simplified models for demonstration purposes
    # In a real application, you might use CNNs or pre-trained models
    
    baselines = {
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        ),
        
        'MLP': (
            MLPClassifier,
            {
                'hidden_layer_sizes': (256, 128),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 64,
                'max_iter': 50,
                'early_stopping': True,
                'random_state': 42
            }
        ),
        
        'XGBoost': (
            XGBClassifier,
            {
                'n_estimators': 100,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softprob',
                'random_state': 42
            }
        )
    }
    
    return baselines


def get_text_baselines() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Get baseline text models for comparison.
    
    Returns:
        dict: Dictionary mapping model names to tuples of (model_class, hyperparameters)
    """
    # For text, we'll use simplified models for demonstration purposes
    # In a real application, you might use transformers or pre-trained models
    
    baselines = {
        'LogisticRegression': (
            LogisticRegression,
            {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42
            }
        ),
        
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        ),
        
        'SVM': (
            SVC,
            {
                'C': 1.0,
                'kernel': 'linear',
                'probability': True,
                'random_state': 42
            }
        ),
        
        'MLP': (
            MLPClassifier,
            {
                'hidden_layer_sizes': (256, 128),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 64,
                'max_iter': 50,
                'early_stopping': True,
                'random_state': 42
            }
        )
    }
    
    return baselines


def get_baselines_for_domain(domain: str, task_type: str = 'classification') -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Get baseline models for a specific domain and task type.
    
    Args:
        domain: Domain name ('tabular', 'vision', 'text')
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        dict: Dictionary mapping model names to tuples of (model_class, hyperparameters)
    """
    if domain == 'vision':
        return get_vision_baselines()
    elif domain == 'text':
        return get_text_baselines()
    else:  # tabular data
        if task_type == 'classification':
            return get_classification_baselines()
        else:  # regression
            return get_regression_baselines()


def get_shap_perturbation_function(
    is_image: bool = False,
    is_text: bool = False,
    noise_level: float = 0.1
) -> callable:
    """
    Get a function for perturbing inputs for SHAP analysis.
    
    Args:
        is_image: Whether the input is an image
        is_text: Whether the input is text
        noise_level: Level of noise to add
        
    Returns:
        callable: Perturbation function
    """
    if is_image:
        # For images, we'll use Gaussian noise
        def perturb_image(x: np.ndarray) -> np.ndarray:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, x.shape)
            # Ensure pixel values stay in [0, 1]
            return np.clip(x + noise, 0, 1)
        
        return perturb_image
    
    elif is_text:
        # For text data (which is often one-hot encoded or embedded),
        # we'll use a different perturbation strategy
        def perturb_text(x: np.ndarray) -> np.ndarray:
            # Randomly set some elements to zero (mask tokens)
            mask = np.random.binomial(1, noise_level, x.shape).astype(bool)
            x_perturbed = x.copy()
            x_perturbed[mask] = 0
            return x_perturbed
        
        return perturb_text
    
    else:
        # For tabular data, we'll use Gaussian noise
        def perturb_tabular(x: np.ndarray) -> np.ndarray:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, x.shape)
            return x + noise
        
        return perturb_tabular


if __name__ == "__main__":
    # Test getting baselines
    for domain in ['tabular', 'vision', 'text']:
        for task_type in ['classification', 'regression']:
            try:
                baselines = get_baselines_for_domain(domain, task_type)
                print(f"Baselines for {domain} {task_type}:")
                for name, (model_class, hyperparams) in baselines.items():
                    print(f"  {name}: {model_class.__name__} with {len(hyperparams)} hyperparameters")
            except Exception as e:
                print(f"Error getting baselines for {domain} {task_type}: {str(e)}")
                
    # Test perturbation functions
    for data_type in ['tabular', 'image', 'text']:
        is_image = data_type == 'image'
        is_text = data_type == 'text'
        
        perturb_func = get_shap_perturbation_function(is_image, is_text)
        
        # Create sample input
        if data_type == 'image':
            x = np.random.rand(1, 28, 28)
        else:
            x = np.random.rand(1, 10)
        
        # Apply perturbation
        x_perturbed = perturb_func(x)
        
        print(f"Perturbation for {data_type} data:")
        print(f"  Original shape: {x.shape}")
        print(f"  Perturbed shape: {x_perturbed.shape}")
        print(f"  Mean abs difference: {np.mean(np.abs(x - x_perturbed)):.6f}")