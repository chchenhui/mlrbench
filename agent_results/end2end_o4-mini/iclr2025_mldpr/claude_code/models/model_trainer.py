"""
Model training module

This module implements functions for training and evaluating machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time
import os
import logging
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    
    def __init__(
        self,
        model_class: Any,
        hyperparams: Dict[str, Any],
        task_type: str = 'classification',
        model_name: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_class: Class of the model to train
            hyperparams: Hyperparameters for the model
            task_type: Type of task ('classification' or 'regression')
            model_name: Name of the model (optional)
            random_state: Random state for reproducibility
        """
        self.model_class = model_class
        self.hyperparams = hyperparams
        self.task_type = task_type
        self.model_name = model_name or model_class.__name__
        self.random_state = random_state
        self.model = None
        self.model_size_mb = None
        self.training_history = {
            'train_scores': [],
            'val_scores': [],
            'training_time': None
        }
    
    def create_model(self) -> Any:
        """
        Create a model instance with the specified hyperparameters.
        
        Returns:
            object: Model instance
        """
        params = self.hyperparams.copy()
        
        # Add random_state if the model supports it
        if 'random_state' in self.model_class().get_params():
            params['random_state'] = self.random_state
        
        return self.model_class(**params)
    
    def calculate_model_size(self, model: Any) -> float:
        """
        Calculate the size of a model in MB.
        
        Args:
            model: Model instance
            
        Returns:
            float: Model size in MB
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile() as tmp:
            joblib.dump(model, tmp.name)
            size_bytes = os.path.getsize(tmp.name)
            size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Any:
        """
        Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            callbacks: List of callback functions to call during training (optional)
            
        Returns:
            object: Trained model
        """
        # Create model
        model = self.create_model()
        
        # Track training time
        start_time = time.time()
        
        # Train model
        if hasattr(model, 'fit'):
            # Handle different fit signatures
            if X_val is not None and y_val is not None and hasattr(model, 'validation_fraction'):
                # Some models like MLPClassifier have validation_fraction
                model.fit(X_train, y_train)
            elif X_val is not None and y_val is not None and hasattr(model, 'partial_fit'):
                # For models that support online learning, we can track validation score after each epoch
                for epoch in range(model.get_params().get('max_iter', 100)):
                    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                    
                    # Calculate training and validation scores
                    train_score = self._calculate_score(model, X_train, y_train)
                    val_score = self._calculate_score(model, X_val, y_val)
                    
                    self.training_history['train_scores'].append(train_score)
                    self.training_history['val_scores'].append(val_score)
                    
                    # Call callbacks if provided
                    if callbacks:
                        for callback in callbacks:
                            callback(epoch, train_score, val_score)
            else:
                # Default fit
                model.fit(X_train, y_train)
                
                # Calculate training score
                train_score = self._calculate_score(model, X_train, y_train)
                self.training_history['train_scores'].append(train_score)
                
                # Calculate validation score if validation data is provided
                if X_val is not None and y_val is not None:
                    val_score = self._calculate_score(model, X_val, y_val)
                    self.training_history['val_scores'].append(val_score)
        else:
            raise ValueError(f"Model {self.model_name} does not have a fit method")
        
        # Record training time
        self.training_history['training_time'] = time.time() - start_time
        
        # Calculate model size
        self.model_size_mb = self.calculate_model_size(model)
        
        # Set model
        self.model = model
        
        return model
    
    def _calculate_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the score of a model on the given data.
        
        Args:
            model: Model instance
            X: Features
            y: Labels
            
        Returns:
            float: Score
        """
        if hasattr(model, 'score'):
            return model.score(X, y)
        else:
            # If model doesn't have a score method, calculate accuracy for classification
            # or R^2 for regression
            y_pred = model.predict(X)
            
            if self.task_type == 'classification':
                return accuracy_score(y, y_pred)
            else:
                from sklearn.metrics import r2_score
                return r2_score(y, y_pred)
    
    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        verbose: int = 1
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform grid search to find the best hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Grid of hyperparameters to search
            cv: Number of cross-validation folds
            verbose: Verbosity level
            
        Returns:
            tuple: (best_model, best_params)
        """
        # Create model
        model = self.create_model()
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            verbose=verbose,
            n_jobs=-1
        )
        
        # Track training time
        start_time = time.time()
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Record training time
        self.training_history['training_time'] = time.time() - start_time
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Calculate model size
        self.model_size_mb = self.calculate_model_size(best_model)
        
        # Set model
        self.model = best_model
        
        # Update hyperparameters with best params
        self.hyperparams.update(best_params)
        
        return best_model, best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities from the trained model.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.model_name} does not support predict_proba")
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values from the trained model.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Decision function values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            # If decision_function is not available but predict_proba is, use the probability of the positive class
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)
                if proba.shape[1] == 2:  # Binary classification
                    return proba[:, 1]
                else:
                    # For multiclass, return the maximum probability
                    return np.max(proba, axis=1)
            else:
                raise ValueError(f"Model {self.model_name} does not support decision_function")
    
    def get_feature_attributions(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get feature attributions using SHAP.
        
        Args:
            X: Features
            feature_names: Names of features (optional)
            
        Returns:
            tuple: (attributions, feature_names)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        try:
            # Use SHAP to explain the model
            # Select a subset of samples for efficiency
            X_subset = X[:min(100, len(X))]
            
            # Check if the model is a tree-based model
            is_tree_based = any(model_type in str(type(self.model)).lower() for model_type in ['tree', 'forest', 'gbm', 'xgb', 'lgbm'])
            
            if is_tree_based:
                explainer = shap.TreeExplainer(self.model)
            else:
                # Use kernel explainer for other model types
                explainer = shap.KernelExplainer(
                    model=self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    data=shap.kmeans(X_subset, min(20, len(X_subset)))
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_subset)
            
            # Handle different formats of SHAP values
            if isinstance(shap_values, list):
                # For multi-class, take the mean absolute value across classes
                if len(shap_values) > 1:
                    attributions = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                else:
                    attributions = np.abs(shap_values[0])
            else:
                attributions = np.abs(shap_values)
            
            # Average over samples
            feature_attributions = np.mean(attributions, axis=0)
            
            return feature_attributions, feature_names
        
        except Exception as e:
            logger.warning(f"Failed to calculate SHAP values: {str(e)}")
            
            # Fall back to feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_, feature_names
            else:
                # Return uniform attributions
                uniform_attributions = np.ones(X.shape[1]) / X.shape[1]
                return uniform_attributions, feature_names
    
    def perturb_input(self, x: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add random noise to an input.
        
        Args:
            x: Input to perturb
            noise_level: Level of noise to add
            
        Returns:
            np.ndarray: Perturbed input
        """
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, x.shape)
        return x + noise
    
    def create_attribution_function(
        self,
        num_samples: int = 100,
        is_image: bool = False
    ) -> Callable:
        """
        Create a function that returns feature attributions for an input.
        
        Args:
            num_samples: Number of background samples for SHAP
            is_image: Whether the input is an image
            
        Returns:
            Callable: Attribution function
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Try to create a SHAP explainer
        try:
            # Check if the model is a tree-based model
            is_tree_based = any(model_type in str(type(self.model)).lower() for model_type in ['tree', 'forest', 'gbm', 'xgb', 'lgbm'])
            
            if is_tree_based:
                explainer = shap.TreeExplainer(self.model)
                
                def attribution_func(x):
                    shap_values = explainer.shap_values(x)
                    if isinstance(shap_values, list):
                        if len(shap_values) > 1:
                            return np.mean([np.abs(sv) for sv in shap_values], axis=0)
                        else:
                            return np.abs(shap_values[0])
                    else:
                        return np.abs(shap_values)
            
            elif is_image:
                # For images, use GradientExplainer
                # This is a simplified version and might not work well
                # For real applications, consider using more advanced explainers
                def attribution_func(x):
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(x)
                        return np.abs(proba - 0.5)  # Simple proxy for attributions
                    else:
                        return np.ones_like(x)  # Fallback
            
            else:
                # For other models, use a simple method based on prediction change
                def attribution_func(x):
                    if not hasattr(self.model, 'predict_proba'):
                        return np.ones_like(x)  # Fallback
                    
                    # Get original prediction
                    orig_pred = self.model.predict_proba(x)
                    
                    # Calculate attributions using prediction differences
                    attributions = np.zeros_like(x, dtype=float)
                    
                    for i in range(x.shape[1]):
                        # Create a perturbed input
                        x_perturbed = x.copy()
                        x_perturbed[:, i] = 0  # Zero out the feature
                        
                        # Get prediction for perturbed input
                        perturbed_pred = self.model.predict_proba(x_perturbed)
                        
                        # Calculate difference in prediction
                        attributions[:, i] = np.abs(orig_pred - perturbed_pred).max(axis=1)
                    
                    return attributions
            
            return attribution_func
        
        except Exception as e:
            logger.warning(f"Failed to create attribution function: {str(e)}")
            
            # Fallback to a simple function
            def simple_attribution_func(x):
                return np.ones_like(x)
            
            return simple_attribution_func
    
    def save_model(self, directory: str, filename: Optional[str] = None) -> str:
        """
        Save the trained model to a file.
        
        Args:
            directory: Directory to save the model
            filename: Filename for the model (optional)
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            filename = f"{self.model_name.lower()}.joblib"
        
        filepath = os.path.join(directory, filename)
        
        joblib.dump(self.model, filepath)
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ModelTrainer':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            ModelTrainer: ModelTrainer instance with the loaded model
        """
        model = joblib.load(filepath)
        
        # Create a new ModelTrainer instance
        model_trainer = cls(
            model_class=type(model),
            hyperparams=model.get_params(),
            model_name=type(model).__name__
        )
        
        model_trainer.model = model
        model_trainer.model_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        return model_trainer


def generate_learning_curve_plot(
    training_history: Dict[str, Any],
    title: str = 'Learning Curve',
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Generate a learning curve plot.
    
    Args:
        training_history: Training history dictionary with train_scores and val_scores
        title: Plot title
        filename: Filename to save the plot (optional)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning curves
    if 'train_scores' in training_history and training_history['train_scores']:
        ax.plot(training_history['train_scores'], label='Training Score')
    
    if 'val_scores' in training_history and training_history['val_scores']:
        ax.plot(training_history['val_scores'], label='Validation Score')
    
    # Add labels and title
    ax.set_xlabel('Epochs / Iterations')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test the model trainer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Initialize model trainer
    model_trainer = ModelTrainer(
        model_class=RandomForestClassifier,
        hyperparams={'n_estimators': 100, 'max_depth': 10},
        model_name='RandomForest'
    )
    
    # Train model
    model = model_trainer.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    y_pred = model_trainer.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Get feature attributions
    attributions, feature_names = model_trainer.get_feature_attributions(X_test)
    
    print("Feature attributions:")
    for name, attr in zip(feature_names, attributions):
        print(f"{name}: {attr:.4f}")
    
    # Generate learning curve plot
    fig = generate_learning_curve_plot(model_trainer.training_history)
    plt.show()