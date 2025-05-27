"""
Dataset loading and preprocessing module

This module implements functions for loading and preprocessing datasets for experiments.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import logging
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_path(dataset_name: str, data_dir: str = '../data') -> str:
    """
    Get the path to a dataset file.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Data directory
        
    Returns:
        str: Path to the dataset file
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    
    return dataset_dir


def download_adult_dataset(data_dir: str = '../data') -> str:
    """
    Download the Adult Census Income dataset if it doesn't exist locally.
    
    Args:
        data_dir: Data directory
        
    Returns:
        str: Path to the dataset file
    """
    import requests
    from io import StringIO
    
    dataset_name = 'adult'
    dataset_dir = get_dataset_path(dataset_name, data_dir)
    dataset_file = os.path.join(dataset_dir, 'adult.csv')
    
    # Check if dataset already exists
    if os.path.exists(dataset_file):
        logger.info(f"Adult dataset already exists at {dataset_file}")
        return dataset_file
    
    # URLs for the dataset
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    
    # Column names
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        # Download training data
        logger.info(f"Downloading Adult dataset training data from {train_url}")
        train_response = requests.get(train_url)
        train_data = pd.read_csv(StringIO(train_response.text), header=None, names=column_names, sep=', ', engine='python')
        
        # Download test data
        logger.info(f"Downloading Adult dataset test data from {test_url}")
        test_response = requests.get(test_url)
        # Skip the first line which is a description
        test_data = pd.read_csv(StringIO(test_response.text), header=None, names=column_names, sep=', ', engine='python', skiprows=1)
        
        # Remove dot from income in test data
        test_data['income'] = test_data['income'].str.rstrip('.')
        
        # Combine train and test data
        adult_data = pd.concat([train_data, test_data], ignore_index=True)
        
        # Save to CSV
        adult_data.to_csv(dataset_file, index=False)
        logger.info(f"Adult dataset saved to {dataset_file}")
        
        return dataset_file
    
    except Exception as e:
        logger.error(f"Error downloading Adult dataset: {str(e)}")
        raise


def download_mnist_dataset(data_dir: str = '../data') -> str:
    """
    Download the MNIST dataset if it doesn't exist locally.
    
    Args:
        data_dir: Data directory
        
    Returns:
        str: Path to the dataset directory
    """
    from tensorflow.keras.datasets import mnist
    
    dataset_name = 'mnist'
    dataset_dir = get_dataset_path(dataset_name, data_dir)
    dataset_file = os.path.join(dataset_dir, 'mnist.npz')
    
    # Check if dataset already exists
    if os.path.exists(dataset_file):
        logger.info(f"MNIST dataset already exists at {dataset_file}")
        return dataset_dir
    
    try:
        # Download MNIST dataset using Keras
        logger.info("Downloading MNIST dataset")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Save to NPZ file
        np.savez_compressed(
            dataset_file,
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test
        )
        
        logger.info(f"MNIST dataset saved to {dataset_file}")
        
        return dataset_dir
    
    except Exception as e:
        logger.error(f"Error downloading MNIST dataset: {str(e)}")
        raise


def download_sst2_dataset(data_dir: str = '../data') -> str:
    """
    Download the SST-2 dataset if it doesn't exist locally.
    
    Args:
        data_dir: Data directory
        
    Returns:
        str: Path to the dataset directory
    """
    from datasets import load_dataset
    
    dataset_name = 'sst2'
    dataset_dir = get_dataset_path(dataset_name, data_dir)
    dataset_file = os.path.join(dataset_dir, 'sst2.csv')
    
    # Check if dataset already exists
    if os.path.exists(dataset_file):
        logger.info(f"SST-2 dataset already exists at {dataset_file}")
        return dataset_dir
    
    try:
        # Download SST-2 dataset using Hugging Face datasets
        logger.info("Downloading SST-2 dataset")
        dataset = load_dataset('glue', 'sst2')
        
        # Convert to DataFrame
        train_df = pd.DataFrame(dataset['train'])
        validation_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Save to CSV
        train_df.to_csv(os.path.join(dataset_dir, 'sst2_train.csv'), index=False)
        validation_df.to_csv(os.path.join(dataset_dir, 'sst2_validation.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_dir, 'sst2_test.csv'), index=False)
        
        # Combine and save to single CSV for convenience
        combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
        combined_df.to_csv(dataset_file, index=False)
        
        logger.info(f"SST-2 dataset saved to {dataset_dir}")
        
        return dataset_dir
    
    except Exception as e:
        logger.error(f"Error downloading SST-2 dataset: {str(e)}")
        raise


def preprocess_adult_dataset(
    data_path: str,
    random_state: int = 42,
    test_size: float = 0.2,
    preprocessing_dir: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Preprocess the Adult Census Income dataset.
    
    Args:
        data_path: Path to the dataset file
        random_state: Random state for reproducibility
        test_size: Test set proportion
        preprocessing_dir: Directory to save preprocessing objects (optional)
        
    Returns:
        tuple: (data_dict, preprocessor_dict)
            - data_dict: Dictionary with X_train, y_train, X_test, y_test, and feature_data
            - preprocessor_dict: Dictionary with preprocessing objects
    """
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Clean up data
    # Strip whitespaces from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Rename target column values for consistency
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
    
    # Split features and target
    X = df.drop('income', axis=1)
    y = df['income']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Extract sensitive attributes (before preprocessing)
    sensitive_features = {}
    
    for feature in ['race', 'sex', 'age']:
        if feature in X_train.columns:
            if feature == 'age':
                # Convert age to categorical (young, adult, elderly)
                age_bins = [0, 25, 60, 100]
                age_labels = ['young', 'adult', 'elderly']
                X_train_age_cat = pd.cut(X_train[feature], bins=age_bins, labels=age_labels)
                X_test_age_cat = pd.cut(X_test[feature], bins=age_bins, labels=age_labels)
                
                sensitive_features[feature] = {
                    'train': X_train_age_cat.astype(str).values,
                    'test': X_test_age_cat.astype(str).values
                }
            else:
                sensitive_features[feature] = {
                    'train': X_train[feature].values,
                    'test': X_test[feature].values
                }
    
    # Create shifted test set (simulate domain shift)
    # For simplicity, let's create a shift by filtering on a condition
    # For example, select only examples with age > 40
    shifted_mask = X_test['age'] > 40
    X_test_shifted = X_test[shifted_mask].copy()
    y_test_shifted = y_test[shifted_mask].copy()
    X_test_shifted_processed = X_test_processed[shifted_mask]
    
    # Extract feature names after one-hot encoding
    try:
        categorical_features = preprocessor.named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
        feature_names = numerical_cols + categorical_features
    except:
        # If get_feature_names_out fails, create generic feature names
        n_features = X_train_processed.shape[1]
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Save preprocessor if directory is provided
    if preprocessing_dir:
        os.makedirs(preprocessing_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(preprocessing_dir, 'adult_preprocessor.joblib'))
        
        # Save feature names
        with open(os.path.join(preprocessing_dir, 'adult_feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
    
    # Prepare return values
    data_dict = {
        'X_train': X_train_processed,
        'y_train': y_train.values,
        'X_test': X_test_processed,
        'y_test': y_test.values,
        'X_test_shifted': X_test_shifted_processed,
        'y_test_shifted': y_test_shifted.values,
        'feature_data': sensitive_features
    }
    
    preprocessor_dict = {
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    return data_dict, preprocessor_dict


def preprocess_mnist_dataset(
    data_dir: str,
    random_state: int = 42,
    test_size: float = 0.2,
    preprocessing_dir: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Preprocess the MNIST dataset.
    
    Args:
        data_dir: Path to the dataset directory
        random_state: Random state for reproducibility
        test_size: Validation set proportion (from training set)
        preprocessing_dir: Directory to save preprocessing objects (optional)
        
    Returns:
        tuple: (data_dict, preprocessor_dict)
            - data_dict: Dictionary with X_train, y_train, X_test, y_test, and feature_data
            - preprocessor_dict: Dictionary with preprocessing objects
    """
    # Load dataset
    dataset_file = os.path.join(data_dir, 'mnist.npz')
    
    with np.load(dataset_file) as data:
        X_train_full = data['X_train']
        y_train_full = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    
    # Split training set into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=random_state, stratify=y_train_full
    )
    
    # Preprocess images: normalize and reshape
    X_train_processed = X_train.reshape(-1, 28*28).astype('float32') / 255
    X_val_processed = X_val.reshape(-1, 28*28).astype('float32') / 255
    X_test_processed = X_test.reshape(-1, 28*28).astype('float32') / 255
    
    # Create shifted test set (simulate domain shift with noise)
    # Add noise to create a domain shift
    np.random.seed(random_state)
    noise_level = 0.1
    X_test_shifted = X_test_processed + np.random.normal(0, noise_level, X_test_processed.shape)
    X_test_shifted = np.clip(X_test_shifted, 0, 1)  # Ensure values stay between 0 and 1
    
    # Create synthetic sensitive attributes for demonstration
    # For MNIST, we'll create artificial sensitive attributes
    # (in a real application, these might be demographic attributes of the writers)
    n_test = len(y_test)
    n_train = len(y_train)
    
    # Create region feature (synthetic: not real data)
    train_regions = np.random.choice(['US', 'EU', 'Asia'], size=n_train, p=[0.5, 0.3, 0.2])
    test_regions = np.random.choice(['US', 'EU', 'Asia'], size=n_test, p=[0.4, 0.3, 0.3])
    
    # Create writer age feature (synthetic: not real data)
    train_ages = np.random.choice(['young', 'adult', 'elderly'], size=n_train, p=[0.2, 0.6, 0.2])
    test_ages = np.random.choice(['young', 'adult', 'elderly'], size=n_test, p=[0.1, 0.6, 0.3])
    
    sensitive_features = {
        'region': {
            'train': train_regions,
            'test': test_regions
        },
        'writer_age': {
            'train': train_ages,
            'test': test_ages
        }
    }
    
    # Prepare return values
    data_dict = {
        'X_train': X_train_processed,
        'y_train': y_train,
        'X_val': X_val_processed,
        'y_val': y_val,
        'X_test': X_test_processed,
        'y_test': y_test,
        'X_test_shifted': X_test_shifted,
        'y_test_shifted': y_test,  # Same labels, different features
        'X_train_original': X_train,  # Keep original images for visualization
        'X_test_original': X_test,  # Keep original images for visualization
        'feature_data': sensitive_features
    }
    
    # No complex preprocessing for MNIST, but still return a dict for consistency
    preprocessor_dict = {
        'feature_names': [f'pixel_{i}' for i in range(28*28)],
        'image_shape': (28, 28)
    }
    
    return data_dict, preprocessor_dict


def preprocess_sst2_dataset(
    data_dir: str,
    random_state: int = 42,
    max_features: int = 10000,
    max_len: int = 100,
    preprocessing_dir: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Preprocess the SST-2 dataset.
    
    Args:
        data_dir: Path to the dataset directory
        random_state: Random state for reproducibility
        max_features: Maximum number of features (vocabulary size)
        max_len: Maximum sequence length
        preprocessing_dir: Directory to save preprocessing objects (optional)
        
    Returns:
        tuple: (data_dict, preprocessor_dict)
            - data_dict: Dictionary with X_train, y_train, X_test, y_test, and feature_data
            - preprocessor_dict: Dictionary with preprocessing objects
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Load dataset
    train_df = pd.read_csv(os.path.join(data_dir, 'sst2_train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'sst2_validation.csv'))
    
    # Combine train and validation sets
    train_sentences = train_df['sentence'].values
    train_labels = train_df['label'].values
    
    val_sentences = val_df['sentence'].values
    val_labels = val_df['label'].values
    
    # Create a tokenizer
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_sentences)
    
    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    
    # Pad sequences
    X_train = pad_sequences(train_sequences, maxlen=max_len)
    X_val = pad_sequences(val_sequences, maxlen=max_len)
    
    # Create shifted test set (simulate domain shift)
    # For text, we'll simulate a domain shift by selecting longer sentences
    sentence_lengths = np.array([len(s.split()) for s in val_sentences])
    length_threshold = np.percentile(sentence_lengths, 70)  # Top 30% longest sentences
    
    shifted_mask = sentence_lengths >= length_threshold
    X_val_shifted = X_val[shifted_mask]
    y_val_shifted = val_labels[shifted_mask]
    
    # Create synthetic sensitive attributes for demonstration
    # (in a real application, these might be demographic attributes of the writers)
    n_train = len(train_labels)
    n_val = len(val_labels)
    
    # Create synthetic text length feature
    train_lengths = np.array([len(s.split()) for s in train_sentences])
    val_lengths = np.array([len(s.split()) for s in val_sentences])
    
    train_length_cat = pd.cut(train_lengths, bins=[0, 10, 20, 100], labels=['short', 'medium', 'long'])
    val_length_cat = pd.cut(val_lengths, bins=[0, 10, 20, 100], labels=['short', 'medium', 'long'])
    
    # Create synthetic sentiment complexity feature
    # For simplicity, let's say sentences with many adjectives are more complex
    # We'll just simulate this with a random assignment correlated with sentence length
    np.random.seed(random_state)
    
    train_complexity = np.random.choice(
        ['simple', 'moderate', 'complex'],
        size=n_train,
        p=[0.3, 0.4, 0.3]
    )
    val_complexity = np.random.choice(
        ['simple', 'moderate', 'complex'],
        size=n_val,
        p=[0.3, 0.4, 0.3]
    )
    
    sensitive_features = {
        'text_length': {
            'train': train_length_cat.astype(str).values,
            'test': val_length_cat.astype(str).values
        },
        'complexity': {
            'train': train_complexity,
            'test': val_complexity
        }
    }
    
    # Save preprocessor if directory is provided
    if preprocessing_dir:
        os.makedirs(preprocessing_dir, exist_ok=True)
        with open(os.path.join(preprocessing_dir, 'sst2_tokenizer.joblib'), 'wb') as f:
            joblib.dump(tokenizer, f)
    
    # Prepare return values
    data_dict = {
        'X_train': X_train,
        'y_train': train_labels,
        'X_test': X_val,
        'y_test': val_labels,
        'X_test_shifted': X_val_shifted,
        'y_test_shifted': y_val_shifted,
        'train_sentences': train_sentences,  # Keep original text for interpretability
        'test_sentences': val_sentences,  # Keep original text for interpretability
        'feature_data': sensitive_features
    }
    
    preprocessor_dict = {
        'tokenizer': tokenizer,
        'max_features': max_features,
        'max_len': max_len,
        'word_index': tokenizer.word_index
    }
    
    return data_dict, preprocessor_dict


def load_and_preprocess_dataset(
    dataset_name: str,
    data_dir: str = '../data',
    random_state: int = 42,
    download: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load and preprocess a dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('adult', 'mnist', 'sst2')
        data_dir: Data directory
        random_state: Random state for reproducibility
        download: Whether to download the dataset if it doesn't exist
        
    Returns:
        tuple: (data_dict, preprocessor_dict)
            - data_dict: Dictionary with preprocessed data
            - preprocessor_dict: Dictionary with preprocessing objects
    """
    dataset_dir = get_dataset_path(dataset_name, data_dir)
    preprocessed_dir = os.path.join(dataset_dir, 'preprocessed')
    
    # Create preprocessed directory
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Check if preprocessed data already exists
    preprocessed_file = os.path.join(preprocessed_dir, f'{dataset_name}_preprocessed.joblib')
    
    if os.path.exists(preprocessed_file):
        logger.info(f"Loading preprocessed data from {preprocessed_file}")
        data = joblib.load(preprocessed_file)
        return data['data_dict'], data['preprocessor_dict']
    
    # Download dataset if needed
    if download:
        if dataset_name == 'adult':
            data_path = download_adult_dataset(data_dir)
            data_dict, preprocessor_dict = preprocess_adult_dataset(
                data_path, random_state, preprocessing_dir=preprocessed_dir
            )
        elif dataset_name == 'mnist':
            data_dir = download_mnist_dataset(data_dir)
            data_dict, preprocessor_dict = preprocess_mnist_dataset(
                data_dir, random_state, preprocessing_dir=preprocessed_dir
            )
        elif dataset_name == 'sst2':
            data_dir = download_sst2_dataset(data_dir)
            data_dict, preprocessor_dict = preprocess_sst2_dataset(
                data_dir, random_state, preprocessing_dir=preprocessed_dir
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    else:
        # Try to find and load the dataset
        if dataset_name == 'adult':
            data_path = os.path.join(dataset_dir, 'adult.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}. Set download=True to download it.")
            
            data_dict, preprocessor_dict = preprocess_adult_dataset(
                data_path, random_state, preprocessing_dir=preprocessed_dir
            )
        elif dataset_name == 'mnist':
            data_path = os.path.join(dataset_dir, 'mnist.npz')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}. Set download=True to download it.")
            
            data_dict, preprocessor_dict = preprocess_mnist_dataset(
                dataset_dir, random_state, preprocessing_dir=preprocessed_dir
            )
        elif dataset_name == 'sst2':
            data_path = os.path.join(dataset_dir, 'sst2.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}. Set download=True to download it.")
            
            data_dict, preprocessor_dict = preprocess_sst2_dataset(
                dataset_dir, random_state, preprocessing_dir=preprocessed_dir
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Save preprocessed data for future use
    joblib.dump(
        {'data_dict': data_dict, 'preprocessor_dict': preprocessor_dict},
        preprocessed_file
    )
    
    return data_dict, preprocessor_dict


def create_adversarial_examples(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    norm: str = 'l_inf'
) -> np.ndarray:
    """
    Create simple adversarial examples using the Fast Gradient Sign Method (FGSM).
    
    Args:
        model: Model with predict_proba method
        X: Input data
        y: True labels
        epsilon: Maximum perturbation size
        norm: Type of norm ('l_inf', 'l2', or 'l1')
        
    Returns:
        np.ndarray: Adversarial examples
    """
    try:
        # This is a simplified implementation and might not work for all models
        # For a real application, consider using a library like Foolbox or cleverhans
        
        # Calculate gradient of loss with respect to input
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        X_adv = X.copy()
        
        # For simplicity, we'll just add random noise in the direction that decreases confidence
        if isinstance(model, (LogisticRegression, SVC)) and hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
            
            # Determine the sign to move away from the correct class
            if len(scores.shape) > 1:
                # Multi-class
                # For each sample, find the direction away from the correct class
                for i in range(len(X)):
                    if norm == 'l_inf':
                        # Add uniform noise of magnitude epsilon
                        noise = np.random.uniform(-epsilon, epsilon, X[i].shape)
                    elif norm == 'l2':
                        # Add Gaussian noise normalized to epsilon
                        noise = np.random.normal(0, 1, X[i].shape)
                        noise = epsilon * noise / np.linalg.norm(noise)
                    elif norm == 'l1':
                        # Add sparse noise
                        noise = np.zeros_like(X[i])
                        idx = np.random.choice(noise.shape[0], size=max(1, int(0.1 * noise.shape[0])))
                        noise[idx] = np.random.choice([-epsilon, epsilon], size=len(idx))
                    
                    # If decision_function is high (positive), add negative noise to reduce it
                    sign = -1 if scores[i] > 0 else 1
                    X_adv[i] += sign * noise
                
        else:
            # For other models, we'll just add random noise
            if norm == 'l_inf':
                noise = np.random.uniform(-epsilon, epsilon, X.shape)
            elif norm == 'l2':
                noise = np.random.normal(0, 1, X.shape)
                for i in range(len(X)):
                    noise[i] = epsilon * noise[i] / np.linalg.norm(noise[i])
            elif norm == 'l1':
                noise = np.zeros_like(X)
                for i in range(len(X)):
                    idx = np.random.choice(X[i].shape[0], size=max(1, int(0.1 * X[i].shape[0])))
                    noise[i, idx] = np.random.choice([-epsilon, epsilon], size=len(idx))
            
            X_adv += noise
        
        return X_adv
    
    except Exception as e:
        logger.warning(f"Failed to create adversarial examples: {str(e)}")
        return X


if __name__ == "__main__":
    # Download and preprocess datasets
    data_dir = '../data'
    
    for dataset_name in ['adult', 'mnist', 'sst2']:
        try:
            logger.info(f"Processing dataset: {dataset_name}")
            data_dict, preprocessor_dict = load_and_preprocess_dataset(dataset_name, data_dir)
            logger.info(f"Successfully processed {dataset_name} dataset")
        except Exception as e:
            logger.error(f"Error processing {dataset_name} dataset: {str(e)}")