#!/usr/bin/env python3
"""
Data processing utilities for Benchmark Cards experiments.
This script handles dataset loading, preprocessing, and analysis.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_name, version=1, test_size=0.2, random_state=42):
    """
    Load and preprocess a dataset from OpenML.
    
    Args:
        dataset_name (str): Name of the dataset on OpenML
        version (int): Version of the dataset
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing:
            - X_train, X_test: Training and test features
            - y_train, y_test: Training and test labels
            - feature_names: List of feature names
            - target_names: List of target class names
            - categorical_features: List of categorical feature indices
            - numerical_features: List of numerical feature indices
            - preprocessor: Fitted preprocessor for transforming new data
    """
    logger.info(f"Loading dataset: {dataset_name} (version {version})")
    
    try:
        # Fetch dataset from OpenML
        data = fetch_openml(name=dataset_name, version=version, as_frame=True)
        X, y = data.data, data.target
        
        # Basic dataset info
        n_samples, n_features = X.shape
        feature_names = X.columns.tolist()
        feature_types = X.dtypes
        target_names = y.unique().tolist()
        
        logger.info(f"Dataset info: {n_samples} samples, {n_features} features, {len(target_names)} classes")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Identify categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for i, (name, dtype) in enumerate(zip(feature_names, feature_types)):
            # Check if feature is categorical
            if dtype == 'object' or dtype == 'category':
                categorical_features.append(name)
            # Check if feature looks categorical (few unique values)
            elif X[name].nunique() < 10:
                categorical_features.append(name)
            # Otherwise, assume numerical
            else:
                numerical_features.append(name)
        
        logger.info(f"Feature types: {len(categorical_features)} categorical, {len(numerical_features)} numerical")
        
        # Create preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit preprocessor on training data
        preprocessor.fit(X_train)
        
        # Transform data
        X_train_processed = pd.DataFrame(
            preprocessor.transform(X_train).toarray(),
            index=X_train.index
        )
        
        X_test_processed = pd.DataFrame(
            preprocessor.transform(X_test).toarray(),
            index=X_test.index
        )
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': target_names,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'raw_X_train': X_train,
            'raw_X_test': X_test,
            'preprocessor': preprocessor
        }
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def analyze_dataset(dataset_info, output_dir=None):
    """
    Analyze a dataset and create visualizations.
    
    Args:
        dataset_info (dict): Dataset information from load_dataset()
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Dictionary of dataset statistics
    """
    logger.info("Analyzing dataset")
    
    # Extract raw data for analysis
    X_train = dataset_info['raw_X_train']
    y_train = dataset_info['y_train']
    feature_names = dataset_info['feature_names']
    target_names = dataset_info['target_names']
    categorical_features = dataset_info['categorical_features']
    numerical_features = dataset_info['numerical_features']
    
    # Create statistics dictionary
    stats = {
        'n_samples': len(X_train),
        'n_features': len(feature_names),
        'n_classes': len(target_names),
        'class_distribution': y_train.value_counts().to_dict(),
        'features': {}
    }
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Class distribution visualizations
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y=y_train)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Class")
    
    if output_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
        plt.close()
    
    # Analyze each feature
    for feature in feature_names:
        # Skip features with too many missing values
        if X_train[feature].isna().sum() / len(X_train) > 0.5:
            continue
            
        # Compute feature statistics
        if feature in numerical_features:
            stats['features'][feature] = {
                'type': 'numerical',
                'mean': X_train[feature].mean(),
                'std': X_train[feature].std(),
                'min': X_train[feature].min(),
                'max': X_train[feature].max(),
                'missing': X_train[feature].isna().sum()
            }
            
            # Create histogram
            if output_dir:
                plt.figure(figsize=(8, 4))
                ax = sns.histplot(X_train[feature].dropna(), kde=True)
                ax.set_title(f"Distribution of {feature}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"feature_{feature}_hist.png"), dpi=300)
                plt.close()
                
        else:  # Categorical feature
            stats['features'][feature] = {
                'type': 'categorical',
                'categories': X_train[feature].value_counts().to_dict(),
                'n_categories': X_train[feature].nunique(),
                'missing': X_train[feature].isna().sum()
            }
            
            # Create bar plot for categorical features
            if output_dir and X_train[feature].nunique() < 15:  # Only if not too many categories
                plt.figure(figsize=(10, 6))
                ax = sns.countplot(y=X_train[feature])
                ax.set_title(f"Distribution of {feature}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"feature_{feature}_bar.png"), dpi=300)
                plt.close()
    
    # Check for correlations between numerical features
    if len(numerical_features) > 1:
        # Compute correlation matrix
        corr_matrix = X_train[numerical_features].corr()
        stats['feature_correlations'] = corr_matrix.to_dict()
        
        # Create correlation heatmap
        if output_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Feature Correlations")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_correlations.png"), dpi=300)
            plt.close()
    
    logger.info("Dataset analysis completed")
    return stats


def main():
    """Main function to process and analyze a dataset."""
    parser = argparse.ArgumentParser(description="Process and analyze a dataset")
    parser.add_argument("--dataset", type=str, default="adult",
                        help="Name of the dataset on OpenML")
    parser.add_argument("--version", type=int, default=1,
                        help="Version of the dataset")
    parser.add_argument("--output-dir", type=str, default="dataset_analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    try:
        # Load dataset
        dataset_info = load_dataset(args.dataset, args.version)
        
        # Analyze dataset
        stats = analyze_dataset(dataset_info, args.output_dir)
        
        # Save statistics to file
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"{args.dataset}_stats.json"), 'w') as f:
            import json
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Dataset statistics saved to {os.path.join(args.output_dir, f'{args.dataset}_stats.json')}")
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")


if __name__ == "__main__":
    main()