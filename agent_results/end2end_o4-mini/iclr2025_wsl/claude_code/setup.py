"""
Setup script for permutation-equivariant weight graph embeddings package.
"""
from setuptools import setup, find_packages

setup(
    name="weight-graph-embeddings",
    version="0.1.0",
    description="Permutation-Equivariant Graph Embeddings of Neural Weights",
    author="Claude AI",
    author_email="noreply@anthropic.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.60.0",
        "pandas>=1.2.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)