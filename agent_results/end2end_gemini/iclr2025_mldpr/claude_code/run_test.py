"""
Basic test script to verify that the modules are loading correctly.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from utils.logger import setup_logger
from utils.data_utils import set_seed, get_device
from target_models.models import SimpleCNN

def main():
    # Set up logging
    logger = setup_logger("test", "logs/test.log")
    logger.info("Running test script")
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load a small sample of CIFAR-10
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        logger.info(f"Successfully loaded CIFAR-10 with {len(dataset)} samples")
        
        # Test creating a model
        model = SimpleCNN()
        model.to(device)
        logger.info(f"Successfully created model: {model.__class__.__name__}")
        
        # Create a small batch
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        inputs, labels = next(iter(dataloader))
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        logger.info(f"Forward pass successful, output shape: {outputs.shape}")
        
        logger.info("All tests passed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)