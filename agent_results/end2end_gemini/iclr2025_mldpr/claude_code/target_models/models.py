"""
Target model implementations for the AEB project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Simple CNN model for image classification."""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size after convolutions and pooling
        # For CIFAR-10: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MLP(nn.Module):
    """Simple MLP model for image classification."""
    
    def __init__(self, num_classes=10, input_channels=3, input_size=32):
        super(MLP, self).__init__()
        input_dim = input_channels * input_size * input_size
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNet18(nn.Module):
    """ResNet18 model adapted for CIFAR-10."""
    
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace the first 7x7 conv with a 3x3 conv more suitable for CIFAR-10
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the first max pooling layer
        self.model.maxpool = nn.Identity()
        
        # Change the final layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class LeNet(nn.Module):
    """LeNet model for image classification."""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # For CIFAR-10 (32x32), calculate the size after convolutions and pooling
        # After conv1 and pool: 32x32 -> 28x28 -> 14x14
        # After conv2 and pool: 14x14 -> 10x10 -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class AdversarialModel(nn.Module):
    """A model trained on both standard and adversarial examples."""
    
    def __init__(self, base_model):
        super(AdversarialModel, self).__init__()
        self.base_model = base_model
    
    def forward(self, x):
        return self.base_model(x)


def get_model(model_name, num_classes=10, pretrained=False):
    """Factory function to get a model by name."""
    model_name = model_name.lower()
    
    if model_name == 'simplecnn':
        return SimpleCNN(num_classes=num_classes)
    
    elif model_name == 'mlp':
        return MLP(num_classes=num_classes)
    
    elif model_name == 'resnet18':
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    
    elif model_name == 'lenet':
        return LeNet(num_classes=num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def create_adversarial_model(base_model_name, base_model_path, num_classes=10):
    """Create an adversarial model based on a trained base model."""
    # Get the base model
    base_model = get_model(base_model_name, num_classes=num_classes)
    
    # Load pretrained weights
    base_model.load_state_dict(torch.load(base_model_path))
    
    # Create and return the adversarial model
    return AdversarialModel(base_model)