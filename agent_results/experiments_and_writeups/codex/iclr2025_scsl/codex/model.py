import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """Simple CNN encoder for 2-channel 28x28 input."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # 2x28x28 -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                # 32x14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 64x7x7
            nn.Flatten(),                   # 64*7*7
            nn.Linear(64*7*7, latent_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    """Classifier head mapping latent vectors to logits."""
    def __init__(self, latent_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
    def forward(self, z):
        return self.fc(z)
