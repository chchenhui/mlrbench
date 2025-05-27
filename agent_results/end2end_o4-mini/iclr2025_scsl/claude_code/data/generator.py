"""
SpurGen - A synthetic multimodal dataset generator for studying spurious correlations.
"""

import os
import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import json

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SpurGen:
    """
    SpurGen: Synthetic multimodal benchmark for spurious correlation detection.
    
    Generates paired data (images and captions) with configurable spurious channels.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        num_samples: int = 20000,
        image_size: Tuple[int, int] = (224, 224),
        class_names: Optional[List[str]] = None,
        spurious_channels: Optional[Dict] = None,
        save_dir: str = "data",
    ):
        """
        Initialize the SpurGen dataset generator.
        
        Args:
            num_classes: Number of different classes to generate
            num_samples: Total number of samples to generate
            image_size: Size of the generated images (height, width)
            class_names: Optional list of class names (defaults to "class_0", "class_1", etc.)
            spurious_channels: Dictionary of spurious channels and their attributes
            save_dir: Directory to save generated data
        """
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_size = image_size
        
        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            assert len(class_names) == num_classes
            self.class_names = class_names
            
        # Default spurious channels if none provided
        if spurious_channels is None:
            self.spurious_channels = {
                "background": {
                    "attributes": ["stripes", "dots", "solid", "checkered"],
                    "p_alignment": {i: 0.9 if i % 2 == 0 else 0.1 for i in range(num_classes)}
                },
                "color": {
                    "attributes": ["red", "blue", "green", "yellow"],
                    "p_alignment": {i: 0.9 if i % 3 == 0 else 0.1 for i in range(num_classes)}
                },
                "shape": {
                    "attributes": ["circle", "square", "triangle", "star"],
                    "p_alignment": {i: 0.9 if i % 5 == 0 else 0.1 for i in range(num_classes)}
                }
            }
        else:
            self.spurious_channels = spurious_channels
            
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Store the dataset splits and metadata
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.metadata = {}
    
    def generate_dataset(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2) -> Dict:
        """
        Generate the synthetic dataset with the specified class and spurious attribute distributions.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            
        Returns:
            Dictionary containing dataset metadata and paths
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Create subdirectories for images
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        
        # Calculate samples per split
        train_samples = int(self.num_samples * train_ratio)
        val_samples = int(self.num_samples * val_ratio)
        test_samples = self.num_samples - train_samples - val_samples
        
        # Samples per class (roughly equal distribution)
        samples_per_class = {i: self.num_samples // self.num_classes for i in range(self.num_classes)}
        
        # Adjust for any remaining samples
        for i in range(self.num_samples % self.num_classes):
            samples_per_class[i] += 1
            
        # Generate data for each class
        all_data = []
        for class_idx in range(self.num_classes):
            class_data = self._generate_class_data(class_idx, samples_per_class[class_idx])
            all_data.extend(class_data)
            
        # Shuffle data
        random.shuffle(all_data)
        
        # Split data
        self.train_data = all_data[:train_samples]
        self.val_data = all_data[train_samples:train_samples+val_samples]
        self.test_data = all_data[train_samples+val_samples:]
        
        # Save data
        self._save_data()
        
        # Create metadata
        self.metadata = {
            "num_classes": self.num_classes,
            "num_samples": self.num_samples,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "class_names": self.class_names,
            "spurious_channels": self.spurious_channels,
            "image_size": self.image_size,
            "train_file": os.path.join(self.save_dir, "train.json"),
            "val_file": os.path.join(self.save_dir, "val.json"),
            "test_file": os.path.join(self.save_dir, "test.json"),
        }
        
        # Save metadata
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)
            
        return self.metadata
    
    def _generate_class_data(self, class_idx: int, num_samples: int) -> List[Dict]:
        """
        Generate data for a specific class with the configured spurious attributes.
        
        Args:
            class_idx: Index of the class to generate
            num_samples: Number of samples to generate for this class
            
        Returns:
            List of dictionaries containing sample data
        """
        class_data = []
        
        for i in range(num_samples):
            sample_id = f"{class_idx}_{i}"
            
            # Select spurious attributes for each channel
            spurious_attributes = {}
            for channel, config in self.spurious_channels.items():
                attrs = config["attributes"]
                p_alignment = config["p_alignment"][class_idx]
                
                # Assign "aligned" attribute with probability p_alignment
                if random.random() < p_alignment:
                    # Use the first attribute as the "aligned" one for simplicity
                    spurious_attributes[channel] = attrs[0]
                else:
                    # Select a random "non-aligned" attribute
                    spurious_attributes[channel] = random.choice(attrs[1:])
            
            # Generate image
            img_path = os.path.join("images", f"{sample_id}.png")
            full_img_path = os.path.join(self.save_dir, img_path)
            
            self._generate_image(
                full_img_path, 
                class_idx=class_idx,
                background=spurious_attributes["background"],
                color=spurious_attributes["color"],
                shape=spurious_attributes["shape"]
            )
            
            # Generate caption
            caption = self._generate_caption(
                class_name=self.class_names[class_idx],
                attributes=spurious_attributes
            )
            
            # Create sample data
            sample = {
                "id": sample_id,
                "class_idx": class_idx,
                "class_name": self.class_names[class_idx],
                "image_path": img_path,
                "caption": caption,
                "spurious_attributes": spurious_attributes
            }
            
            class_data.append(sample)
            
        return class_data
    
    def _generate_image(self, img_path: str, class_idx: int, background: str, color: str, shape: str):
        """
        Generate a synthetic image with the specified attributes.
        
        Args:
            img_path: Path to save the generated image
            class_idx: Class index
            background: Background texture/pattern
            color: Object color
            shape: Object shape
        """
        width, height = self.image_size
        
        # Create blank image with white background
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Apply background
        if background == "stripes":
            for i in range(0, height, 10):
                draw.rectangle([(0, i), (width, i + 5)], fill=(200, 200, 200))
        elif background == "dots":
            for i in range(0, width, 20):
                for j in range(0, height, 20):
                    draw.ellipse([(i, j), (i + 10, j + 10)], fill=(200, 200, 200))
        elif background == "checkered":
            for i in range(0, width, 20):
                for j in range(0, height, 20):
                    if (i // 20 + j // 20) % 2 == 0:
                        draw.rectangle([(i, j), (i + 20, j + 20)], fill=(200, 200, 200))
        # solid background is just the default white
        
        # Map color string to RGB
        color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0)
        }
        obj_color = color_map.get(color, (0, 0, 0))
        
        # Draw shape in the center of the image
        center_x, center_y = width // 2, height // 2
        size = min(width, height) // 3
        
        if shape == "circle":
            draw.ellipse(
                [(center_x - size, center_y - size), 
                 (center_x + size, center_y + size)], 
                fill=obj_color
            )
        elif shape == "square":
            draw.rectangle(
                [(center_x - size, center_y - size),
                 (center_x + size, center_y + size)],
                fill=obj_color
            )
        elif shape == "triangle":
            draw.polygon(
                [(center_x, center_y - size),
                 (center_x - size, center_y + size),
                 (center_x + size, center_y + size)],
                fill=obj_color
            )
        elif shape == "star":
            # Simplified star
            points = []
            for i in range(5):
                # Outer points
                angle = i * 2 * np.pi / 5 - np.pi / 2
                points.append((
                    center_x + int(size * np.cos(angle)),
                    center_y + int(size * np.sin(angle))
                ))
                # Inner points
                angle += np.pi / 5
                points.append((
                    center_x + int(size * 0.4 * np.cos(angle)),
                    center_y + int(size * 0.4 * np.sin(angle))
                ))
            draw.polygon(points, fill=obj_color)
            
        # Save the image
        img.save(img_path)
    
    def _generate_caption(self, class_name: str, attributes: Dict[str, str]) -> str:
        """
        Generate a caption for the image based on the class and spurious attributes.
        
        Args:
            class_name: Name of the class
            attributes: Dictionary of spurious attributes
            
        Returns:
            Generated caption string
        """
        templates = [
            f"A {attributes['color']} {class_name} with a {attributes['background']} background.",
            f"This is a {attributes['color']} {attributes['shape']} representing {class_name}.",
            f"The image shows a {attributes['color']} {class_name} {attributes['shape']} on a {attributes['background']} background."
        ]
        
        return random.choice(templates)
    
    def _save_data(self):
        """Save the generated data to JSON files for each split."""
        # Save train data
        with open(os.path.join(self.save_dir, "train.json"), "w") as f:
            json.dump(self.train_data, f, indent=2)
            
        # Save validation data
        with open(os.path.join(self.save_dir, "val.json"), "w") as f:
            json.dump(self.val_data, f, indent=2)
            
        # Save test data
        with open(os.path.join(self.save_dir, "test.json"), "w") as f:
            json.dump(self.test_data, f, indent=2)
    
    def visualize_samples(self, num_samples: int = 5, save_path: Optional[str] = None):
        """
        Visualize random samples from the dataset.
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Optional path to save the visualization
        """
        # Randomly select samples from train, val, and test
        all_samples = self.train_data + self.val_data + self.test_data
        if len(all_samples) == 0:
            print("No samples available. Generate the dataset first.")
            return
            
        samples = random.sample(all_samples, min(num_samples, len(all_samples)))
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))
        if num_samples == 1:
            axes = [axes]
            
        for i, sample in enumerate(samples):
            # Load image
            img_path = os.path.join(self.save_dir, sample["image_path"])
            img = Image.open(img_path)
            
            # Display image and caption
            axes[i].imshow(img)
            axes[i].set_title(f"Class: {sample['class_name']}")
            axes[i].set_xlabel(sample["caption"])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def perform_attribute_shuffling(self, data_split: str = "test", channel: str = "background") -> List[Dict]:
        """
        Create a shuffled version of the dataset by shuffling attributes in a specific channel.
        
        Args:
            data_split: Dataset split to use ("train", "val", or "test")
            channel: Spurious channel to shuffle
            
        Returns:
            List of samples with shuffled attributes
        """
        if data_split == "train":
            data = self.train_data
        elif data_split == "val":
            data = self.val_data
        else:
            data = self.test_data
            
        shuffled_data = []
        
        # Get all available attributes for the channel
        attributes = self.spurious_channels[channel]["attributes"]
        
        for sample in data:
            # Create a copy of the sample
            new_sample = sample.copy()
            
            # Choose a random attribute different from the current one
            current_attr = sample["spurious_attributes"][channel]
            other_attrs = [attr for attr in attributes if attr != current_attr]
            new_attr = random.choice(other_attrs)
            
            # Update the attribute
            new_sample["spurious_attributes"] = sample["spurious_attributes"].copy()
            new_sample["spurious_attributes"][channel] = new_attr
            
            # Generate a new image with the shuffled attribute
            img_path = f"images/shuffled_{new_sample['id']}_{channel}.png"
            full_img_path = os.path.join(self.save_dir, img_path)
            
            self._generate_image(
                full_img_path, 
                class_idx=sample["class_idx"],
                background=new_sample["spurious_attributes"]["background"],
                color=new_sample["spurious_attributes"]["color"],
                shape=new_sample["spurious_attributes"]["shape"]
            )
            
            # Update the image path
            new_sample["image_path"] = img_path
            
            # Generate a new caption
            new_sample["caption"] = self._generate_caption(
                class_name=sample["class_name"],
                attributes=new_sample["spurious_attributes"]
            )
            
            # Add the shuffled sample
            shuffled_data.append(new_sample)
            
        return shuffled_data


# Example usage
if __name__ == "__main__":
    # Define class names for a toy dataset
    class_names = ["dog", "cat", "car", "tree", "flower", 
                   "building", "airplane", "bird", "boat", "bicycle"]
    
    # Initialize with smaller number of samples for testing
    spurgen = SpurGen(
        num_classes=10,
        num_samples=1000,  # Reduced for quick testing
        class_names=class_names,
        save_dir="data"
    )
    
    # Generate dataset
    metadata = spurgen.generate_dataset()
    print(f"Generated dataset with {metadata['num_samples']} samples")
    
    # Visualize samples
    spurgen.visualize_samples(num_samples=5, save_path="sample_visualization.png")
    
    # Create shuffled version for testing SpurSensitivity
    shuffled_background = spurgen.perform_attribute_shuffling(channel="background")
    print(f"Created shuffled dataset with {len(shuffled_background)} samples")