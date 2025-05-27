#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WaterbirdsDataset: A dataset based on the Waterbirds dataset from Sagawa et al. (2020),
where land birds and water birds are spuriously correlated with land and water backgrounds.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
import random
import json


class WaterbirdsDataset(Dataset):
    """
    Waterbirds dataset with land/water birds on land/water backgrounds.
    Simulated version based on the original Waterbirds dataset.
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the Waterbirds dataset.
        
        Args:
            data_dir: Directory to store/load the dataset
            split: Dataset split ('train', 'val', 'test', or 'ood_test')
            transform: Image transformation pipeline
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Create dataset directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if metadata exists, otherwise create it
        metadata_path = self.data_dir / 'metadata.json'
        if not metadata_path.exists():
            self._create_synthetic_waterbirds()
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Filter by split
        self.metadata = [item for item in metadata if item['split'] == split]
        
        # Extract data
        self.data = {
            'vision': [item['image_path'] for item in self.metadata],
            'labels': [item['y'] for item in self.metadata],  # Bird type (0: landbird, 1: waterbird)
            'group_labels': [item['group'] for item in self.metadata],  # Group (bird-background combination)
            'backgrounds': [item['place'] for item in self.metadata],  # Background (0: land, 1: water)
            'perturbed': {
                'vision': [item.get('perturbed_image_path') for item in self.metadata]
            },
            'counterfactual': {
                'vision': [item.get('counterfactual_image_path') for item in self.metadata]
            }
        }
        
        # Add text descriptions to create a multi-modal dataset
        self.data['text'] = [self._generate_description(item) for item in self.metadata]
        self.data['perturbed']['text'] = [self._generate_perturbed_description(item) for item in self.metadata]
        self.data['counterfactual']['text'] = [self._generate_counterfactual_description(item) for item in self.metadata]
        
        # Filter out None values in perturbed and counterfactual data
        self.data['perturbed']['vision'] = [p for p in self.data['perturbed']['vision'] if p is not None]
        self.data['counterfactual']['vision'] = [p for p in self.data['counterfactual']['vision'] if p is not None]
    
    def _create_synthetic_waterbirds(self):
        """
        Create a synthetic version of the Waterbirds dataset.
        """
        # Create image directories
        images_dir = self.data_dir / 'images'
        perturbed_dir = self.data_dir / 'perturbed'
        counterfactual_dir = self.data_dir / 'counterfactual'
        
        for directory in [images_dir, perturbed_dir, counterfactual_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Define bird colors and background colors
        landbird_colors = [(139, 69, 19), (160, 82, 45), (205, 133, 63), (210, 105, 30)]  # Brown tones
        waterbird_colors = [(0, 0, 139), (0, 0, 205), (65, 105, 225), (70, 130, 180)]  # Blue tones
        
        land_bg_colors = [(34, 139, 34), (0, 128, 0), (0, 100, 0), (85, 107, 47)]  # Green tones
        water_bg_colors = [(0, 191, 255), (30, 144, 255), (100, 149, 237), (135, 206, 235)]  # Blue tones
        
        # Create metadata
        metadata = []
        
        # Number of samples per split
        num_train = 2400
        num_val = 300
        num_test = 300
        num_ood_test = 300
        
        splits = {
            'train': num_train,
            'val': num_val,
            'test': num_test,
            'ood_test': num_ood_test
        }
        
        # Spurious correlation strengths
        # In train/val/test: 95% of landbirds on land, 95% of waterbirds on water
        # In OOD test: 5% of landbirds on land, 5% of waterbirds on water (inverted)
        correlation_strengths = {
            'train': 0.95,
            'val': 0.95,
            'test': 0.95,
            'ood_test': 0.05
        }
        
        # Create samples for each split
        for split, num_samples in splits.items():
            # Equal number of landbirds and waterbirds
            for bird_type in [0, 1]:  # 0: landbird, 1: waterbird
                split_samples = num_samples // 2
                
                for i in range(split_samples):
                    # Determine if this sample follows the spurious correlation
                    follows_correlation = random.random() < correlation_strengths[split]
                    
                    # Assign background based on correlation
                    if bird_type == 0:  # landbird
                        bg_type = 0 if follows_correlation else 1  # 0: land, 1: water
                    else:  # waterbird
                        bg_type = 1 if follows_correlation else 0  # 1: water, 0: land
                    
                    # Assign group based on bird-background combination
                    # Group 0: landbird on land (majority)
                    # Group 1: waterbird on water (majority)
                    # Group 2: landbird on water (minority)
                    # Group 3: waterbird on land (minority)
                    group = bird_type * 1 + bg_type * (1 - 2 * bird_type)
                    
                    # Create synthetic images
                    image_path, perturbed_path, counterfactual_path = self._create_synthetic_waterbird_image(
                        bird_type=bird_type,
                        bg_type=bg_type,
                        landbird_colors=landbird_colors,
                        waterbird_colors=waterbird_colors,
                        land_bg_colors=land_bg_colors,
                        water_bg_colors=water_bg_colors,
                        images_dir=images_dir,
                        perturbed_dir=perturbed_dir,
                        counterfactual_dir=counterfactual_dir,
                        index=len(metadata)
                    )
                    
                    # Add metadata
                    metadata.append({
                        'id': len(metadata),
                        'image_path': str(image_path),
                        'perturbed_image_path': str(perturbed_path) if split == 'train' else None,
                        'counterfactual_image_path': str(counterfactual_path) if split == 'train' else None,
                        'y': bird_type,  # Bird type (0: landbird, 1: waterbird)
                        'place': bg_type,  # Background (0: land, 1: water)
                        'group': group,  # Group (bird-background combination)
                        'split': split
                    })
        
        # Save metadata
        with open(self.data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def _create_synthetic_waterbird_image(
        self, 
        bird_type, 
        bg_type, 
        landbird_colors, 
        waterbird_colors, 
        land_bg_colors, 
        water_bg_colors, 
        images_dir, 
        perturbed_dir, 
        counterfactual_dir, 
        index
    ):
        """
        Create a synthetic waterbird image.
        
        Args:
            bird_type: Bird type (0: landbird, 1: waterbird)
            bg_type: Background type (0: land, 1: water)
            landbird_colors: List of colors for landbirds
            waterbird_colors: List of colors for waterbirds
            land_bg_colors: List of colors for land backgrounds
            water_bg_colors: List of colors for water backgrounds
            images_dir: Directory to save images
            perturbed_dir: Directory to save perturbed images
            counterfactual_dir: Directory to save counterfactual images
            index: Sample index
            
        Returns:
            image_path: Path to the created image
            perturbed_path: Path to the perturbed image
            counterfactual_path: Path to the counterfactual image
        """
        # Create a blank image (224x224)
        image_size = 224
        image = Image.new('RGB', (image_size, image_size), (255, 255, 255))
        
        # Draw background
        bg_color = random.choice(land_bg_colors if bg_type == 0 else water_bg_colors)
        for y in range(image_size):
            for x in range(image_size):
                # Add some noise to the background color
                noise = random.randint(-20, 20)
                r = max(0, min(255, bg_color[0] + noise))
                g = max(0, min(255, bg_color[1] + noise))
                b = max(0, min(255, bg_color[2] + noise))
                image.putpixel((x, y), (r, g, b))
        
        # Draw bird (simplified as a distinctive shape)
        bird_color = random.choice(landbird_colors if bird_type == 0 else waterbird_colors)
        
        if bird_type == 0:  # landbird (draw as triangle)
            points = [
                (image_size // 2, image_size // 3),
                (image_size // 3, image_size * 2 // 3),
                (image_size * 2 // 3, image_size * 2 // 3)
            ]
            draw = Image.new('RGB', (image_size, image_size), (0, 0, 0, 0))
            draw_obj = ImageDraw.Draw(draw)
            draw_obj.polygon(points, fill=bird_color)
            image.paste(draw, (0, 0), mask=draw.convert('L'))
        else:  # waterbird (draw as circle)
            draw = Image.new('RGB', (image_size, image_size), (0, 0, 0, 0))
            draw_obj = ImageDraw.Draw(draw)
            center = (image_size // 2, image_size // 2)
            radius = image_size // 6
            draw_obj.ellipse(
                [(center[0] - radius, center[1] - radius), 
                 (center[0] + radius, center[1] + radius)], 
                fill=bird_color
            )
            image.paste(draw, (0, 0), mask=draw.convert('L'))
        
        # Save original image
        image_path = images_dir / f"waterbird_{index}.jpg"
        image.save(image_path)
        
        # Create perturbed image (same bird, slightly different background)
        perturbed_image = Image.new('RGB', (image_size, image_size), (255, 255, 255))
        
        # Draw perturbed background (same type but different color)
        perturbed_bg_color = random.choice(land_bg_colors if bg_type == 0 else water_bg_colors)
        while perturbed_bg_color == bg_color:
            perturbed_bg_color = random.choice(land_bg_colors if bg_type == 0 else water_bg_colors)
        
        for y in range(image_size):
            for x in range(image_size):
                # Add some noise to the background color
                noise = random.randint(-20, 20)
                r = max(0, min(255, perturbed_bg_color[0] + noise))
                g = max(0, min(255, perturbed_bg_color[1] + noise))
                b = max(0, min(255, perturbed_bg_color[2] + noise))
                perturbed_image.putpixel((x, y), (r, g, b))
        
        # Draw the same bird
        if bird_type == 0:  # landbird (triangle)
            draw = Image.new('RGB', (image_size, image_size), (0, 0, 0, 0))
            draw_obj = ImageDraw.Draw(draw)
            draw_obj.polygon(points, fill=bird_color)
            perturbed_image.paste(draw, (0, 0), mask=draw.convert('L'))
        else:  # waterbird (circle)
            draw = Image.new('RGB', (image_size, image_size), (0, 0, 0, 0))
            draw_obj = ImageDraw.Draw(draw)
            draw_obj.ellipse(
                [(center[0] - radius, center[1] - radius), 
                 (center[0] + radius, center[1] + radius)], 
                fill=bird_color
            )
            perturbed_image.paste(draw, (0, 0), mask=draw.convert('L'))
        
        # Save perturbed image
        perturbed_path = perturbed_dir / f"waterbird_{index}_perturbed.jpg"
        perturbed_image.save(perturbed_path)
        
        # Create counterfactual image (same bird, opposite background)
        counterfactual_image = Image.new('RGB', (image_size, image_size), (255, 255, 255))
        
        # Draw counterfactual background (opposite type)
        cf_bg_color = random.choice(water_bg_colors if bg_type == 0 else land_bg_colors)
        
        for y in range(image_size):
            for x in range(image_size):
                # Add some noise to the background color
                noise = random.randint(-20, 20)
                r = max(0, min(255, cf_bg_color[0] + noise))
                g = max(0, min(255, cf_bg_color[1] + noise))
                b = max(0, min(255, cf_bg_color[2] + noise))
                counterfactual_image.putpixel((x, y), (r, g, b))
        
        # Draw the same bird
        if bird_type == 0:  # landbird (triangle)
            draw = Image.new('RGB', (image_size, image_size), (0, 0, 0, 0))
            draw_obj = ImageDraw.Draw(draw)
            draw_obj.polygon(points, fill=bird_color)
            counterfactual_image.paste(draw, (0, 0), mask=draw.convert('L'))
        else:  # waterbird (circle)
            draw = Image.new('RGB', (image_size, image_size), (0, 0, 0, 0))
            draw_obj = ImageDraw.Draw(draw)
            draw_obj.ellipse(
                [(center[0] - radius, center[1] - radius), 
                 (center[0] + radius, center[1] + radius)], 
                fill=bird_color
            )
            counterfactual_image.paste(draw, (0, 0), mask=draw.convert('L'))
        
        # Save counterfactual image
        counterfactual_path = counterfactual_dir / f"waterbird_{index}_counterfactual.jpg"
        counterfactual_image.save(counterfactual_path)
        
        return image_path, perturbed_path, counterfactual_path
    
    def _generate_description(self, metadata):
        """
        Generate a textual description for a waterbird image.
        
        Args:
            metadata: Image metadata
            
        Returns:
            description: Textual description
        """
        bird_type = metadata['y']
        background = metadata['place']
        
        # Bird species (causal feature)
        land_birds = ["sparrow", "robin", "cardinal", "finch", "woodpecker", "hawk", "eagle"]
        water_birds = ["duck", "swan", "pelican", "seagull", "heron", "flamingo", "goose"]
        
        # Background descriptions (spurious feature)
        land_backgrounds = ["on a grassy field", "in a forest", "on a mountain", "in a meadow"]
        water_backgrounds = ["on a lake", "by the ocean", "near a river", "in a pond"]
        
        # Select bird and background
        bird = random.choice(land_birds if bird_type == 0 else water_birds)
        bg = random.choice(land_backgrounds if background == 0 else water_backgrounds)
        
        # Generate description
        description = f"A {bird} {bg}."
        
        return description
    
    def _generate_perturbed_description(self, metadata):
        """
        Generate a perturbed textual description (same bird, different background wording).
        
        Args:
            metadata: Image metadata
            
        Returns:
            description: Perturbed textual description
        """
        bird_type = metadata['y']
        background = metadata['place']
        
        # Bird species (causal feature)
        land_birds = ["sparrow", "robin", "cardinal", "finch", "woodpecker", "hawk", "eagle"]
        water_birds = ["duck", "swan", "pelican", "seagull", "heron", "flamingo", "goose"]
        
        # Alternative background descriptions (same type but different wording)
        alt_land_backgrounds = ["on the ground", "surrounded by trees", "in a woodland area", "with hills in the background"]
        alt_water_backgrounds = ["surrounded by water", "near the shore", "with waves in the background", "in shallow water"]
        
        # Select bird and alternative background
        bird = random.choice(land_birds if bird_type == 0 else water_birds)
        bg = random.choice(alt_land_backgrounds if background == 0 else alt_water_backgrounds)
        
        # Generate description
        description = f"A {bird} {bg}."
        
        return description
    
    def _generate_counterfactual_description(self, metadata):
        """
        Generate a counterfactual textual description (same bird, opposite background).
        
        Args:
            metadata: Image metadata
            
        Returns:
            description: Counterfactual textual description
        """
        bird_type = metadata['y']
        background = metadata['place']
        
        # Bird species (causal feature)
        land_birds = ["sparrow", "robin", "cardinal", "finch", "woodpecker", "hawk", "eagle"]
        water_birds = ["duck", "swan", "pelican", "seagull", "heron", "flamingo", "goose"]
        
        # Background descriptions (spurious feature)
        land_backgrounds = ["on a grassy field", "in a forest", "on a mountain", "in a meadow"]
        water_backgrounds = ["on a lake", "by the ocean", "near a river", "in a pond"]
        
        # Select bird and OPPOSITE background
        bird = random.choice(land_birds if bird_type == 0 else water_birds)
        bg = random.choice(water_backgrounds if background == 0 else land_backgrounds)
        
        # Generate description
        description = f"A {bird} {bg}."
        
        return description
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            sample: Dictionary containing the sample data
        """
        # Image path
        image_path = self.data['vision'][idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform if available
        if self.transform:
            image = self.transform(image)
        
        return {
            'vision': image,
            'text': self.data['text'][idx],
            'labels': self.data['labels'][idx],
            'group_labels': self.data['group_labels'][idx],
            'backgrounds': self.data['backgrounds'][idx],
            'perturbed': {
                'vision': None if idx >= len(self.data['perturbed']['vision']) else self.data['perturbed']['vision'][idx],
                'text': self.data['perturbed']['text'][idx]
            },
            'counterfactual': {
                'vision': None if idx >= len(self.data['counterfactual']['vision']) else self.data['counterfactual']['vision'][idx],
                'text': self.data['counterfactual']['text'][idx]
            }
        }