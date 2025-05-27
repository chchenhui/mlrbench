#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create synthetic multi-modal datasets with controlled spurious correlations.
"""

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import random
import string
from pathlib import Path

def create_synthetic_multimodal_dataset(data_config, split='train', num_samples=1000):
    """
    Create a synthetic multi-modal dataset with controlled spurious correlations.
    
    Args:
        data_config: Dictionary containing dataset configuration parameters
        split: Dataset split ('train', 'val', 'test', or 'ood_test')
        num_samples: Number of samples to generate
        
    Returns:
        data: Dictionary containing the generated dataset
    """
    # Set random seed for reproducibility (different for each split)
    if split == 'train':
        np.random.seed(42)
    elif split == 'val':
        np.random.seed(43)
    elif split == 'test':
        np.random.seed(44)
    elif split == 'ood_test':
        np.random.seed(45)
    
    # Get parameters from config
    num_classes = data_config.get('num_classes', 2)
    modalities = data_config.get('modalities', ['vision', 'text'])
    spurious_correlation_strength = data_config.get('spurious_correlation_strength', 0.9)
    
    # For OOD test set, invert the spurious correlation
    if split == 'ood_test':
        spurious_correlation_strength = 1.0 - spurious_correlation_strength
    
    # Initialize data dictionary
    data = {
        'labels': [],
        'group_labels': [],
        'vision': [] if 'vision' in modalities else None,
        'text': [] if 'text' in modalities else None,
        'perturbed': {
            'vision': [] if 'vision' in modalities else None,
            'text': [] if 'text' in modalities else None
        },
        'counterfactual': {
            'vision': [] if 'vision' in modalities else None,
            'text': [] if 'text' in modalities else None
        }
    }
    
    # Generate data for each class
    for class_idx in range(num_classes):
        # Number of samples per class
        class_samples = num_samples // num_classes
        
        # Generate labels
        data['labels'].extend([class_idx] * class_samples)
        
        # Generate vision data if needed
        if 'vision' in modalities:
            # For each class, the causal feature is the color of the central shape
            # The spurious feature is the background texture
            
            # Define causal features (different colors for different classes)
            causal_colors = [
                (255, 0, 0),      # Red for class 0
                (0, 0, 255),      # Blue for class 1
                (0, 255, 0),      # Green for class 2
                (255, 255, 0)     # Yellow for class 3
            ]
            
            # Define spurious features (different background textures)
            spurious_patterns = [
                'stripes',        # Stripes for class 0
                'dots',           # Dots for class 1
                'grid',           # Grid for class 2
                'waves'           # Waves for class 3
            ]
            
            for i in range(class_samples):
                # Determine if this sample follows the spurious correlation
                follows_spurious = np.random.random() < spurious_correlation_strength
                
                # Choose the causal feature (always aligned with the class)
                causal_color = causal_colors[class_idx % len(causal_colors)]
                
                # Choose the spurious feature (aligned with class if follows_spurious)
                if follows_spurious:
                    spurious_pattern = spurious_patterns[class_idx % len(spurious_patterns)]
                    group_label = class_idx  # Aligned with class
                else:
                    # Choose a random different pattern
                    choices = [p for p in range(len(spurious_patterns)) if p != class_idx % len(spurious_patterns)]
                    random_idx = np.random.choice(choices)
                    spurious_pattern = spurious_patterns[random_idx]
                    group_label = random_idx + num_classes  # Misaligned with class
                
                # Generate the image
                image = create_synthetic_image(causal_color, spurious_pattern)
                data['vision'].append(np.array(image))
                
                # Generate perturbed image (same class features, different spurious features)
                perturbed_pattern = np.random.choice([p for p in spurious_patterns if p != spurious_pattern])
                perturbed_image = create_synthetic_image(causal_color, perturbed_pattern)
                data['perturbed']['vision'].append(np.array(perturbed_image))
                
                # Generate counterfactual image (intervene on spurious feature)
                cf_pattern = np.random.choice([p for p in spurious_patterns if p != spurious_pattern])
                cf_image = create_synthetic_image(causal_color, cf_pattern)
                data['counterfactual']['vision'].append(np.array(cf_image))
                
                # Store group label
                data['group_labels'].append(group_label)
        
        # Generate text data if needed
        if 'text' in modalities:
            # For each class, the causal feature is the topic/content
            # The spurious feature is the style/tone
            
            # Define causal features (different topics for different classes)
            causal_topics = [
                "animals",        # Topic for class 0
                "technology",     # Topic for class 1
                "sports",         # Topic for class 2
                "food"            # Topic for class 3
            ]
            
            # Define spurious features (different styles)
            spurious_styles = [
                "formal",         # Style for class 0
                "casual",         # Style for class 1
                "enthusiastic",   # Style for class 2
                "technical"       # Style for class 3
            ]
            
            for i in range(class_samples):
                # Determine if this sample follows the spurious correlation
                follows_spurious = np.random.random() < spurious_correlation_strength
                
                # Choose the causal feature (always aligned with the class)
                causal_topic = causal_topics[class_idx % len(causal_topics)]
                
                # Choose the spurious feature (aligned with class if follows_spurious)
                if follows_spurious:
                    spurious_style = spurious_styles[class_idx % len(spurious_styles)]
                else:
                    # Choose a random different style
                    choices = [s for s in range(len(spurious_styles)) if s != class_idx % len(spurious_styles)]
                    random_idx = np.random.choice(choices)
                    spurious_style = spurious_styles[random_idx]
                
                # Generate the text
                text = create_synthetic_text(causal_topic, spurious_style)
                data['text'].append(text)
                
                # Generate perturbed text (same topic, different style)
                perturbed_style = np.random.choice([s for s in spurious_styles if s != spurious_style])
                perturbed_text = create_synthetic_text(causal_topic, perturbed_style)
                data['perturbed']['text'].append(perturbed_text)
                
                # Generate counterfactual text (intervene on spurious feature)
                cf_style = np.random.choice([s for s in spurious_styles if s != spurious_style])
                cf_text = create_synthetic_text(causal_topic, cf_style)
                data['counterfactual']['text'].append(cf_text)
    
    # Shuffle data while keeping alignment between modalities
    indices = np.arange(len(data['labels']))
    np.random.shuffle(indices)
    
    shuffled_data = {
        'labels': [data['labels'][i] for i in indices],
        'group_labels': [data['group_labels'][i] for i in indices]
    }
    
    if 'vision' in modalities:
        shuffled_data['vision'] = [data['vision'][i] for i in indices]
        shuffled_data['perturbed'] = {'vision': [data['perturbed']['vision'][i] for i in indices]}
        shuffled_data['counterfactual'] = {'vision': [data['counterfactual']['vision'][i] for i in indices]}
    
    if 'text' in modalities:
        shuffled_data['text'] = [data['text'][i] for i in indices]
        if 'perturbed' not in shuffled_data:
            shuffled_data['perturbed'] = {}
        if 'counterfactual' not in shuffled_data:
            shuffled_data['counterfactual'] = {}
        shuffled_data['perturbed']['text'] = [data['perturbed']['text'][i] for i in indices]
        shuffled_data['counterfactual']['text'] = [data['counterfactual']['text'][i] for i in indices]
    
    return shuffled_data


def create_synthetic_image(color, pattern, size=224):
    """
    Create a synthetic image with a colored shape and a patterned background.
    
    Args:
        color: RGB color tuple for the central shape
        pattern: Background pattern type ('stripes', 'dots', 'grid', or 'waves')
        size: Image size
        
    Returns:
        image: PIL Image
    """
    # Create a white background
    image = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Draw background pattern
    if pattern == 'stripes':
        # Horizontal stripes
        stripe_width = size // 10
        for y in range(0, size, stripe_width * 2):
            draw.rectangle([(0, y), (size, y + stripe_width)], fill=(240, 240, 240))
    
    elif pattern == 'dots':
        # Random dots
        num_dots = 100
        dot_size = size // 30
        for _ in range(num_dots):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            draw.ellipse([(x - dot_size, y - dot_size), (x + dot_size, y + dot_size)], fill=(240, 240, 240))
    
    elif pattern == 'grid':
        # Grid lines
        grid_spacing = size // 10
        for i in range(0, size, grid_spacing):
            draw.line([(i, 0), (i, size)], fill=(240, 240, 240), width=2)
            draw.line([(0, i), (size, i)], fill=(240, 240, 240), width=2)
    
    elif pattern == 'waves':
        # Wavy lines
        wave_count = 5
        amplitude = size // 20
        line_spacing = size // wave_count
        for i in range(0, size, line_spacing):
            points = []
            for x in range(0, size, 5):
                y = i + amplitude * np.sin(x * 6 * np.pi / size)
                points.append((x, y))
            draw.line(points, fill=(240, 240, 240), width=3)
    
    # Draw central shape (the causal feature) - a circle
    center = size // 2
    radius = size // 5
    draw.ellipse([(center - radius, center - radius), (center + radius, center + radius)], fill=color)
    
    return image


def create_synthetic_text(topic, style):
    """
    Create synthetic text with a specific topic and style.
    
    Args:
        topic: Text topic (e.g., 'animals', 'technology')
        style: Text style (e.g., 'formal', 'casual')
        
    Returns:
        text: Generated text
    """
    # Topic-specific vocabulary
    topic_vocab = {
        'animals': [
            "cat", "dog", "elephant", "lion", "tiger", "bear", "zebra", "giraffe", 
            "monkey", "gorilla", "panda", "koala", "kangaroo", "wolf", "fox"
        ],
        'technology': [
            "computer", "smartphone", "internet", "robot", "algorithm", "software", 
            "hardware", "device", "gadget", "program", "app", "code", "data", "cloud"
        ],
        'sports': [
            "football", "basketball", "tennis", "golf", "baseball", "soccer", "volleyball", 
            "swimming", "running", "cycling", "hockey", "boxing", "skiing", "surfing"
        ],
        'food': [
            "pizza", "burger", "pasta", "sushi", "salad", "bread", "cheese", "vegetables", 
            "fruits", "meat", "fish", "rice", "noodles", "dessert", "cake"
        ]
    }
    
    # Style-specific templates and vocabulary
    style_templates = {
        'formal': [
            "An examination of {topic} reveals important insights.",
            "Recent studies have shown that {topic} demonstrate remarkable characteristics.",
            "It is worth noting that {topic} serve an essential function in their environment.",
            "The analysis of {topic} provides valuable information for further research.",
            "Multiple factors contribute to the significance of {topic} in contemporary society."
        ],
        'casual': [
            "Hey, check out these cool {topic}!",
            "I really like how {topic} are so awesome.",
            "Have you seen the latest {topic}? They're pretty amazing!",
            "So, I was thinking about {topic} the other day...",
            "These {topic} are my favorite things ever!"
        ],
        'enthusiastic': [
            "WOW!! These {topic} are AMAZING!!",
            "I absolutely LOVE {topic}! They're the BEST!!!",
            "Nothing gets me more excited than awesome {topic}!!!",
            "OMG! You won't BELIEVE how incredible these {topic} are!!",
            "I'm OBSESSED with these fantastic {topic}!!!"
        ],
        'technical': [
            "The functional characteristics of {topic} can be systematically categorized.",
            "A quantitative analysis of {topic} reveals correlation with environmental parameters.",
            "Implementation procedures for {topic} require specific configuration protocols.",
            "The optimization algorithm for {topic} utilizes advanced computational methods.",
            "Technical specifications of {topic} include multifaceted operational components."
        ]
    }
    
    # Select random words from the topic vocabulary
    topic_words = np.random.choice(topic_vocab[topic], size=3, replace=False)
    topic_text = " and ".join(topic_words)
    
    # Select a random template for the style
    template = np.random.choice(style_templates[style])
    
    # Fill in the template with topic words
    text = template.format(topic=topic_text)
    
    return text