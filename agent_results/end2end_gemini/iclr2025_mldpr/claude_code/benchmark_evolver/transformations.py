"""
Transformation operations for evolving benchmark instances.
"""

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps

class ParameterizedTransformation:
    """Base class for parameterized transformations."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def __call__(self, img):
        raise NotImplementedError("Subclasses must implement __call__")
    
    def mutate(self, mutation_strength=0.2):
        """Mutate the parameters of this transformation."""
        raise NotImplementedError("Subclasses must implement mutate")
    
    def crossover(self, other):
        """Crossover with another transformation of the same type."""
        raise NotImplementedError("Subclasses must implement crossover")
    
    def get_params(self):
        """Get the parameters of this transformation."""
        return self.params
    
    def set_params(self, params):
        """Set the parameters of this transformation."""
        self.params = params
    
    def __repr__(self):
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.__class__.__name__}({params_str})"


class RotationTransform(ParameterizedTransformation):
    """Apply rotation to an image."""
    
    def __init__(self, angle=0):
        super().__init__(angle=angle)
    
    def __call__(self, img):
        return TF.rotate(img, self.params['angle'])
    
    def mutate(self, mutation_strength=0.2):
        # Mutate angle
        self.params['angle'] += np.random.normal(0, 30 * mutation_strength)
        self.params['angle'] = self.params['angle'] % 360
        return self
    
    def crossover(self, other):
        if not isinstance(other, RotationTransform):
            return self
        
        # Blend angles
        child = RotationTransform()
        alpha = random.random()
        child.params['angle'] = alpha * self.params['angle'] + (1 - alpha) * other.params['angle']
        return child


class ColorJitterTransform(ParameterizedTransformation):
    """Apply color jitter transformation."""
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        return self.transform(img)
    
    def mutate(self, mutation_strength=0.2):
        # Mutate parameters
        self.params['brightness'] = max(0, self.params['brightness'] + np.random.normal(0, 0.2 * mutation_strength))
        self.params['contrast'] = max(0, self.params['contrast'] + np.random.normal(0, 0.2 * mutation_strength))
        self.params['saturation'] = max(0, self.params['saturation'] + np.random.normal(0, 0.2 * mutation_strength))
        self.params['hue'] = min(0.5, max(0, self.params['hue'] + np.random.normal(0, 0.05 * mutation_strength)))
        
        # Update transform
        self.transform = T.ColorJitter(
            brightness=self.params['brightness'],
            contrast=self.params['contrast'],
            saturation=self.params['saturation'],
            hue=self.params['hue']
        )
        
        return self
    
    def crossover(self, other):
        if not isinstance(other, ColorJitterTransform):
            return self
        
        # Create a child with blended parameters
        child = ColorJitterTransform()
        alpha = random.random()
        for param in ['brightness', 'contrast', 'saturation', 'hue']:
            child.params[param] = alpha * self.params[param] + (1 - alpha) * other.params[param]
        
        # Update child's transform
        child.transform = T.ColorJitter(
            brightness=child.params['brightness'],
            contrast=child.params['contrast'],
            saturation=child.params['saturation'],
            hue=child.params['hue']
        )
        
        return child


class GaussianBlurTransform(ParameterizedTransformation):
    """Apply Gaussian blur."""
    
    def __init__(self, radius=2):
        super().__init__(radius=radius)
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL
            img = TF.to_pil_image(img)
        
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=self.params['radius']))
        
        if isinstance(img, torch.Tensor):
            # Convert back to tensor
            blurred_img = TF.to_tensor(blurred_img)
        
        return blurred_img
    
    def mutate(self, mutation_strength=0.2):
        # Mutate radius
        self.params['radius'] = max(0, self.params['radius'] + np.random.normal(0, 1 * mutation_strength))
        return self
    
    def crossover(self, other):
        if not isinstance(other, GaussianBlurTransform):
            return self
        
        # Blend radius
        child = GaussianBlurTransform()
        alpha = random.random()
        child.params['radius'] = alpha * self.params['radius'] + (1 - alpha) * other.params['radius']
        return child


class PerspectiveTransform(ParameterizedTransformation):
    """Apply perspective transformation."""
    
    def __init__(self, distortion_scale=0.2):
        super().__init__(distortion_scale=distortion_scale)
        self.transform = T.RandomPerspective(
            distortion_scale=distortion_scale,
            p=1.0
        )
    
    def __call__(self, img):
        return self.transform(img)
    
    def mutate(self, mutation_strength=0.2):
        # Mutate distortion scale
        self.params['distortion_scale'] = min(0.9, max(0.1, 
                                           self.params['distortion_scale'] + np.random.normal(0, 0.1 * mutation_strength)))
        
        # Update transform
        self.transform = T.RandomPerspective(
            distortion_scale=self.params['distortion_scale'],
            p=1.0
        )
        
        return self
    
    def crossover(self, other):
        if not isinstance(other, PerspectiveTransform):
            return self
        
        # Blend distortion scale
        child = PerspectiveTransform()
        alpha = random.random()
        child.params['distortion_scale'] = alpha * self.params['distortion_scale'] + (1 - alpha) * other.params['distortion_scale']
        
        # Update child's transform
        child.transform = T.RandomPerspective(
            distortion_scale=child.params['distortion_scale'],
            p=1.0
        )
        
        return child


class NoisyTransform(ParameterizedTransformation):
    """Add random noise to image."""
    
    def __init__(self, noise_level=0.2, noise_type='gaussian'):
        super().__init__(noise_level=noise_level, noise_type=noise_type)
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            # Convert PIL to tensor
            img = TF.to_tensor(img)
        
        if self.params['noise_type'] == 'gaussian':
            # Add Gaussian noise
            noise = torch.randn_like(img) * self.params['noise_level']
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
        elif self.params['noise_type'] == 'salt_and_pepper':
            # Add salt and pepper noise
            noisy_img = img.clone()
            noise_mask = torch.rand_like(img) < self.params['noise_level']
            salt_vs_pepper = torch.rand_like(img) < 0.5
            # salt: random pixels set to 1
            noisy_img[noise_mask & salt_vs_pepper] = 1
            # pepper: random pixels set to 0
            noisy_img[noise_mask & ~salt_vs_pepper] = 0
        elif self.params['noise_type'] == 'poisson':
            # Poisson noise
            noisy_img = torch.poisson(img * 255.0 * self.params['noise_level']) / (255.0 * self.params['noise_level'])
            noisy_img = torch.clamp(noisy_img, 0, 1)
        else:
            noisy_img = img
        
        if isinstance(img, Image.Image):
            # Convert back to PIL
            noisy_img = TF.to_pil_image(noisy_img)
        
        return noisy_img
    
    def mutate(self, mutation_strength=0.2):
        # Mutate noise level
        self.params['noise_level'] = min(0.9, max(0.05, 
                                       self.params['noise_level'] + np.random.normal(0, 0.1 * mutation_strength)))
        
        # Occasionally change noise type
        if random.random() < 0.1 * mutation_strength:
            self.params['noise_type'] = random.choice(['gaussian', 'salt_and_pepper', 'poisson'])
        
        return self
    
    def crossover(self, other):
        if not isinstance(other, NoisyTransform):
            return self
        
        # Create child
        child = NoisyTransform()
        
        # Blend noise level
        alpha = random.random()
        child.params['noise_level'] = alpha * self.params['noise_level'] + (1 - alpha) * other.params['noise_level']
        
        # Choose one parent's noise type
        child.params['noise_type'] = self.params['noise_type'] if random.random() < 0.5 else other.params['noise_type']
        
        return child


class ElasticTransform(ParameterizedTransformation):
    """Apply elastic transformation."""
    
    def __init__(self, alpha=40, sigma=5):
        super().__init__(alpha=alpha, sigma=sigma)
        self.transform = T.ElasticTransform(alpha=alpha, sigma=sigma)
    
    def __call__(self, img):
        return self.transform(img)
    
    def mutate(self, mutation_strength=0.2):
        # Mutate parameters
        self.params['alpha'] = max(1, self.params['alpha'] + np.random.normal(0, 10 * mutation_strength))
        self.params['sigma'] = max(1, self.params['sigma'] + np.random.normal(0, 2 * mutation_strength))
        
        # Update transform
        self.transform = T.ElasticTransform(
            alpha=self.params['alpha'],
            sigma=self.params['sigma']
        )
        
        return self
    
    def crossover(self, other):
        if not isinstance(other, ElasticTransform):
            return self
        
        # Create child with blended parameters
        child = ElasticTransform()
        alpha = random.random()
        child.params['alpha'] = alpha * self.params['alpha'] + (1 - alpha) * other.params['alpha']
        child.params['sigma'] = alpha * self.params['sigma'] + (1 - alpha) * other.params['sigma']
        
        # Update child's transform
        child.transform = T.ElasticTransform(
            alpha=child.params['alpha'],
            sigma=child.params['sigma']
        )
        
        return child


# Dictionary of available transformations with their parameter ranges
AVAILABLE_TRANSFORMATIONS = {
    'rotation': {
        'class': RotationTransform,
        'param_ranges': {
            'angle': (0, 360)
        }
    },
    'color_jitter': {
        'class': ColorJitterTransform,
        'param_ranges': {
            'brightness': (0, 0.5),
            'contrast': (0, 0.5),
            'saturation': (0, 0.5),
            'hue': (0, 0.25)
        }
    },
    'gaussian_blur': {
        'class': GaussianBlurTransform,
        'param_ranges': {
            'radius': (0.1, 5.0)
        }
    },
    'perspective': {
        'class': PerspectiveTransform,
        'param_ranges': {
            'distortion_scale': (0.1, 0.5)
        }
    },
    'noise': {
        'class': NoisyTransform,
        'param_ranges': {
            'noise_level': (0.05, 0.5),
            'noise_type': ['gaussian', 'salt_and_pepper', 'poisson']
        }
    },
    'elastic': {
        'class': ElasticTransform,
        'param_ranges': {
            'alpha': (10, 100),
            'sigma': (1, 10)
        }
    }
}

def generate_random_transformation():
    """Generate a random transformation from the available ones."""
    # Select a random transformation type
    transform_type = random.choice(list(AVAILABLE_TRANSFORMATIONS.keys()))
    transform_info = AVAILABLE_TRANSFORMATIONS[transform_type]
    transform_class = transform_info['class']
    param_ranges = transform_info['param_ranges']
    
    # Generate random parameters based on the ranges
    params = {}
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple):
            # Numerical parameter
            params[param_name] = random.uniform(param_range[0], param_range[1])
        elif isinstance(param_range, list):
            # Categorical parameter
            params[param_name] = random.choice(param_range)
    
    # Create and return the transformation
    return transform_class(**params)

def generate_random_transformation_sequence(min_length=1, max_length=5):
    """Generate a random sequence of transformations."""
    seq_length = random.randint(min_length, max_length)
    return [generate_random_transformation() for _ in range(seq_length)]

def apply_transformation_sequence(image, transformations):
    """Apply a sequence of transformations to an image."""
    transformed_image = image
    for transform in transformations:
        transformed_image = transform(transformed_image)
    return transformed_image