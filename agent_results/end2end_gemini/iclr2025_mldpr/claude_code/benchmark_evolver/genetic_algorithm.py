"""
Genetic algorithm implementation for the Benchmark Evolver.
"""

import random
import numpy as np
import os
import json
import pickle
from collections import defaultdict
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_evolver.transformations import (
    generate_random_transformation_sequence,
    apply_transformation_sequence
)

logger = logging.getLogger(__name__)

class Individual:
    """Individual in the genetic algorithm representing a transformation sequence."""
    
    def __init__(self, transformations=None, fitness=None):
        """
        Initialize an individual.
        
        Args:
            transformations: List of transformation objects
            fitness: Fitness value
        """
        self.transformations = transformations or []
        self.fitness = fitness
        self.fitness_components = {}  # Detailed fitness components
        self.id = random.getrandbits(64)  # Unique ID for tracking
    
    def __repr__(self):
        """String representation of the individual."""
        return f"Individual(fitness={self.fitness}, transforms={len(self.transformations)})"
    
    def to_dict(self):
        """Convert individual to a dictionary for saving."""
        return {
            'id': self.id,
            'fitness': self.fitness,
            'fitness_components': self.fitness_components,
            'transformations': self.transformations
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create an individual from a dictionary."""
        ind = cls(transformations=data['transformations'], fitness=data['fitness'])
        ind.id = data['id']
        ind.fitness_components = data.get('fitness_components', {})
        return ind


class BenchmarkEvolver:
    """Genetic algorithm implementation for evolving benchmark transformations."""
    
    def __init__(self, 
                 pop_size=50, 
                 max_generations=100,
                 tournament_size=3,
                 crossover_prob=0.7,
                 mutation_prob=0.3,
                 elitism_count=2,
                 min_transformations=1,
                 max_transformations=5,
                 seed=None,
                 save_dir='./results/evolutionary_runs'):
        """
        Initialize the Benchmark Evolver.
        
        Args:
            pop_size: Population size
            max_generations: Maximum number of generations
            tournament_size: Tournament selection size
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elitism_count: Number of best individuals to preserve unchanged
            min_transformations: Minimum number of transformations in a sequence
            max_transformations: Maximum number of transformations in a sequence
            seed: Random seed for reproducibility
            save_dir: Directory to save evolutionary runs
        """
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_count = elitism_count
        self.min_transformations = min_transformations
        self.max_transformations = max_transformations
        self.save_dir = save_dir
        
        # Initialize random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create directory for saving
        os.makedirs(save_dir, exist_ok=True)
        
        # Population
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }
    
    def initialize_population(self):
        """Initialize the population with random individuals."""
        logger.info(f"Initializing population with {self.pop_size} individuals")
        self.population = []
        
        for _ in range(self.pop_size):
            transformations = generate_random_transformation_sequence(
                min_length=self.min_transformations,
                max_length=self.max_transformations
            )
            ind = Individual(transformations=transformations)
            self.population.append(ind)
        
        self.generation = 0
        logger.info(f"Population initialized with {len(self.population)} individuals")
    
    def evaluate_population(self, evaluation_function):
        """
        Evaluate all individuals in the population.
        
        Args:
            evaluation_function: Function that takes an individual and returns its fitness
        """
        logger.info(f"Evaluating population of {len(self.population)} individuals")
        
        for i, ind in enumerate(self.population):
            if ind.fitness is None:  # Only evaluate if not already evaluated
                fitness_result = evaluation_function(ind)
                
                # Handle different return types from evaluation function
                if isinstance(fitness_result, (int, float)):
                    ind.fitness = fitness_result
                elif isinstance(fitness_result, dict):
                    # If the evaluation function returns detailed components
                    ind.fitness = fitness_result.get('overall', 0)
                    ind.fitness_components = fitness_result
                
                logger.debug(f"Individual {i} evaluated: fitness={ind.fitness}")
        
        # Update best individual
        self.update_best_individual()
        
        # Update history
        self.update_history()
        
        logger.info("Population evaluation completed")
    
    def update_best_individual(self):
        """Update the best individual found so far."""
        current_best = max(self.population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
        
        if self.best_individual is None or (current_best.fitness > self.best_individual.fitness):
            self.best_individual = Individual(
                transformations=current_best.transformations.copy(),
                fitness=current_best.fitness
            )
            self.best_individual.fitness_components = current_best.fitness_components.copy()
            self.best_individual.id = current_best.id
            
            logger.info(f"New best individual found: fitness={self.best_individual.fitness}")
    
    def update_history(self):
        """Update history with current population statistics."""
        # Filter out individuals with None fitness
        valid_individuals = [ind for ind in self.population if ind.fitness is not None]
        
        if valid_individuals:
            fitnesses = [ind.fitness for ind in valid_individuals]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = max(fitnesses)
            
            # Calculate population diversity (average pairwise distance)
            diversity = self.calculate_population_diversity()
            
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['diversity'].append(diversity)
            
            logger.info(f"Generation {self.generation}: best_fitness={best_fitness:.4f}, "
                       f"avg_fitness={avg_fitness:.4f}, diversity={diversity:.4f}")
    
    def calculate_population_diversity(self):
        """Calculate diversity in the population based on transformation types."""
        if not self.population:
            return 0
        
        # Count unique transformation types in each individual
        transformation_types = []
        for ind in self.population:
            types = {type(t).__name__ for t in ind.transformations}
            transformation_types.append(types)
        
        # Calculate Jaccard distance between pairs
        n = len(transformation_types)
        total_distance = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                set_i = transformation_types[i]
                set_j = transformation_types[j]
                
                # Jaccard distance
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                if union > 0:
                    distance = 1 - (intersection / union)
                    total_distance += distance
                    count += 1
        
        # Average distance
        avg_distance = total_distance / count if count > 0 else 0
        return avg_distance
    
    def select_parent(self):
        """Select a parent using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        valid_tournament = [ind for ind in tournament if ind.fitness is not None]
        
        if not valid_tournament:
            # If no valid individuals in tournament, pick a random one
            return random.choice(self.population)
        
        return max(valid_tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1, parent2: Parent individuals
        
        Returns:
            Two child individuals
        """
        if random.random() > self.crossover_prob:
            # No crossover, return copies of parents
            child1 = Individual(transformations=parent1.transformations.copy())
            child2 = Individual(transformations=parent2.transformations.copy())
            return child1, child2
        
        # One-point crossover
        min_len = min(len(parent1.transformations), len(parent2.transformations))
        
        if min_len <= 1:
            # Not enough transformations for crossover
            child1 = Individual(transformations=parent1.transformations.copy())
            child2 = Individual(transformations=parent2.transformations.copy())
            return child1, child2
        
        crossover_point = random.randint(1, min_len - 1)
        
        child1_transforms = parent1.transformations[:crossover_point] + parent2.transformations[crossover_point:]
        child2_transforms = parent2.transformations[:crossover_point] + parent1.transformations[crossover_point:]
        
        # Ensure length constraints
        if len(child1_transforms) > self.max_transformations:
            child1_transforms = child1_transforms[:self.max_transformations]
        if len(child2_transforms) > self.max_transformations:
            child2_transforms = child2_transforms[:self.max_transformations]
        
        child1 = Individual(transformations=child1_transforms)
        child2 = Individual(transformations=child2_transforms)
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
        
        Returns:
            Mutated individual
        """
        if random.random() > self.mutation_prob:
            return individual
        
        # Clone transformations to avoid modifying original
        mutated_transforms = individual.transformations.copy()
        
        # Possible mutation operations
        mutation_ops = ['modify', 'add', 'remove', 'swap']
        
        # Choose one mutation operation
        op = random.choice(mutation_ops)
        
        if op == 'modify' and mutated_transforms:
            # Modify parameters of a random transformation
            idx = random.randrange(len(mutated_transforms))
            mutated_transforms[idx] = mutated_transforms[idx].mutate()
        
        elif op == 'add' and len(mutated_transforms) < self.max_transformations:
            # Add a random transformation
            new_transform = generate_random_transformation_sequence(min_length=1, max_length=1)[0]
            insert_pos = random.randint(0, len(mutated_transforms))
            mutated_transforms.insert(insert_pos, new_transform)
        
        elif op == 'remove' and len(mutated_transforms) > self.min_transformations:
            # Remove a random transformation
            idx = random.randrange(len(mutated_transforms))
            mutated_transforms.pop(idx)
        
        elif op == 'swap' and len(mutated_transforms) >= 2:
            # Swap two transformations
            idx1, idx2 = random.sample(range(len(mutated_transforms)), 2)
            mutated_transforms[idx1], mutated_transforms[idx2] = mutated_transforms[idx2], mutated_transforms[idx1]
        
        return Individual(transformations=mutated_transforms)
    
    def evolve(self, evaluation_function, progress_callback=None):
        """
        Run the evolutionary process.
        
        Args:
            evaluation_function: Function to evaluate individual fitness
            progress_callback: Optional callback function for progress updates
        
        Returns:
            The best individual found
        """
        logger.info(f"Starting evolutionary process: {self.max_generations} generations, "
                   f"population size {self.pop_size}")
        
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population(evaluation_function)
        
        # Evolution loop
        for generation in range(1, self.max_generations + 1):
            self.generation = generation
            logger.info(f"Generation {generation}/{self.max_generations}")
            
            # Create new population
            new_population = []
            
            # Elitism - preserve best individuals
            sorted_pop = sorted(self.population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'), reverse=True)
            elites = sorted_pop[:self.elitism_count]
            new_population.extend([Individual(transformations=elite.transformations.copy(), fitness=elite.fitness) for elite in elites])
            
            # Generate new individuals
            while len(new_population) < self.pop_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Replace population
            self.population = new_population
            
            # Evaluate new population
            self.evaluate_population(evaluation_function)
            
            # Save checkpoint
            if generation % 10 == 0 or generation == self.max_generations:
                self.save_checkpoint(f"checkpoint_gen_{generation}.pkl")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(generation, self.best_individual, self.history)
        
        logger.info(f"Evolution completed. Best fitness: {self.best_individual.fitness}")
        return self.best_individual
    
    def save_checkpoint(self, filename):
        """Save the current state of the evolutionary process."""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'generation': self.generation,
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'population': [ind.to_dict() for ind in self.population],
            'history': self.history,
            'config': {
                'pop_size': self.pop_size,
                'max_generations': self.max_generations,
                'tournament_size': self.tournament_size,
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob,
                'elitism_count': self.elitism_count,
                'min_transformations': self.min_transformations,
                'max_transformations': self.max_transformations
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved to {filepath}")
        
        # Also save a JSON version of history
        history_filepath = os.path.join(self.save_dir, 'evolution_history.json')
        with open(history_filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def load_checkpoint(self, filepath):
        """Load a saved checkpoint."""
        logger.info(f"Loading checkpoint from {filepath}")
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.generation = checkpoint['generation']
        self.history = checkpoint['history']
        
        # Restore config
        config = checkpoint['config']
        self.pop_size = config['pop_size']
        self.max_generations = config['max_generations']
        self.tournament_size = config['tournament_size']
        self.crossover_prob = config['crossover_prob']
        self.mutation_prob = config['mutation_prob']
        self.elitism_count = config['elitism_count']
        self.min_transformations = config['min_transformations']
        self.max_transformations = config['max_transformations']
        
        # Restore best individual
        if checkpoint['best_individual']:
            self.best_individual = Individual.from_dict(checkpoint['best_individual'])
        
        # Restore population
        self.population = [Individual.from_dict(ind_dict) for ind_dict in checkpoint['population']]
        
        logger.info(f"Checkpoint loaded: generation {self.generation}, population size {len(self.population)}")
    
    def get_best_individuals(self, n=10):
        """Get the n best individuals from the current population."""
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'), reverse=True)
        return sorted_pop[:n]
    
    def generate_adversarial_batch(self, dataset, indices=None, batch_size=32):
        """
        Generate a batch of adversarially transformed images.
        
        Args:
            dataset: The original dataset
            indices: Optional list of indices to use from dataset
            batch_size: Number of images to generate
        
        Returns:
            Tuple of (images, transformed_images, labels)
        """
        if self.best_individual is None:
            logger.warning("No best individual found. Using a random one from population.")
            if not self.population:
                raise ValueError("Population is empty. Cannot generate adversarial batch.")
            
            # Use the best available individual
            individual = max(self.population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
        else:
            individual = self.best_individual
        
        # If no indices are provided, select random ones
        if indices is None:
            indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
        
        # Get original images and labels
        original_images = []
        transformed_images = []
        labels = []
        
        for idx in indices:
            image, label = dataset[idx]
            
            # Check if the image is a tensor, if not convert it
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)
            
            # Apply transformations
            transformed_image = apply_transformation_sequence(image, individual.transformations)
            
            original_images.append(image)
            transformed_images.append(transformed_image)
            labels.append(label)
        
        # Stack tensors
        original_images = torch.stack(original_images)
        transformed_images = torch.stack(transformed_images)
        labels = torch.tensor(labels)
        
        return original_images, transformed_images, labels
    
    def create_adversarial_dataloader(self, dataset, indices=None, batch_size=32):
        """
        Create a DataLoader with adversarially transformed images.
        
        Args:
            dataset: The original dataset
            indices: Optional list of indices to use from dataset
            batch_size: Batch size
        
        Returns:
            DataLoader with transformed data
        """
        # Generate the transformed data
        _, transformed_images, labels = self.generate_adversarial_batch(
            dataset, indices, batch_size=len(dataset) if indices is None else len(indices)
        )
        
        # Create a TensorDataset
        transformed_dataset = TensorDataset(transformed_images, labels)
        
        # Create a DataLoader
        return DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=True
        )