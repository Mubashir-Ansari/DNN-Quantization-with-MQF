"""EVOP: Evolutionary Pruning implementation."""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import DataLoader, Subset
import copy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class EVOPPruning:
    """EVOP (Evolutionary Pruning) implementation."""
    
    def __init__(self, 
                 num_generations: int = 20,
                 population_size: int = 30,
                 mutation_rate: float = 0.1,
                 num_clusters: int = 10,
                 samples_per_cluster: int = 50,
                 calibration_samples: int = 500):
        """
        Initialize EVOP pruning.
        
        Args:
            num_generations: Number of evolutionary generations
            population_size: Size of population of pruning patterns
            mutation_rate: Probability of mutation for each gene
            num_clusters: Number of clusters for calibration dataset
            samples_per_cluster: Samples to select from each cluster
            calibration_samples: Total calibration samples
        """
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_clusters = num_clusters
        self.samples_per_cluster = samples_per_cluster
        self.calibration_samples = calibration_samples
        self.prunable_layers = []
        
    def _create_calibration_dataset(self, dataloader: DataLoader) -> DataLoader:
        """
        Create diverse calibration dataset using clustering.
        
        Args:
            dataloader: Original training dataloader
            
        Returns:
            Calibration dataloader with diverse samples
        """
        print("Creating calibration dataset using clustering...")
        
        # Extract features from a subset of data for clustering
        features = []
        indices = []
        
        for i, (data, target) in enumerate(dataloader):
            if len(features) >= self.calibration_samples:
                break
            
            # Flatten data for feature extraction
            batch_features = data.view(data.size(0), -1).numpy()
            features.extend(batch_features)
            indices.extend([i * dataloader.batch_size + j for j in range(data.size(0))])
        
        features = np.array(features[:self.calibration_samples])
        indices = indices[:self.calibration_samples]
        
        # Perform clustering
        n_clusters = min(self.num_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select representative samples from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = [idx for idx, label in zip(indices, cluster_labels) if label == cluster_id]
            
            # Select samples_per_cluster from this cluster
            samples_to_select = min(self.samples_per_cluster, len(cluster_indices))
            selected_cluster_indices = random.sample(cluster_indices, samples_to_select)
            selected_indices.extend(selected_cluster_indices)
        
        print(f"Selected {len(selected_indices)} calibration samples from {n_clusters} clusters")
        
        # Create subset dataset
        subset = Subset(dataloader.dataset, selected_indices)
        return DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
    
    def _get_prunable_layers(self, model: nn.Module) -> List[str]:
        """Get list of prunable layer names."""
        prunable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable_layers.append(name)
        return prunable_layers
    
    def _create_random_pattern(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """
        Create a random pruning pattern.
        
        Args:
            model: Neural network model
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Dictionary of pruning masks for each layer
        """
        patterns = {}
        device = next(model.parameters()).device
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_shape = module.weight.shape
                total_params = module.weight.numel()
                
                # Create random binary mask on the same device
                num_to_prune = int(total_params * sparsity_ratio)
                mask = torch.ones(total_params, dtype=torch.bool, device=device)
                if num_to_prune > 0:
                    prune_indices = torch.randperm(total_params, device=device)[:num_to_prune]
                    mask[prune_indices] = False
                
                patterns[name] = mask.reshape(weight_shape)
        
        return patterns
    
    def _mutate_pattern(self, pattern: Dict[str, torch.Tensor], mutation_rate: float) -> Dict[str, torch.Tensor]:
        """
        Mutate a pruning pattern.
        
        Args:
            pattern: Original pruning pattern
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated pruning pattern
        """
        mutated_pattern = {}
        
        for layer_name, mask in pattern.items():
            new_mask = mask.clone()
            
            # Apply mutations
            mutation_prob = torch.rand_like(mask.float())
            mutation_locations = mutation_prob < mutation_rate
            
            # Flip bits at mutation locations
            new_mask[mutation_locations] = ~new_mask[mutation_locations]
            
            mutated_pattern[layer_name] = new_mask
        
        return mutated_pattern
    
    def _crossover_patterns(self, parent1: Dict[str, torch.Tensor], 
                          parent2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform crossover between two pruning patterns.
        
        Args:
            parent1: First parent pattern
            parent2: Second parent pattern
            
        Returns:
            Offspring pattern
        """
        offspring = {}
        
        for layer_name in parent1.keys():
            mask1 = parent1[layer_name]
            mask2 = parent2[layer_name]
            
            # Random binary crossover mask
            crossover_mask = torch.rand_like(mask1.float()) < 0.5
            
            # Create offspring by combining parents
            offspring_mask = torch.where(crossover_mask, mask1, mask2)
            offspring[layer_name] = offspring_mask
        
        return offspring
    
    def _evaluate_pattern(self, model: nn.Module, pattern: Dict[str, torch.Tensor], 
                         calibration_loader: DataLoader) -> float:
        """
        Evaluate a pruning pattern using calibration data.
        
        Args:
            model: Neural network model
            pattern: Pruning pattern to evaluate
            calibration_loader: Calibration dataloader
            
        Returns:
            Fitness score (higher is better)
        """
        # Create pruned model
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning pattern
        with torch.no_grad():
            for name, module in pruned_model.named_modules():
                if name in pattern:
                    mask = pattern[name]
                    module.weight.data *= mask.float()
        
        # Evaluate on calibration data
        pruned_model.eval()
        total_correct = 0
        total_samples = 0
        
        device = next(pruned_model.parameters()).device
        
        with torch.no_grad():
            for data, target in calibration_loader:
                data, target = data.to(device), target.to(device)
                outputs = pruned_model(data)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return accuracy
    
    def _evolutionary_search(self, model: nn.Module, sparsity_ratio: float, 
                           calibration_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Perform evolutionary search for optimal pruning pattern.
        
        Args:
            model: Neural network model
            sparsity_ratio: Target sparsity ratio
            calibration_loader: Calibration dataloader
            
        Returns:
            Best pruning pattern found
        """
        print(f"Starting evolutionary search...")
        print(f"Generations: {self.num_generations}, Population: {self.population_size}")
        
        # Initialize population with random patterns
        population = []
        fitness_scores = []
        
        print("Initializing population...")
        for i in range(self.population_size):
            pattern = self._create_random_pattern(model, sparsity_ratio)
            fitness = self._evaluate_pattern(model, pattern, calibration_loader)
            population.append(pattern)
            fitness_scores.append(fitness)
            
            if (i + 1) % 10 == 0:
                print(f"  Initialized {i + 1}/{self.population_size} individuals")
        
        best_fitness = max(fitness_scores)
        best_pattern = population[fitness_scores.index(best_fitness)]
        
        print(f"Initial best fitness: {best_fitness:.4f}")
        
        # Evolution loop
        for generation in range(self.num_generations):
            new_population = []
            new_fitness_scores = []
            
            # Keep best individual (elitism)
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(copy.deepcopy(population[best_idx]))
            new_fitness_scores.append(fitness_scores[best_idx])
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1_idx = max(random.sample(range(len(population)), 3), key=lambda x: fitness_scores[x])
                parent2_idx = max(random.sample(range(len(population)), 3), key=lambda x: fitness_scores[x])
                
                # Crossover
                offspring = self._crossover_patterns(population[parent1_idx], population[parent2_idx])
                
                # Mutation
                offspring = self._mutate_pattern(offspring, self.mutation_rate)
                
                # Evaluate offspring
                offspring_fitness = self._evaluate_pattern(model, offspring, calibration_loader)
                
                new_population.append(offspring)
                new_fitness_scores.append(offspring_fitness)
            
            population = new_population
            fitness_scores = new_fitness_scores
            
            # Update best
            current_best_fitness = max(fitness_scores)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_pattern = population[fitness_scores.index(current_best_fitness)]
            
            print(f"Generation {generation + 1}/{self.num_generations}: "
                  f"Best fitness = {best_fitness:.4f}, Avg fitness = {np.mean(fitness_scores):.4f}")
        
        print(f"Evolution completed. Best fitness: {best_fitness:.4f}")
        return best_pattern
    
    def prune_model(self, model: nn.Module, train_loader: DataLoader, 
                   sparsity_ratio: float = 0.5) -> nn.Module:
        """
        Prune model using EVOP method.
        
        Args:
            model: Model to prune
            train_loader: Training data loader
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Pruned model
        """
        print("Starting EVOP (Evolutionary Pruning)...")
        
        # Create calibration dataset
        calibration_loader = self._create_calibration_dataset(train_loader)
        
        # Perform evolutionary search
        best_pattern = self._evolutionary_search(model, sparsity_ratio, calibration_loader)
        
        # Apply best pattern to model
        pruned_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, module in pruned_model.named_modules():
                if name in best_pattern:
                    mask = best_pattern[name]
                    module.weight.data *= mask.float()
        
        print("EVOP pruning completed")
        return pruned_model
    
    def get_pruning_masks(self, model: nn.Module, train_loader: DataLoader, 
                         sparsity_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Get pruning masks using EVOP method.
        
        Args:
            model: Model to analyze
            train_loader: Training data loader
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Dictionary of pruning masks
        """
        # Create calibration dataset
        calibration_loader = self._create_calibration_dataset(train_loader)
        
        # Perform evolutionary search
        best_pattern = self._evolutionary_search(model, sparsity_ratio, calibration_loader)
        
        return best_pattern


def create_evop_dataloader(train_loader: DataLoader, num_samples: int) -> DataLoader:
    """Create a calibration dataloader for EVOP."""
    from torch.utils.data import Subset
    
    # Create subset indices
    total_samples = len(train_loader.dataset)
    indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    subset = Subset(train_loader.dataset, indices)
    
    # Create new dataloader
    return DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )