"""Random pruning methods for baseline comparison."""

import torch
import torch.nn as nn
import copy
import random
from typing import Dict, List, Tuple, Optional
from core.utils import calculate_sparsity, set_random_seeds
import numpy as np

class RandomPruning:
    """Random pruning implementation for baseline comparisons."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize random pruning.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
    
    def get_random_pruning_mask(self, model: nn.Module, sparsity_ratio: float,
                               global_pruning: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate random pruning masks.
        
        Args:
            model: Model to generate masks for
            sparsity_ratio: Ratio of weights to prune (0-1)
            global_pruning: Whether to prune globally or layer-wise
            
        Returns:
            Dictionary of random pruning masks
        """
        set_random_seeds(self.seed)
        
        if global_pruning:
            return self._global_random_pruning(model, sparsity_ratio)
        else:
            return self._layer_wise_random_pruning(model, sparsity_ratio)
    
    def _global_random_pruning(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Global random pruning across all layers."""
        pruning_masks = {}
        
        # Collect all prunable parameters
        all_params = []
        layer_info = []
        
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name, param):
                param_flat = param.data.view(-1)
                all_params.append(param_flat)
                
                layer_info.append({
                    'name': name,
                    'shape': param.shape,
                    'size': param.numel(),
                    'start_idx': sum(info['size'] for info in layer_info),
                    'end_idx': sum(info['size'] for info in layer_info) + param.numel()
                })
        
        if not all_params:
            return pruning_masks
        
        # Create global mask
        total_params = sum(info['size'] for info in layer_info)
        num_to_prune = int(total_params * sparsity_ratio)
        
        # Generate random indices to prune
        all_indices = list(range(total_params))
        random.shuffle(all_indices)
        indices_to_prune = set(all_indices[:num_to_prune])
        
        # Create layer-specific masks
        for layer in layer_info:
            layer_mask = torch.ones(layer['size'])
            
            # Mark pruned weights as 0
            for i in range(layer['start_idx'], layer['end_idx']):
                if i in indices_to_prune:
                    layer_mask[i - layer['start_idx']] = 0.0
            
            # Reshape mask to original parameter shape
            layer_mask = layer_mask.view(layer['shape'])
            pruning_masks[layer['name']] = layer_mask
        
        return pruning_masks
    
    def _layer_wise_random_pruning(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Layer-wise random pruning."""
        pruning_masks = {}
        
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name, param):
                param_size = param.numel()
                num_to_prune = int(param_size * sparsity_ratio)
                
                # Create mask
                mask = torch.ones(param_size)
                
                if num_to_prune > 0:
                    # Randomly select indices to prune
                    indices = list(range(param_size))
                    random.shuffle(indices)
                    indices_to_prune = indices[:num_to_prune]
                    
                    for idx in indices_to_prune:
                        mask[idx] = 0.0
                
                # Reshape to parameter shape
                mask = mask.view(param.shape)
                pruning_masks[name] = mask
        
        return pruning_masks
    
    def _is_prunable_layer(self, name: str, param: torch.Tensor) -> bool:
        """Check if layer is prunable."""
        return (
            param.requires_grad and 
            len(param.shape) > 1 and  # Not bias terms
            'weight' in name and
            ('conv' in name.lower() or 'linear' in name.lower() or 
             'features' in name or 'classifier' in name)
        )
    
    def apply_pruning_masks(self, model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Apply pruning masks to model.
        
        Args:
            model: Model to prune
            pruning_masks: Pruning masks to apply
            
        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for name, mask in pruning_masks.items():
                if name in dict(pruned_model.named_parameters()):
                    param = dict(pruned_model.named_parameters())[name]
                    param.data.mul_(mask.to(param.device))
        
        return pruned_model
    
    def prune_model(self, model: nn.Module, target_sparsity: float,
                   global_pruning: bool = True) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Complete random pruning pipeline.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)
            global_pruning: Whether to use global pruning
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        print(f"DEBUG: random prune_model() received target_sparsity={target_sparsity}")
        sparsity_ratio = target_sparsity / 100.0
        print(f"DEBUG: converted to sparsity_ratio={sparsity_ratio}")
        
        # Generate random masks
        pruning_masks = self.get_random_pruning_mask(model, sparsity_ratio, global_pruning)
        
        # Apply masks
        pruned_model = self.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"Random pruning: Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


class StructuredRandomPruning:
    """Structured random pruning (removes entire channels/filters)."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize structured random pruning.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
    
    def get_structured_random_mask(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """
        Generate structured random pruning masks.
        
        Args:
            model: Model to generate masks for
            sparsity_ratio: Ratio of structures to prune
            
        Returns:
            Dictionary of structured pruning masks
        """
        set_random_seeds(self.seed)
        pruning_masks = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune entire output channels
                out_channels = module.out_channels
                num_to_prune = int(out_channels * sparsity_ratio)
                
                if num_to_prune > 0:
                    # Randomly select channels to prune
                    channels = list(range(out_channels))
                    random.shuffle(channels)
                    channels_to_prune = channels[:num_to_prune]
                    
                    # Create channel mask
                    mask = torch.ones(module.weight.shape)
                    for channel in channels_to_prune:
                        mask[channel, :, :, :] = 0.0
                    
                    pruning_masks[f"{name}.weight"] = mask
            
            elif isinstance(module, nn.Linear):
                # Prune entire output neurons
                out_features = module.out_features
                num_to_prune = int(out_features * sparsity_ratio)
                
                if num_to_prune > 0:
                    # Randomly select neurons to prune
                    neurons = list(range(out_features))
                    random.shuffle(neurons)
                    neurons_to_prune = neurons[:num_to_prune]
                    
                    # Create neuron mask
                    mask = torch.ones(module.weight.shape)
                    for neuron in neurons_to_prune:
                        mask[neuron, :] = 0.0
                    
                    pruning_masks[f"{name}.weight"] = mask
        
        return pruning_masks
    
    def prune_model_structured(self, model: nn.Module, target_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Perform structured random pruning.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        sparsity_ratio = target_sparsity / 100.0
        
        # Generate structured masks
        pruning_masks = self.get_structured_random_mask(model, sparsity_ratio)
        
        # Apply masks
        base_pruner = RandomPruning(self.seed)
        pruned_model = base_pruner.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"Structured random pruning: Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


class GradualRandomPruning:
    """Gradual random pruning over multiple iterations."""
    
    def __init__(self, num_iterations: int = 10, seed: int = 42):
        """
        Initialize gradual random pruning.
        
        Args:
            num_iterations: Number of pruning iterations
            seed: Random seed
        """
        self.num_iterations = num_iterations
        self.seed = seed
        self.random_pruner = RandomPruning(seed)
    
    def prune_gradually(self, model: nn.Module, target_sparsity: float,
                       fine_tune_func: Optional[callable] = None) -> nn.Module:
        """
        Perform gradual random pruning.
        
        Args:
            model: Model to prune
            target_sparsity: Final target sparsity
            fine_tune_func: Optional fine-tuning function
            
        Returns:
            Gradually pruned model
        """
        current_model = copy.deepcopy(model)
        target_ratio = target_sparsity / 100.0
        
        for iteration in range(self.num_iterations):
            # Calculate sparsity for this iteration
            progress = (iteration + 1) / self.num_iterations
            current_sparsity = target_ratio * progress
            
            print(f"Gradual random pruning iteration {iteration + 1}/{self.num_iterations}: "
                  f"{current_sparsity * 100:.1f}% sparsity")
            
            # Update seed for each iteration to get different random patterns
            self.random_pruner.seed = self.seed + iteration
            
            # Prune model
            current_model, masks = self.random_pruner.prune_model(
                current_model, current_sparsity * 100, global_pruning=True
            )
            
            # Fine-tune if function provided
            if fine_tune_func and iteration < self.num_iterations - 1:
                current_model = fine_tune_func(current_model, masks)
        
        return current_model


class BiasedRandomPruning:
    """Random pruning with bias towards larger/smaller weights."""
    
    def __init__(self, bias_factor: float = 0.1, favor_large: bool = False, seed: int = 42):
        """
        Initialize biased random pruning.
        
        Args:
            bias_factor: How much to bias selection (0 = pure random, 1 = magnitude-based)
            favor_large: Whether to bias towards large weights (True) or small weights (False)
            seed: Random seed
        """
        self.bias_factor = bias_factor
        self.favor_large = favor_large
        self.seed = seed
    
    def get_biased_random_mask(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """
        Generate biased random pruning masks.
        
        Args:
            model: Model to generate masks for
            sparsity_ratio: Ratio of weights to prune
            
        Returns:
            Dictionary of biased random pruning masks
        """
        set_random_seeds(self.seed)
        pruning_masks = {}
        
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name, param):
                weights = param.data.view(-1)
                magnitudes = torch.abs(weights)
                
                # Calculate probabilities based on magnitude bias
                if self.bias_factor > 0:
                    if self.favor_large:
                        # Higher probability for larger weights
                        probs = magnitudes / (magnitudes.sum() + 1e-8)
                    else:
                        # Higher probability for smaller weights
                        probs = (1.0 / (magnitudes + 1e-8))
                        probs = probs / probs.sum()
                    
                    # Blend with uniform probability
                    uniform_prob = 1.0 / len(weights)
                    final_probs = (1 - self.bias_factor) * uniform_prob + self.bias_factor * probs
                else:
                    # Pure random
                    final_probs = torch.ones(len(weights)) / len(weights)
                
                # Sample indices to prune based on probabilities
                num_to_prune = int(len(weights) * sparsity_ratio)
                
                if num_to_prune > 0:
                    # Convert to numpy for sampling
                    probs_np = final_probs.cpu().numpy()
                    indices_to_prune = torch.from_numpy(
                        np.random.choice(len(weights), size=num_to_prune, 
                                       replace=False, p=probs_np)
                    )
                    
                    # Create mask
                    mask = torch.ones(len(weights))
                    mask[indices_to_prune] = 0.0
                    mask = mask.view(param.shape)
                else:
                    mask = torch.ones_like(param)
                
                pruning_masks[name] = mask
        
        return pruning_masks
    
    def _is_prunable_layer(self, name: str, param: torch.Tensor) -> bool:
        """Check if layer is prunable."""
        return (
            param.requires_grad and 
            len(param.shape) > 1 and  # Not bias terms
            'weight' in name and
            ('conv' in name.lower() or 'linear' in name.lower() or 
             'features' in name or 'classifier' in name)
        )


def create_random_pruning_variants() -> Dict[str, RandomPruning]:
    """Create different variants of random pruning."""
    return {
        'random_global': RandomPruning(seed=42),
        'random_layerwise': RandomPruning(seed=43),
        'random_structured': StructuredRandomPruning(seed=44),
        'random_gradual': GradualRandomPruning(seed=45),
        'random_biased_small': BiasedRandomPruning(bias_factor=0.3, favor_large=False, seed=46),
        'random_biased_large': BiasedRandomPruning(bias_factor=0.3, favor_large=True, seed=47)
    }