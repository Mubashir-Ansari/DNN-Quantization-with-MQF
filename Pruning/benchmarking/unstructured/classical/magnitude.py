"""Magnitude-based pruning methods."""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
from core.utils import count_nonzero_parameters, calculate_sparsity


class MagnitudePruning:
    """Implementation of magnitude-based pruning methods."""
    
    def __init__(self, structured: bool = False):
        """
        Initialize magnitude pruning.
        
        Args:
            structured: Whether to perform structured pruning
        """
        self.structured = structured
    
    def get_pruning_mask(self, model: nn.Module, sparsity_ratio: float, 
                        global_pruning: bool = True, 
                        layer_wise_ratios: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Generate pruning masks based on weight magnitudes.
        
        Args:
            model: Model to generate masks for
            sparsity_ratio: Overall sparsity ratio (0-1)
            global_pruning: Whether to prune globally or layer-wise
            layer_wise_ratios: Specific ratios for each layer
            
        Returns:
            Dictionary of pruning masks
        """
        if global_pruning:
            return self._global_magnitude_pruning(model, sparsity_ratio)
        else:
            return self._layer_wise_magnitude_pruning(model, sparsity_ratio, layer_wise_ratios)
    
    def _global_magnitude_pruning(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Global magnitude-based pruning."""
        print(f"DEBUG: _global_magnitude_pruning received sparsity_ratio={sparsity_ratio}")
        pruning_masks = {}
        
        # Collect all weights and their absolute values
        all_weights = []
        layer_info = []
        
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name, param):
                weights_flat = param.data.view(-1)
                magnitudes = torch.abs(weights_flat)
                
                all_weights.append(magnitudes)
                layer_info.append({
                    'name': name,
                    'shape': param.shape,
                    'start_idx': len(torch.cat(all_weights[:-1])) if len(all_weights) > 1 else 0,
                    'end_idx': len(torch.cat(all_weights))
                })
                print(f"DEBUG: Added layer {name} with {len(weights_flat)} parameters")
        
        if not all_weights:
            print("DEBUG: No prunable layers found!")
            return pruning_masks
        
        # Concatenate all weights
        all_weights_cat = torch.cat(all_weights)
        print(f"DEBUG: Total parameters: {len(all_weights_cat)}")
        
        # Find global threshold
        num_params_to_prune = int(len(all_weights_cat) * sparsity_ratio)
        print(f"DEBUG: Parameters to prune: {num_params_to_prune} (ratio: {sparsity_ratio})")
        
        if num_params_to_prune > 0:
            threshold = torch.kthvalue(all_weights_cat, num_params_to_prune)[0]
            print(f"DEBUG: Pruning threshold: {threshold}")
            
            # Create masks for each layer
            for layer in layer_info:
                layer_weights = all_weights_cat[layer['start_idx']:layer['end_idx']]
                mask = (layer_weights > threshold).float()
                mask = mask.view(layer['shape'])
                pruning_masks[layer['name']] = mask
                pruned_count = (mask == 0).sum().item()
                total_count = mask.numel()
                layer_sparsity = (pruned_count / total_count) * 100
                print(f"DEBUG: Layer {layer['name']}: {pruned_count}/{total_count} pruned ({layer_sparsity:.2f}%)")
        else:
            # No pruning
            print("DEBUG: No parameters to prune (num_params_to_prune = 0)")
            for layer in layer_info:
                pruning_masks[layer['name']] = torch.ones(layer['shape'])
        
        return pruning_masks
    
    def _layer_wise_magnitude_pruning(self, model: nn.Module, sparsity_ratio: float,
                                     layer_wise_ratios: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Layer-wise magnitude-based pruning."""
        pruning_masks = {}
        
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name, param):
                # Use layer-specific ratio if provided
                if layer_wise_ratios and name in layer_wise_ratios:
                    layer_sparsity = layer_wise_ratios[name]
                else:
                    layer_sparsity = sparsity_ratio
                
                # Get layer weights and magnitudes
                weights = param.data
                magnitudes = torch.abs(weights.view(-1))
                
                # Calculate threshold for this layer
                num_params_to_prune = int(len(magnitudes) * layer_sparsity)
                
                if num_params_to_prune > 0:
                    threshold = torch.kthvalue(magnitudes, num_params_to_prune)[0]
                    mask = (magnitudes > threshold).float()
                    mask = mask.view(weights.shape)
                else:
                    mask = torch.ones_like(weights)
                
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
        Complete pruning pipeline.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)
            global_pruning: Whether to use global pruning
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        print(f"DEBUG: magnitude prune_model() received target_sparsity={target_sparsity}")
        sparsity_ratio = target_sparsity / 100.0
        print(f"DEBUG: converted to sparsity_ratio={sparsity_ratio}")
        
        # Generate masks
        pruning_masks = self.get_pruning_mask(model, sparsity_ratio, global_pruning)
        
        # Apply masks
        pruned_model = self.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        
        # Debug parameter counts
        from core.utils import count_total_parameters, count_nonzero_parameters
        total_all = count_total_parameters(pruned_model)
        nonzero_all = count_nonzero_parameters(pruned_model)
        print(f"DEBUG: Total ALL parameters: {total_all}, Non-zero ALL: {nonzero_all}")
        print(f"DEBUG: Pruned ALL parameters: {total_all - nonzero_all} ({(total_all-nonzero_all)/total_all*100:.2f}%)")
        
        print(f"Magnitude pruning: Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


class LayerWiseMagnitudePruning(MagnitudePruning):
    """Layer-wise magnitude pruning with custom ratios per layer type."""
    
    def __init__(self):
        super().__init__(structured=False)
    
    def get_adaptive_layer_ratios(self, model: nn.Module, target_sparsity: float) -> Dict[str, float]:
        """
        Get adaptive pruning ratios for different layer types.
        
        Args:
            model: Model to analyze
            target_sparsity: Overall target sparsity
            
        Returns:
            Dictionary of layer-specific sparsity ratios
        """
        layer_ratios = {}
        base_ratio = target_sparsity / 100.0
        
        for name, param in model.named_parameters():
            if self._is_prunable_layer(name, param):
                if 'features' in name:  # Convolutional layers
                    # More conservative pruning for conv layers
                    layer_ratios[name] = base_ratio * 0.7
                elif 'classifier' in name:  # Fully connected layers  
                    # More aggressive pruning for FC layers
                    layer_ratios[name] = base_ratio * 1.3
                else:
                    layer_ratios[name] = base_ratio
        
        return layer_ratios
    
    def prune_model_adaptive(self, model: nn.Module, target_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune model with adaptive layer-wise ratios.
        
        Args:
            model: Model to prune
            target_sparsity: Target overall sparsity
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        layer_ratios = self.get_adaptive_layer_ratios(model, target_sparsity)
        
        # Generate masks with adaptive ratios
        pruning_masks = self._layer_wise_magnitude_pruning(
            model, target_sparsity / 100.0, layer_ratios
        )
        
        # Apply masks
        pruned_model = self.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"Adaptive magnitude pruning: Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


class GradualMagnitudePruning:
    """Gradual magnitude pruning over multiple iterations."""
    
    def __init__(self, num_iterations: int = 10):
        """
        Initialize gradual pruning.
        
        Args:
            num_iterations: Number of pruning iterations
        """
        self.num_iterations = num_iterations
        self.magnitude_pruner = MagnitudePruning()
    
    def prune_gradually(self, model: nn.Module, target_sparsity: float,
                       fine_tune_func: Optional[callable] = None) -> nn.Module:
        """
        Perform gradual magnitude pruning.
        
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
            
            print(f"Gradual pruning iteration {iteration + 1}/{self.num_iterations}: "
                  f"{current_sparsity * 100:.1f}% sparsity")
            
            # Prune model
            current_model, masks = self.magnitude_pruner.prune_model(
                current_model, current_sparsity * 100, global_pruning=True
            )
            
            # Fine-tune if function provided
            if fine_tune_func and iteration < self.num_iterations - 1:
                current_model = fine_tune_func(current_model, masks)
        
        return current_model


def create_magnitude_pruning_variants() -> Dict[str, MagnitudePruning]:
    """Create different variants of magnitude pruning."""
    return {
        'magnitude_global': MagnitudePruning(),
        'magnitude_layerwise': LayerWiseMagnitudePruning(),
        'magnitude_gradual': GradualMagnitudePruning()
    }