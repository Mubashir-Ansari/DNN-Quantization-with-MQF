"""Torque-Based Structured Pruning implementation.

Based on: "Torque Based Structured Pruning for Deep Neural Network" 
by Arshita Gupta et al., WACV 2024.

Paper: https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Torque_Based_Structured_Pruning_for_Deep_Neural_Network_WACV_2024_paper.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from core.utils import calculate_sparsity


class TorqueBasedPruning:
    """
    Torque-Based Structured Pruning for Deep Neural Networks.
    
    This implementation follows the WACV 2024 paper methodology:
    1. Calculate torque values for each filter/neuron
    2. Rank filters based on torque importance
    3. Remove filters with lowest torque values
    4. No architecture changes required
    """
    
    def __init__(self, 
                 calibration_samples: int = 128,
                 torque_threshold: float = 0.1,
                 use_global_ranking: bool = True):
        """
        Initialize Torque-Based Pruning.
        
        Args:
            calibration_samples: Number of samples for torque calculation
            torque_threshold: Minimum torque value to consider
            use_global_ranking: Whether to use global or layer-wise ranking
        """
        self.calibration_samples = calibration_samples
        self.torque_threshold = torque_threshold
        self.use_global_ranking = use_global_ranking
        self.torque_values = {}
        
    def calculate_torque_values(self, model: nn.Module, dataloader: DataLoader, 
                               device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Calculate torque values for all prunable layers.
        
        The torque of a filter represents its rotational impact on the feature space.
        Higher torque indicates more important features that contribute significantly
        to the network's decision-making process.
        
        Args:
            model: Neural network model
            dataloader: Calibration dataloader
            device: Device to run calculations on
            
        Returns:
            Dictionary mapping layer names to torque values
        """
        model.eval()
        model.to(device)
        
        # Store activation statistics
        activation_stats = {}
        gradient_stats = {}
        
        # Register hooks to collect activations and gradients
        hooks = []
        
        def make_activation_hook(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = []
                
                # Store activations for torque calculation
                if isinstance(module, nn.Conv2d):
                    # For conv layers: [batch, channels, h, w]
                    # Calculate channel-wise statistics
                    activations = output.detach()
                    # Mean activation per channel across spatial dimensions
                    channel_means = torch.mean(activations, dim=(2, 3))  # [batch, channels]
                    activation_stats[name].append(channel_means)
                    
                elif isinstance(module, nn.Linear):
                    # For linear layers: [batch, features]
                    activations = output.detach()
                    activation_stats[name].append(activations)
                    
            return hook
        
        # Register hooks for prunable layers
        for name, module in model.named_modules():
            if self._is_prunable_module(module):
                hook = module.register_forward_hook(make_activation_hook(name))
                hooks.append(hook)
        
        # Collect activation statistics
        print(f"Calculating torque values using {self.calibration_samples} samples...")
        
        sample_count = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            if sample_count >= self.calibration_samples:
                break
                
            data, targets = data.to(device), targets.to(device)
            batch_size = data.size(0)
            
            # Forward pass to collect activations
            with torch.no_grad():
                outputs = model(data)
            
            sample_count += batch_size
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate torque values from collected statistics
        torque_values = {}
        
        for layer_name, activations_list in activation_stats.items():
            if activations_list:
                # Concatenate all activation batches
                all_activations = torch.cat(activations_list, dim=0)  # [total_samples, channels/features]
                
                # Calculate torque based on activation variance and correlation
                torque = self._calculate_layer_torque(all_activations, layer_name, model)
                torque_values[layer_name] = torque
        
        self.torque_values = torque_values
        print(f"✓ Calculated torque values for {len(torque_values)} layers")
        
        return torque_values
    
    def _calculate_layer_torque(self, activations: torch.Tensor, layer_name: str, 
                               model: nn.Module) -> torch.Tensor:
        """
        Calculate torque values for a specific layer.
        
        Torque represents the rotational moment of each filter's contribution
        to the feature space. It combines:
        1. Activation magnitude (force)
        2. Feature correlation (distance/lever arm)
        3. Weight significance (mass)
        
        Args:
            activations: Activation tensor [samples, channels/features]
            layer_name: Name of the layer
            model: Neural network model
            
        Returns:
            Torque values for each filter/neuron
        """
        # Get the corresponding module and weights
        module = None
        for name, mod in model.named_modules():
            if name == layer_name:
                module = mod
                break
        
        if module is None:
            # Fallback to uniform importance
            return torch.ones(activations.size(1), device=activations.device)
        
        # Calculate activation statistics
        activation_mean = torch.mean(activations, dim=0)  # [channels/features]
        activation_std = torch.std(activations, dim=0)    # [channels/features]
        
        # Calculate correlation matrix for feature interactions
        centered_activations = activations - activation_mean.unsqueeze(0)
        correlation_matrix = torch.mm(centered_activations.t(), centered_activations) / (activations.size(0) - 1)
        
        # Weight statistics
        weight_magnitude = torch.norm(module.weight.data.view(module.weight.size(0), -1), 
                                    dim=1, p=2)  # L2 norm per filter/neuron
        
        # Torque calculation combining multiple factors
        # Factor 1: Activation magnitude (force component)
        force_component = activation_std * torch.abs(activation_mean)
        
        # Factor 2: Feature correlation (lever arm component)
        # Use diagonal and off-diagonal correlations
        diagonal_corr = torch.diag(correlation_matrix)
        off_diagonal_corr = torch.sum(torch.abs(correlation_matrix), dim=1) - torch.abs(diagonal_corr)
        lever_arm_component = diagonal_corr + 0.1 * off_diagonal_corr  # Weight self vs cross correlations
        
        # Factor 3: Weight significance (mass component)
        mass_component = weight_magnitude / (torch.max(weight_magnitude) + 1e-8)
        
        # Combined torque: Torque = Force × Lever Arm × Mass
        torque = force_component * lever_arm_component * mass_component
        
        # Normalize to prevent numerical issues
        torque = torch.clamp(torque, min=self.torque_threshold)
        
        return torque
    
    def get_pruning_masks(self, model: nn.Module, target_sparsity: float) -> Dict[str, torch.Tensor]:
        """
        Generate structured pruning masks based on torque values.
        
        Args:
            model: Neural network model
            target_sparsity: Target sparsity percentage (0-100)
            
        Returns:
            Dictionary of structured pruning masks
        """
        print(f"DEBUG: TorqueBasedPruning get_pruning_masks() received target_sparsity={target_sparsity}")
        if not self.torque_values:
            raise ValueError("Torque values not calculated. Run calculate_torque_values first.")
        
        sparsity_ratio = target_sparsity / 100.0
        print(f"DEBUG: TorqueBasedPruning converted to sparsity_ratio={sparsity_ratio}")
        pruning_masks = {}
        
        if self.use_global_ranking:
            pruning_masks = self._global_torque_pruning(model, sparsity_ratio)
        else:
            pruning_masks = self._layer_wise_torque_pruning(model, sparsity_ratio)
        
        return pruning_masks
    
    def _global_torque_pruning(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Global structured pruning based on torque ranking."""
        pruning_masks = {}
        
        # Collect all torque values with layer information
        all_torques = []
        layer_info = []
        
        total_filters = 0
        for layer_name, torque_vals in self.torque_values.items():
            all_torques.append(torque_vals)
            
            layer_info.append({
                'name': layer_name,
                'num_filters': len(torque_vals),
                'start_idx': total_filters,
                'end_idx': total_filters + len(torque_vals)
            })
            total_filters += len(torque_vals)
        
        if not all_torques:
            return pruning_masks
        
        # Global ranking of all filters
        all_torques_cat = torch.cat(all_torques)
        num_filters_to_prune = int(total_filters * sparsity_ratio)
        print(f"DEBUG: TORQUE total_filters={total_filters}, to_prune={num_filters_to_prune}, sparsity_ratio={sparsity_ratio}")
        
        if num_filters_to_prune > 0:
            # Find global threshold (keep filters with highest torque)
            # We want to KEEP the top (1-sparsity_ratio) filters, so prune the bottom sparsity_ratio
            num_filters_to_keep = total_filters - num_filters_to_prune
            threshold = torch.kthvalue(all_torques_cat, num_filters_to_keep)[0]
            print(f"DEBUG: TORQUE threshold={threshold:.6f}, torques min={all_torques_cat.min():.6f}, max={all_torques_cat.max():.6f}")
            
            # Create structured masks
            for layer in layer_info:
                layer_torques = all_torques_cat[layer['start_idx']:layer['end_idx']]
                
                # Structured mask: 1 = keep filter, 0 = prune filter  
                # Keep filters with torques ABOVE threshold (higher torque = more important)
                filter_mask = (layer_torques > threshold).float()
                
                # Ensure at least one filter remains per layer
                if filter_mask.sum() == 0:
                    # Keep the filter with highest torque
                    max_idx = torch.argmax(layer_torques)
                    filter_mask[max_idx] = 1.0
                
                pruning_masks[layer['name']] = filter_mask
                
                # Log pruning statistics
                num_kept = int(filter_mask.sum().item())
                num_pruned = layer['num_filters'] - num_kept
                prune_percent = (num_pruned / layer['num_filters']) * 100
                
                print(f"  {layer['name']}: {num_pruned}/{layer['num_filters']} filters pruned ({prune_percent:.1f}%)")
        else:
            # No pruning
            for layer in layer_info:
                pruning_masks[layer['name']] = torch.ones(layer['num_filters'])
        
        return pruning_masks
    
    def _layer_wise_torque_pruning(self, model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Layer-wise structured pruning based on torque values."""
        pruning_masks = {}
        
        for layer_name, torque_vals in self.torque_values.items():
            num_filters = len(torque_vals)
            num_filters_to_prune = int(num_filters * sparsity_ratio)
            
            if num_filters_to_prune > 0 and num_filters_to_prune < num_filters:
                # Keep filters with highest torque values
                threshold = torch.kthvalue(torque_vals, num_filters_to_prune + 1)[0]
                filter_mask = (torque_vals >= threshold).float()
            else:
                # Keep all filters if pruning would remove everything
                filter_mask = torch.ones_like(torque_vals)
            
            pruning_masks[layer_name] = filter_mask
            
            # Log pruning statistics
            num_kept = int(filter_mask.sum().item())
            num_pruned = num_filters - num_kept
            prune_percent = (num_pruned / num_filters) * 100 if num_filters > 0 else 0
            
            print(f"  {layer_name}: {num_pruned}/{num_filters} filters pruned ({prune_percent:.1f}%)")
        
        return pruning_masks
    
    def apply_structured_pruning(self, model: nn.Module, 
                                pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Apply structured pruning masks to create a pruned model.
        
        Args:
            model: Original model
            pruning_masks: Structured pruning masks
            
        Returns:
            Structurally pruned model
        """
        # Note: True structured pruning requires architecture modification
        # For compatibility with existing evaluation framework, we implement
        # "pseudo-structured" pruning by zeroing out entire filters
        
        pruned_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for layer_name, mask in pruning_masks.items():
                # Find the corresponding module
                module = None
                for name, mod in pruned_model.named_modules():
                    if name == layer_name:
                        module = mod
                        break
                
                if module is None:
                    continue
                
                # Apply structured mask to filters
                if isinstance(module, nn.Conv2d):
                    # For conv layers: zero out entire filters
                    for i, keep_filter in enumerate(mask):
                        if keep_filter.item() == 0:
                            module.weight.data[i, :, :, :] = 0
                            if module.bias is not None:
                                module.bias.data[i] = 0
                                
                elif isinstance(module, nn.Linear):
                    # For linear layers: zero out entire neurons
                    for i, keep_neuron in enumerate(mask):
                        if keep_neuron.item() == 0:
                            module.weight.data[i, :] = 0
                            if module.bias is not None:
                                module.bias.data[i] = 0
        
        return pruned_model
    
    def _is_prunable_module(self, module: nn.Module) -> bool:
        """Check if module is prunable."""
        return isinstance(module, (nn.Conv2d, nn.Linear))
    
    def prune_model(self, model: nn.Module, target_sparsity: float,
                   calibration_dataloader: DataLoader, device: torch.device) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        print(f"DEBUG: TORQUE prune_model() received target_sparsity={target_sparsity}")
        """
        Complete Torque-Based pruning pipeline.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)
            calibration_dataloader: Data for torque calculation
            device: Device to run on
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        print(f"DEBUG: TorqueBasedPruning prune_model() received target_sparsity={target_sparsity}")
        print(f"Starting Torque-Based Structured Pruning (target: {target_sparsity:.1f}%)")
        
        # Step 1: Calculate torque values
        torque_values = self.calculate_torque_values(model, calibration_dataloader, device)
        
        # Step 2: Generate structured pruning masks
        pruning_masks = self.get_pruning_masks(model, target_sparsity)
        
        # Step 3: Apply structured pruning
        pruned_model = self.apply_structured_pruning(model, pruning_masks)
        
        # Step 4: Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        ranking_type = "global" if self.use_global_ranking else "layer-wise"
        
        print(f"Torque-Based pruning ({ranking_type}): Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


def create_torque_variants() -> Dict[str, TorqueBasedPruning]:
    """Create different variants of Torque-Based pruning."""
    return {
        'torque_global': TorqueBasedPruning(use_global_ranking=True, calibration_samples=128),
        'torque_layerwise': TorqueBasedPruning(use_global_ranking=False, calibration_samples=128),
        'torque_conservative': TorqueBasedPruning(use_global_ranking=True, torque_threshold=0.2),
        'torque_aggressive': TorqueBasedPruning(use_global_ranking=True, torque_threshold=0.05)
    }