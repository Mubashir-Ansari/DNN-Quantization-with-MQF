"""WANDA (Pruning by Weights And activations) implementation."""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional, Callable
from core.utils import calculate_sparsity
from torch.utils.data import DataLoader


class WANDAPruning:
    """
    WANDA: A Simple and Effective Pruning Approach for Large Language Models.
    Adapted for vision models (VGG).
    """
    
    def __init__(self, use_activation: bool = True, num_calibration_samples: int = 128):
        """
        Initialize WANDA pruning.
        
        Args:
            use_activation: Whether to use activation statistics
            num_calibration_samples: Number of samples for activation calibration
        """
        self.use_activation = use_activation
        self.num_calibration_samples = num_calibration_samples
        self.activation_stats = {}
    
    def collect_activation_statistics(self, model: nn.Module, dataloader: DataLoader, device: torch.device):
        """
        Collect activation statistics for WANDA scoring.
        
        Args:
            model: Model to collect statistics from
            dataloader: Calibration dataloader
            device: Device to run on
        """
        model.eval()
        self.activation_stats = {}
        
        # Register hooks to collect activations
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if name not in self.activation_stats:
                    self.activation_stats[name] = []
                
                # Store input activations for linear/conv layers
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if isinstance(module, nn.Linear):
                        # For linear layers, input is [batch_size, features]
                        act = input[0].detach()
                        if len(act.shape) > 2:
                            act = act.view(act.size(0), -1)
                    else:
                        # For conv layers, input is [batch_size, channels, h, w]
                        act = input[0].detach()
                        # The norm should be computed over the flattened spatial dimensions for each channel
                        # Reshape to [batch_size, channels, h*w] and then compute norm over the last dim
                        if act.dim() == 4:
                            act = act.permute(1, 0, 2, 3).flatten(1) # [channels, batch*h*w]
                        else:
                            act = act.permute(1, 0) # [channels, batch] for flattened inputs
                    
                    self.activation_stats[name].append(act.T)
            return hook
        
        # Register hooks for prunable layers
        for name, module in model.named_modules():
            if self._is_prunable_module(module):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Collect activation statistics
        print(f"Collecting activation statistics from {self.num_calibration_samples} samples...")
        
        with torch.no_grad():
            sample_count = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                if sample_count >= self.num_calibration_samples:
                    break
                
                data = data.to(device)
                model(data)
                sample_count += data.size(0)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process collected statistics
        self._process_activation_stats()
        
        print(f"✓ Collected activation statistics for {len(self.activation_stats)} layers")
    
    def _process_activation_stats(self):
        """Process collected activation statistics."""
        processed_stats = {}
        
        for layer_name, activations in self.activation_stats.items():
            if activations:
                # Concatenate all activations
                all_acts = torch.cat(activations, dim=0)
                
                # Calculate statistics (L2 norm per input dimension for WANDA)
                # Following original paper: X.norm(p=2, dim=0)
                act_stats = torch.norm(all_acts, p=2, dim=0)
                processed_stats[layer_name] = act_stats.pow(2) # Use squared L2 norm for stability
        
        self.activation_stats = processed_stats
    
    def calculate_wanda_scores(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Calculate WANDA importance scores.
        
        Args:
            model: Model to calculate scores for
            
        Returns:
            Dictionary of importance scores per layer
        """
        importance_scores = {}
        
        for name, module in model.named_modules():
            if self._is_prunable_module(module):
                weight = module.weight.data
                
                if self.use_activation and name in self.activation_stats:
                    # WANDA score: |weight| * activation_norm
                    # This follows the original WANDA paper: W_metric = |W| * X.norm(p=2, dim=0)
                    act_stats = self.activation_stats[name]
                    
                    # Add a small epsilon to avoid division by zero
                    act_stats += 1e-8
                    
                    if isinstance(module, nn.Linear):
                        # For linear layers: weight is [out_features, in_features]
                        # act_stats is [in_features] (L2 norm per input feature)
                        act_stats = act_stats.view(1, -1)  # [1, in_features]
                        scores = torch.abs(module.weight) / torch.sqrt(act_stats)
                    
                    elif isinstance(module, nn.Conv2d):
                        # For conv layers: weight is [out_channels, in_channels, h, w]
                        # act_stats is [in_channels] (L2 norm per input channel)
                        act_stats = act_stats.view(1, -1, 1, 1)  # [1, in_channels, 1, 1]
                        scores = torch.abs(module.weight) / torch.sqrt(act_stats)
                    
                    else:
                        scores = torch.abs(module.weight)
                else:
                    # Fallback to magnitude-based scoring
                    scores = torch.abs(weight)
                
                importance_scores[f"{name}.weight"] = scores
                print(f"DEBUG: WANDA {name}.weight scores - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}")
        
        return importance_scores
    
    def get_wanda_pruning_mask(self, model: nn.Module, sparsity_ratio: float, 
                              global_pruning: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate WANDA-based pruning masks.
        
        Args:
            model: Model to generate masks for
            sparsity_ratio: Ratio of weights to prune (0-1)
            global_pruning: Whether to prune globally or layer-wise
            
        Returns:
            Dictionary of pruning masks
        """
        importance_scores = self.calculate_wanda_scores(model)
        
        if global_pruning:
            return self._global_wanda_pruning(importance_scores, sparsity_ratio)
        else:
            return self._layer_wise_wanda_pruning(importance_scores, sparsity_ratio)
    
    def _global_wanda_pruning(self, importance_scores: Dict[str, torch.Tensor], 
                             sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Global WANDA pruning across all layers."""
        pruning_masks = {}
        
        # Flatten all scores
        all_scores = []
        layer_info = []
        
        for layer_name, scores in importance_scores.items():
            scores_flat = scores.view(-1)
            all_scores.append(scores_flat)
            
            layer_info.append({
                'name': layer_name,
                'shape': scores.shape,
                'start_idx': len(torch.cat(all_scores[:-1])) if len(all_scores) > 1 else 0,
                'end_idx': len(torch.cat(all_scores))
            })
        
        if not all_scores:
            return pruning_masks
        
        # Find global threshold
        all_scores_cat = torch.cat(all_scores)
        num_params_to_prune = int(len(all_scores_cat) * sparsity_ratio)
        
        if num_params_to_prune > 0:
            # Find threshold (keep weights with highest scores)
            threshold = torch.kthvalue(all_scores_cat, num_params_to_prune)[0]
            
            # Create masks
            for layer in layer_info:
                layer_scores = all_scores_cat[layer['start_idx']:layer['end_idx']]
                mask = (layer_scores >= threshold).float()
                mask = mask.view(layer['shape'])
                pruning_masks[layer['name']] = mask
        else:
            # No pruning
            for layer in layer_info:
                pruning_masks[layer['name']] = torch.ones(layer['shape'])
        
        return pruning_masks
    
    def _layer_wise_wanda_pruning(self, importance_scores: Dict[str, torch.Tensor], 
                                 sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Layer-wise WANDA pruning."""
        pruning_masks = {}
        
        for layer_name, scores in importance_scores.items():
            scores_flat = scores.view(-1)
            num_params_to_prune = int(len(scores_flat) * sparsity_ratio)
            
            if num_params_to_prune > 0:
                threshold = torch.kthvalue(scores_flat, num_params_to_prune)[0]
                mask = (scores_flat > threshold).float()
                mask = mask.view(scores.shape)
            else:
                mask = torch.ones_like(scores)
            
            pruning_masks[layer_name] = mask
        
        return pruning_masks
    
    def _is_prunable_module(self, module: nn.Module) -> bool:
        """Check if module is prunable."""
        return isinstance(module, (nn.Linear, nn.Conv2d))
    
    def apply_pruning_masks(self, model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """Apply pruning masks to model."""
        pruned_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for mask_name, mask in pruning_masks.items():
                # Find corresponding parameter
                param_found = False
                for param_name, param in pruned_model.named_parameters():
                    if param_name == mask_name:
                        param.data.mul_(mask.to(param.device))
                        param_found = True
                        break
                
                if not param_found:
                    print(f"Warning: Could not find parameter for mask {mask_name}")
        
        return pruned_model
    
    def prune_model(self, model: nn.Module, target_sparsity: float, 
                   calibration_dataloader: DataLoader, device: torch.device,
                   global_pruning: bool = True) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Complete WANDA pruning pipeline.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)  
            calibration_dataloader: Data for activation statistics
            device: Device to run on
            global_pruning: Whether to use global pruning
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        print(f"DEBUG: WANDA prune_model() received target_sparsity={target_sparsity}")
        sparsity_ratio = target_sparsity / 100.0
        print(f"DEBUG: converted to sparsity_ratio={sparsity_ratio}")
        
        # Collect activation statistics if needed
        if self.use_activation:
            self.collect_activation_statistics(model, calibration_dataloader, device)
        
        # Generate WANDA masks
        pruning_masks = self.get_wanda_pruning_mask(model, sparsity_ratio, global_pruning)
        
        # Apply masks
        pruned_model = self.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        activation_str = "with activations" if self.use_activation else "magnitude-only"
        print(f"WANDA pruning ({activation_str}): Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


class AdaptiveWANDA(WANDAPruning):
    """Adaptive WANDA with layer-specific activation weighting."""
    
    def __init__(self, use_activation: bool = True, num_calibration_samples: int = 128,
                 adaptive_factor: float = 0.5):
        """
        Initialize Adaptive WANDA.
        
        Args:
            use_activation: Whether to use activation statistics
            num_calibration_samples: Number of calibration samples
            adaptive_factor: How much to weight activation vs magnitude
        """
        super().__init__(use_activation, num_calibration_samples)
        self.adaptive_factor = adaptive_factor
    
    def calculate_wanda_scores(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Calculate adaptive WANDA scores with layer-specific weighting."""
        importance_scores = {}
        
        for name, module in model.named_modules():
            if self._is_prunable_module(module):
                weight = module.weight.data
                magnitude_scores = torch.abs(weight)
                
                if self.use_activation and name in self.activation_stats:
                    act_stats = self.activation_stats[name]
                    
                    # Calculate activation contribution
                    if isinstance(module, nn.Linear):
                        act_stats = act_stats.view(1, -1)
                        activation_scores = torch.sqrt(act_stats + 1e-8)
                    elif isinstance(module, nn.Conv2d):
                        act_stats = act_stats.view(1, -1, 1, 1)
                        activation_scores = torch.sqrt(act_stats + 1e-8)
                    else:
                        activation_scores = torch.ones_like(magnitude_scores)
                    
                    # Adaptive combination
                    layer_depth = self._estimate_layer_depth(name)
                    depth_factor = min(1.0, layer_depth / 10.0)  # Deeper layers get more activation weight
                    
                    final_adaptive_factor = self.adaptive_factor * depth_factor
                    scores = ((1 - final_adaptive_factor) * magnitude_scores + 
                             final_adaptive_factor * magnitude_scores * activation_scores)
                else:
                    scores = magnitude_scores
                
                importance_scores[f"{name}.weight"] = scores
        
        return importance_scores
    
    def _estimate_layer_depth(self, layer_name: str) -> int:
        """Estimate layer depth from name (simple heuristic)."""
        if 'features' in layer_name:
            # Extract number from features.X.weight
            try:
                parts = layer_name.split('.')
                for part in parts:
                    if part.isdigit():
                        return int(part)
            except:
                pass
        elif 'classifier' in layer_name:
            return 20  # Assume classifier is deep
        
        return 1  # Default depth


def create_wanda_variants() -> Dict[str, WANDAPruning]:
    """Create different variants of WANDA pruning."""
    return {
        'wanda_magnitude': WANDAPruning(use_activation=False),
        'wanda_activation': WANDAPruning(use_activation=True),
        'wanda_adaptive': AdaptiveWANDA(use_activation=True, adaptive_factor=0.7),
        'wanda_conservative': WANDAPruning(use_activation=True, num_calibration_samples=256)
    }