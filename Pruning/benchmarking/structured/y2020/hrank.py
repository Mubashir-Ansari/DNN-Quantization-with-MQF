"""HRank: Filter Pruning using Feature Map Rank."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import copy


class HRankPruning:
    """HRank-based structured filter pruning implementation."""
    
    def __init__(self, num_samples: int = 500, batch_size: int = 32):
        """
        Initialize HRank pruning.
        
        Args:
            num_samples: Number of samples to use for rank estimation
            batch_size: Batch size for rank estimation
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.feature_maps = {}
        self.hooks = []
    
    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks to capture feature maps."""
        def get_activation(name):
            def hook(model, input, output):
                if name not in self.feature_maps:
                    self.feature_maps[name] = []
                self.feature_maps[name].append(output.detach().cpu())
            return hook
        
        # Register hooks for all convolutional layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _calculate_feature_map_ranks(self, feature_maps: torch.Tensor) -> List[float]:
        """
        Calculate average rank for each filter's feature maps.
        
        Args:
            feature_maps: Tensor of shape (batch, channels, height, width)
            
        Returns:
            List of average ranks for each filter
        """
        batch_size, num_filters, height, width = feature_maps.shape
        avg_ranks = []
        
        for filter_idx in range(num_filters):
            total_rank = 0
            for batch_idx in range(batch_size):
                # Get feature map for this filter and batch
                feature_map = feature_maps[batch_idx, filter_idx, :, :].numpy()
                
                # Calculate rank using SVD
                try:
                    _, s, _ = np.linalg.svd(feature_map, full_matrices=False)
                    rank = np.sum(s > 1e-6)  # Count non-zero singular values
                    total_rank += rank
                except:
                    # If SVD fails, assume full rank
                    total_rank += min(feature_map.shape)
            
            avg_rank = total_rank / batch_size
            avg_ranks.append(avg_rank)
        
        return avg_ranks
    
    def _estimate_ranks(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, List[float]]:
        """
        Estimate feature map ranks for all convolutional layers.
        
        Args:
            model: The model to analyze
            dataloader: DataLoader for rank estimation
            
        Returns:
            Dictionary mapping layer names to average ranks per filter
        """
        model.eval()
        self.feature_maps = {}
        
        # Register hooks
        self._register_hooks(model)
        
        # Collect feature maps
        samples_processed = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if samples_processed >= self.num_samples:
                    break
                
                data = data.to(next(model.parameters()).device)
                _ = model(data)
                samples_processed += data.size(0)
        
        # Remove hooks
        self._remove_hooks()
        
        # Calculate ranks for each layer
        layer_ranks = {}
        for layer_name, feature_maps_list in self.feature_maps.items():
            # Concatenate all feature maps for this layer
            all_feature_maps = torch.cat(feature_maps_list, dim=0)
            # Limit to num_samples
            if all_feature_maps.size(0) > self.num_samples:
                all_feature_maps = all_feature_maps[:self.num_samples]
            
            # Calculate average ranks
            avg_ranks = self._calculate_feature_map_ranks(all_feature_maps)
            layer_ranks[layer_name] = avg_ranks
        
        return layer_ranks
    
    def prune_model(self, model: nn.Module, dataloader: DataLoader, 
                   sparsity_ratio: float = 0.5) -> nn.Module:
        print(f"DEBUG: HRANK prune_model() received sparsity_ratio={sparsity_ratio} (target_sparsity={sparsity_ratio*100:.2f}%)")
        """
        Prune model using HRank method.
        
        Args:
            model: Model to prune
            dataloader: DataLoader for rank estimation
            sparsity_ratio: Target sparsity ratio (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        print(f"DEBUG: HRankPruning prune_model() received sparsity_ratio={sparsity_ratio}")
        # Create a copy of the model
        pruned_model = copy.deepcopy(model)
        
        # Estimate ranks
        print("Estimating feature map ranks...")
        layer_ranks = self._estimate_ranks(model, dataloader)
        
        # Prune each layer
        for layer_name, ranks in layer_ranks.items():
            if len(ranks) == 0:
                continue
                
            # Determine number of filters to prune
            num_filters = len(ranks)
            num_to_prune = int(num_filters * sparsity_ratio)
            print(f"DEBUG: HRANK {layer_name}: {num_filters} filters, pruning {num_to_prune} ({sparsity_ratio*100:.1f}%)")
            
            if num_to_prune == 0:
                continue
            
            # Sort by rank (ascending - prune lowest ranks first)
            rank_indices = np.argsort(ranks)
            filters_to_prune = rank_indices[:num_to_prune]
            
            # Get the actual layer in the pruned model
            layer_parts = layer_name.split('.')
            current_module = pruned_model
            for part in layer_parts:
                current_module = getattr(current_module, part)
            
            # Prune filters (set to zero)
            if isinstance(current_module, nn.Conv2d):
                total_params_layer = current_module.weight.numel()
                params_per_filter = current_module.weight.size(1) * current_module.weight.size(2) * current_module.weight.size(3)
                params_pruned = len(filters_to_prune) * params_per_filter
                print(f"DEBUG: HRANK {layer_name}: pruning {params_pruned}/{total_params_layer} parameters ({params_pruned/total_params_layer*100:.1f}%)")
                
                with torch.no_grad():
                    for filter_idx in filters_to_prune:
                        current_module.weight[filter_idx].zero_()
                        if current_module.bias is not None:
                            current_module.bias[filter_idx].zero_()
            
            elif isinstance(current_module, nn.Linear):
                # For Linear layers, treat output neurons as "filters"
                total_params_layer = current_module.weight.numel()
                params_per_neuron = current_module.weight.size(1)  # input features per output neuron
                params_pruned = len(filters_to_prune) * params_per_neuron
                print(f"DEBUG: HRANK {layer_name}: pruning {params_pruned}/{total_params_layer} parameters ({params_pruned/total_params_layer*100:.1f}%)")
                
                with torch.no_grad():
                    for neuron_idx in filters_to_prune:
                        current_module.weight[neuron_idx].zero_()  # Zero entire output neuron
                        if current_module.bias is not None:
                            current_module.bias[neuron_idx].zero_()
                
                print(f"Pruned {len(filters_to_prune)}/{num_filters} filters from {layer_name}")
        
        return pruned_model
    
    def get_pruning_masks(self, model: nn.Module, dataloader: DataLoader, 
                         sparsity_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Get pruning masks based on HRank method.
        
        Args:
            model: Model to analyze
            dataloader: DataLoader for rank estimation
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Dictionary of pruning masks for each layer
        """
        print(f"DEBUG: HRankPruning get_pruning_masks() received sparsity_ratio={sparsity_ratio}")
        masks = {}
        
        # Estimate ranks
        layer_ranks = self._estimate_ranks(model, dataloader)
        
        # Generate masks
        for layer_name, ranks in layer_ranks.items():
            if len(ranks) == 0:
                continue
                
            num_filters = len(ranks)
            num_to_prune = int(num_filters * sparsity_ratio)
            
            # Create mask (1 = keep, 0 = prune)
            mask = torch.ones(num_filters, dtype=torch.bool)
            
            if num_to_prune > 0:
                # Sort by rank and prune lowest
                rank_indices = np.argsort(ranks)
                filters_to_prune = rank_indices[:num_to_prune]
                mask[filters_to_prune] = False
            
            masks[layer_name] = mask
        
        return masks


def create_hrank_dataloader(train_loader: DataLoader, num_samples: int) -> DataLoader:
    """Create a smaller dataloader for HRank estimation."""
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