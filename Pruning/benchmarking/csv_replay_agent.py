"""Simple CSV replay agent for benchmarking."""

import torch
from typing import List, Dict
from genetic_algorithm.agents import get_layer_score_files_map


class CSVReplayAgent:
    """Simple agent that applies CSV percentages directly without any weight scoring."""
    
    def __init__(self, strategy_params: List[float], model_state_dict: Dict, score_dir_path: str):
        """
        Initialize CSV replay agent.
        
        Args:
            strategy_params: List of pruning percentages from CSV
            model_state_dict: Model state dictionary  
            score_dir_path: Path to sensitivity score directory
        """
        self.strategy_params = strategy_params
        self.model_state_dict = model_state_dict
        self.layer_score_files = get_layer_score_files_map(score_dir_path, model_state_dict)
        
        # Get layers that have sensitivity scores (for parameter mapping)
        self.scored_layers = sorted([
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name 
            and name in self.layer_score_files
        ])
        
        # Get ALL weight layers for mask generation (needed for fine-tuning)
        self.all_weight_layers = sorted([
            name for name in model_state_dict.keys()
            if ('features' in name or 'classifier' in name) and 'weight' in name
        ])
        
        # Use scored layers for parameter mapping
        self.prunable_layers = self.scored_layers
        
        print(f"DEBUG: CSVReplayAgent prunable_layers: {self.prunable_layers}")
        print(f"DEBUG: CSVReplayAgent strategy_params length: {len(self.strategy_params)}")
        
        if len(self.strategy_params) != len(self.prunable_layers):
            raise ValueError(f"Strategy params ({len(self.strategy_params)}) != prunable layers ({len(self.prunable_layers)})")
    
    def generate_pruning_mask(self, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """
        Generate pruning masks by applying CSV percentages directly.
        
        Args:
            device: Device to create masks on
            
        Returns:
            Dictionary of pruning masks
        """
        pruning_masks = {}
        
        # Apply CSV percentages to scored layers
        for i, layer_name in enumerate(self.prunable_layers):
            percentage = self.strategy_params[i]
            weights = self.model_state_dict[layer_name]
            
            print(f"DEBUG: {layer_name} - weights.shape={weights.shape}, weights.numel()={weights.numel()}")
            
            # Calculate how many weights to remove
            total_weights = weights.numel()
            num_to_remove = int(total_weights * percentage / 100.0)
            
            if num_to_remove == 0:
                # Keep all weights
                mask = torch.ones_like(weights)
            elif num_to_remove >= total_weights:
                # Remove all weights
                mask = torch.zeros_like(weights)
            else:
                # Remove specified percentage - just take first N weights (doesn't matter which ones)
                mask = torch.ones_like(weights)
                weights_flat = mask.view(-1)
                weights_flat[:num_to_remove] = 0
                mask = weights_flat.view(weights.shape)
            
            # Move to requested device
            mask = mask.to(device)
            
            print(f"DEBUG: {layer_name} - mask.shape={mask.shape}, mask.numel()={mask.numel()}")
            
            pruning_masks[layer_name] = mask
            
            # Log what we did
            actual_removed = (mask == 0).sum().item()
            actual_percentage = (actual_removed / total_weights) * 100
            print(f"  {layer_name}: {percentage:.1f}% -> {actual_removed}/{total_weights} pruned ({actual_percentage:.1f}%)")
        
        # Create identity masks for layers without sensitivity scores (to avoid tensor mismatch)
        for layer_name in self.all_weight_layers:
            if layer_name not in pruning_masks:
                weights = self.model_state_dict[layer_name]
                mask = torch.ones_like(weights).to(device)
                pruning_masks[layer_name] = mask
                print(f"  {layer_name}: 0.0% -> 0/{weights.numel()} pruned (0.0%) [no sensitivity scores]")
        
        return pruning_masks