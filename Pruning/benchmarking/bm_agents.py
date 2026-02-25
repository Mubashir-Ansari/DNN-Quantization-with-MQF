"""Benchmarking-specific agents - decoupled from GA agents."""

import torch
import torch.nn as nn
import copy
from typing import Dict, Any
from core.utils import test_accuracy


class BenchmarkModelPruner:
    """Simple model pruner for benchmarking - no GA dependencies."""
    
    def prune_model(self, model_instance: nn.Module, 
                   pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """Apply pruning masks to model."""
        pruned_model = copy.deepcopy(model_instance)
        
        with torch.no_grad():
            for layer_name, mask in pruning_masks.items():
                if layer_name in pruned_model.state_dict():
                    weights = pruned_model.state_dict()[layer_name]
                    if weights.numel() == mask.numel():
                        # Apply mask
                        weights.data.view(-1).mul_(mask.to(weights.device))
                    else:
                        print(f"Warning: Shape mismatch for {layer_name}. "
                              f"Model: {weights.numel()}, Mask: {mask.numel()}. Skipping.")
        
        return pruned_model


class BenchmarkEvaluator:
    """Simple evaluator for benchmarking - no GA dependencies."""
    
    def __init__(self, val_loader, device):
        self.val_loader = val_loader
        self.device = device
    
    def evaluate(self, model: nn.Module, dummy_input=None) -> Dict[str, Any]:
        """Evaluate model and return metrics."""
        accuracy = test_accuracy(model, self.val_loader, self.device)
        
        # Calculate sparsity
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'sparsity': sparsity,
            'total_params': total_params,
            'zero_params': zero_params
        }