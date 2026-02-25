"""SNIP (Single-shot Network Pruning) implementation."""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
from core.utils import calculate_sparsity
from torch.utils.data import DataLoader


class SNIPPruning:
    """
    SNIP: Single-shot Network Pruning based on connection sensitivity.
    """
    
    def __init__(self, num_samples: int = 1):
        """
        Initialize SNIP pruning.
        
        Args:
            num_samples: Number of samples for gradient computation
        """
        self.num_samples = num_samples
        self.gradient_scores = {}
    
    def compute_snip_scores(self, model: nn.Module, dataloader: DataLoader, 
                           device: torch.device, loss_fn: nn.Module = None) -> Dict[str, torch.Tensor]:
        """
        Compute SNIP connection sensitivity scores.
        
        SNIP measures the sensitivity of the loss function to the removal of each connection.
        The sensitivity is calculated as ∂L/∂c_j where c_j is the connection indicator.
        
        Args:
            model: Model to compute scores for
            dataloader: Data loader for gradient computation
            device: Device to run on
            loss_fn: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Dictionary of connection sensitivity scores per layer
        """
        model.train()
        model.to(device)
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Create connection indicators (masks) for each prunable parameter
        connection_indicators = {}
        original_params = {}
        
        for name, param in model.named_parameters():
            if self._is_prunable_parameter(name, param):
                # Store original parameter
                original_params[name] = param.data.clone()
                
                # Create connection indicator: c_j ∈ {0,1} for each weight
                # Initialize as ones (all connections present)
                connection_indicator = torch.ones_like(param, requires_grad=True, device=device)
                connection_indicators[name] = connection_indicator
                
                # Modify parameter to be: w_j = c_j * w_j^original
                # This allows us to compute ∂L/∂c_j
                param.data = connection_indicator * original_params[name]
        
        # Zero out gradients
        model.zero_grad()
        
        # Accumulate gradients over samples
        total_loss = 0
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= self.num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Forward pass with masked weights
            output = model(data)
            loss = loss_fn(output, target)
            
            # Backward pass to compute gradients w.r.t connection indicators
            loss.backward()
            
            total_loss += loss.item()
            sample_count += data.size(0)
        
        # Extract SNIP connection sensitivity scores
        # SNIP score = |∂L/∂c_j| where c_j is the connection indicator
        sensitivity_scores = {}
        
        for name, connection_indicator in connection_indicators.items():
            if connection_indicator.grad is not None:
                # Connection sensitivity: absolute gradient w.r.t. connection indicator
                sensitivity_scores[name] = torch.abs(connection_indicator.grad).detach().clone()
            else:
                # Fallback to magnitude-based importance if no gradient
                original_param = original_params[name]
                sensitivity_scores[name] = torch.abs(original_param)
        
        # Normalize scores to be comparable across layers (as in some reference implementations)
        all_scores_flat = torch.cat([scores.view(-1) for scores in sensitivity_scores.values()])
        if not torch.all(all_scores_flat == 0):
            total_norm = torch.norm(all_scores_flat, p=1)
            if total_norm > 0:
                for name in sensitivity_scores:
                    sensitivity_scores[name] /= total_norm
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        
        # Clear gradient computation graph
        for connection_indicator in connection_indicators.values():
            connection_indicator.grad = None
        
        print(f"✓ Computed SNIP connection sensitivity scores using {sample_count} samples")
        return sensitivity_scores
    
    def get_snip_pruning_mask(self, sensitivity_scores: Dict[str, torch.Tensor], 
                             sparsity_ratio: float, global_pruning: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate pruning masks based on SNIP scores.
        
        Args:
            sensitivity_scores: SNIP sensitivity scores
            sparsity_ratio: Ratio of weights to prune (0-1)
            global_pruning: Whether to prune globally or layer-wise
            
        Returns:
            Dictionary of pruning masks
        """
        if global_pruning:
            return self._global_snip_pruning(sensitivity_scores, sparsity_ratio)
        else:
            return self._layer_wise_snip_pruning(sensitivity_scores, sparsity_ratio)
    
    def _global_snip_pruning(self, sensitivity_scores: Dict[str, torch.Tensor], 
                            sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Global SNIP pruning across all layers."""
        pruning_masks = {}
        
        # Flatten all scores
        all_scores = []
        layer_info = []
        
        for layer_name, scores in sensitivity_scores.items():
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
        print(f"DEBUG: SNIP total_params={len(all_scores_cat)}, to_prune={num_params_to_prune}, sparsity_ratio={sparsity_ratio}")
        
        if num_params_to_prune > 0:
            # Keep weights with highest SNIP scores
            threshold = torch.kthvalue(all_scores_cat, num_params_to_prune)[0]
            print(f"DEBUG: SNIP threshold={threshold}, scores min={all_scores_cat.min()}, max={all_scores_cat.max()}")
            
            # Create masks: keep weights with scores > threshold (remove <= threshold)  
            for layer in layer_info:
                layer_scores = all_scores_cat[layer['start_idx']:layer['end_idx']]
                mask = (layer_scores > threshold).float()
                mask = mask.view(layer['shape'])
                pruning_masks[layer['name']] = mask
        else:
            # No pruning
            for layer in layer_info:
                pruning_masks[layer['name']] = torch.ones(layer['shape'])
        
        return pruning_masks
    
    def _layer_wise_snip_pruning(self, sensitivity_scores: Dict[str, torch.Tensor], 
                                sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Layer-wise SNIP pruning."""
        pruning_masks = {}
        
        for layer_name, scores in sensitivity_scores.items():
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
    
    def _is_prunable_parameter(self, name: str, param: torch.Tensor) -> bool:
        """Check if parameter is prunable."""
        return (
            param.requires_grad and 
            len(param.shape) > 1 and  # Not bias terms
            'weight' in name and
            ('conv' in name.lower() or 'linear' in name.lower() or 
             'features' in name or 'classifier' in name)
        )
    
    def apply_pruning_masks(self, model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """Apply pruning masks to model."""
        pruned_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for name, mask in pruning_masks.items():
                if name in dict(pruned_model.named_parameters()):
                    param = dict(pruned_model.named_parameters())[name]
                    param.data.mul_(mask.to(param.device))
        
        return pruned_model
    
    def prune_model(self, model: nn.Module, target_sparsity: float, 
                   dataloader: DataLoader, device: torch.device,
                   global_pruning: bool = True) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Complete SNIP pruning pipeline.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)
            dataloader: Data loader for gradient computation
            device: Device to run on
            global_pruning: Whether to use global pruning
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        sparsity_ratio = target_sparsity / 100.0
        
        # Compute SNIP scores
        sensitivity_scores = self.compute_snip_scores(model, dataloader, device)
        
        # Generate masks
        pruning_masks = self.get_snip_pruning_mask(sensitivity_scores, sparsity_ratio, global_pruning)
        
        # Apply masks
        pruned_model = self.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"SNIP pruning: Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


class GraSPPruning:
    """
    GraSP: Gradient Signal Preservation for pruning.
    """
    
    def __init__(self, num_samples: int = 1, T: int = 200, reinitialize: bool = False):
        """
        Initialize GraSP pruning.
        
        Args:
            num_samples: Number of samples for gradient computation
            T: Temperature for Hessian approximation
            reinitialize: Whether to reinitialize weights after pruning
        """
        self.num_samples = num_samples
        self.T = T
        self.reinitialize = reinitialize
    
    def compute_grasp_scores(self, model: nn.Module, dataloader: DataLoader, 
                           device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute GraSP scores based on gradient flow preservation.
        
        Args:
            model: Model to compute scores for
            dataloader: Data loader for gradient computation
            device: Device to run on
            
        Returns:
            Dictionary of GraSP scores per layer
        """
        model.train()
        model.to(device)
        
        # Step 1: Compute gradients for original network
        original_gradients = {}
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= self.num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # FIXED: Get number of classes dynamically and handle different model structures
            try:
                # Try to get output features from last classifier layer
                if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                    num_classes = model.classifier[-1].out_features
                elif hasattr(model, 'fc'):
                    num_classes = model.fc.out_features
                else:
                    # Fallback: run a forward pass to determine output shape
                    with torch.no_grad():
                        test_output = model(data[:1])
                        num_classes = test_output.shape[1]
                
                # Random target for gradient computation (GraSP uses random labels)
                random_target = torch.randint_like(target, high=num_classes)
            except Exception as e:
                print(f"Warning: Could not determine num_classes, using original targets. Error: {e}")
                random_target = target
            
            model.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, random_target)
            loss.backward()
            
            # Store gradients
            for name, param in model.named_parameters():
                if self._is_prunable_parameter(name, param) and param.grad is not None:
                    if name not in original_gradients:
                        original_gradients[name] = torch.zeros_like(param.grad)
                    original_gradients[name] += param.grad
            
            sample_count += data.size(0)
        
        # FIXED: Normalize gradients by number of batches, not samples
        num_batches = batch_idx + 1
        for name in original_gradients:
            original_gradients[name] /= num_batches
        
        # Step 2: Compute GraSP scores using Hessian diagonal approximation
        grasp_scores = {}
        
        for name, param in model.named_parameters():
            if self._is_prunable_parameter(name, param):
                if name in original_gradients:
                    # FIXED: GraSP score should be based on the Hessian-gradient product, Hg.
                    # The paper shows this is approximated by (-g * w).
                    # We take the absolute value for the final score, so the sign doesn't matter.
                    grad_tensor = original_gradients[name].to(param.device)
                    weight_tensor = param.data  # Use the actual weight, not its absolute value yet
                    scores = torch.abs(grad_tensor * weight_tensor)
                    grasp_scores[name] = scores
                    
                    # Debug: Check if scores are reasonable
                    if scores.sum() == 0:
                        print(f"Warning: Zero GraSP scores for layer {name}")
                    
                else:
                    # Fallback to magnitude
                    print(f"Warning: No gradients found for {name}, using magnitude scores")
                    grasp_scores[name] = torch.abs(param.data)
        
        print(f"✓ Computed GraSP scores using {sample_count} samples across {num_batches} batches")
        return grasp_scores
    
    def get_grasp_pruning_mask(self, grasp_scores: Dict[str, torch.Tensor], 
                             sparsity_ratio: float, global_pruning: bool = True) -> Dict[str, torch.Tensor]:
        """Generate pruning masks based on GraSP scores."""
        if global_pruning:
            return self._global_grasp_pruning(grasp_scores, sparsity_ratio)
        else:
            return self._layer_wise_grasp_pruning(grasp_scores, sparsity_ratio)
    
    def _global_grasp_pruning(self, grasp_scores: Dict[str, torch.Tensor], 
                            sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Global GraSP pruning across all layers."""
        pruning_masks = {}
        
        # Flatten all scores
        all_scores = []
        layer_info = []
        
        for layer_name, scores in grasp_scores.items():
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
        print(f"DEBUG: GRASP total_params={len(all_scores_cat)}, to_prune={num_params_to_prune}")
        
        if num_params_to_prune > 0:
            # For GraSP: Keep weights with highest scores (most important)
            threshold = torch.kthvalue(all_scores_cat, num_params_to_prune)[0]
            print(f"DEBUG: GRASP threshold={threshold:.6f}, scores min={all_scores_cat.min():.6f}, max={all_scores_cat.max():.6f}")
            
            # Create masks - keep weights with scores > threshold
            for layer in layer_info:
                layer_scores = all_scores_cat[layer['start_idx']:layer['end_idx']]
                mask = (layer_scores > threshold).float()
                mask = mask.view(layer['shape'])
                pruning_masks[layer['name']] = mask
                
                kept_count = mask.sum().item()
                total_count = mask.numel()
                layer_sparsity = (total_count - kept_count) / total_count * 100
                print(f"DEBUG: GRASP {layer['name']}: kept={kept_count}/{total_count}, sparsity={layer_sparsity:.2f}%")
        else:
            # No pruning
            for layer in layer_info:
                pruning_masks[layer['name']] = torch.ones(layer['shape'])
        
        return pruning_masks
    
    def _layer_wise_grasp_pruning(self, grasp_scores: Dict[str, torch.Tensor], 
                                sparsity_ratio: float) -> Dict[str, torch.Tensor]:
        """Layer-wise GraSP pruning."""
        pruning_masks = {}
        
        for layer_name, scores in grasp_scores.items():
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
    
    def _is_prunable_parameter(self, name: str, param: torch.Tensor) -> bool:
        """Check if parameter is prunable."""
        return (
            param.requires_grad and 
            len(param.shape) > 1 and
            'weight' in name and
            ('conv' in name.lower() or 'linear' in name.lower() or 
             'features' in name or 'classifier' in name)
        )
    
    def apply_pruning_masks(self, model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """Apply pruning masks to model."""
        pruned_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for name, mask in pruning_masks.items():
                if name in dict(pruned_model.named_parameters()):
                    param = dict(pruned_model.named_parameters())[name]
                    param.data.mul_(mask.to(param.device))
                    
                    # Reinitialize remaining weights if specified (only the non-pruned weights)
                    if self.reinitialize and mask.sum() > 0:
                        remaining_weights = param.data[mask.bool()]
                        if len(remaining_weights) > 0:
                            std = remaining_weights.std()
                            # Only reinitialize the KEPT weights, leave pruned weights as zero
                            random_vals = torch.randn_like(param.data) * std
                            param.data = random_vals * mask  # Only reinitialize where mask=1
        
        return pruned_model
    
    def prune_model(self, model: nn.Module, target_sparsity: float, 
                   dataloader: DataLoader, device: torch.device,
                   global_pruning: bool = True) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Complete GraSP pruning pipeline.
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)
            dataloader: Data loader for gradient computation
            device: Device to run on
            global_pruning: Whether to use global pruning
            
        Returns:
            Tuple of (pruned_model, pruning_masks)
        """
        print(f"DEBUG: GRASP prune_model() received target_sparsity={target_sparsity}")
        sparsity_ratio = target_sparsity / 100.0
        print(f"DEBUG: converted to sparsity_ratio={sparsity_ratio}")
        
        # Compute GraSP scores
        grasp_scores = self.compute_grasp_scores(model, dataloader, device)
        
        # Generate masks
        pruning_masks = self.get_grasp_pruning_mask(grasp_scores, sparsity_ratio, global_pruning)
        
        # Apply masks
        print(f"DEBUG: GRASP applying {len(pruning_masks)} masks to model")
        pruned_model = self.apply_pruning_masks(model, pruning_masks)
        
        # Verify sparsity
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"DEBUG: GRASP after applying masks, sparsity = {actual_sparsity:.2f}%")
        reinit_str = "with reinit" if self.reinitialize else "no reinit"
        print(f"GraSP pruning ({reinit_str}): Target={target_sparsity:.1f}%, Actual={actual_sparsity:.1f}%")
        
        return pruned_model, pruning_masks


def create_snip_variants() -> Dict[str, object]:
    """Create different variants of SNIP and GraSP pruning."""
    return {
        'snip_single': SNIPPruning(num_samples=1),
        'snip_multi': SNIPPruning(num_samples=10),
        'grasp_default': GraSPPruning(num_samples=1, reinitialize=False),
        'grasp_no_reinit': GraSPPruning(num_samples=1, reinitialize=False),
        'grasp_multi': GraSPPruning(num_samples=10, reinitialize=True)
    }