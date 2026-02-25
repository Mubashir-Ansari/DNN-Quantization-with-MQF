"""AdaSAP: Adaptive Sharpness-Aware Pruning implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import DataLoader
import copy


class AdaSAPPruning:
    """AdaSAP (Adaptive Sharpness-Aware Pruning) implementation."""
    
    def __init__(self, 
                 rho_min: float = 0.01,
                 rho_max: float = 2.0,
                 warmup_epochs: int = 5,
                 pruning_epochs: int = 10,
                 finetune_epochs: int = 15,
                 prune_frequency: int = 30,
                 learning_rate: float = 0.01):
        """
        Initialize AdaSAP pruning.
        
        Args:
            rho_min: Minimum perturbation radius
            rho_max: Maximum perturbation radius
            warmup_epochs: Epochs for adaptive weight perturbation phase
            pruning_epochs: Epochs for neuron removal phase
            finetune_epochs: Epochs for robustness encouragement phase
            prune_frequency: How often to prune (iterations)
            learning_rate: Learning rate for optimization
        """
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.warmup_epochs = warmup_epochs
        self.pruning_epochs = pruning_epochs
        self.finetune_epochs = finetune_epochs
        self.prune_frequency = prune_frequency
        self.learning_rate = learning_rate
    
    def _compute_neuron_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute neuron importance scores (L2 norm of weights).
        
        Args:
            weight: Weight tensor
            
        Returns:
            Importance scores for each neuron/filter
        """
        if len(weight.shape) == 4:  # Conv layer: [out_channels, in_channels, h, w]
            return torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
        elif len(weight.shape) == 2:  # Linear layer: [out_features, in_features]
            return torch.norm(weight, p=2, dim=1)
        else:
            return torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
    
    def _compute_adaptive_rho(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive perturbation radii based on importance scores.
        
        Args:
            importance_scores: Tensor of importance scores
            
        Returns:
            Adaptive rho values for each neuron
        """
        s_min = importance_scores.min()
        s_max = importance_scores.max()
        
        if s_max == s_min:
            return torch.full_like(importance_scores, self.rho_max)
        
        # Equation 2 from AdaSAP: rho_i = rho_max - ((si - s_min) / (s_max - s_min)) * (rho_max - rho_min)
        normalized_scores = (importance_scores - s_min) / (s_max - s_min)
        rho_values = self.rho_max - normalized_scores * (self.rho_max - self.rho_min)
        
        return rho_values
    
    def _compute_perturbation(self, weight: torch.Tensor, grad: torch.Tensor, 
                            rho: torch.Tensor) -> torch.Tensor:
        """
        Compute ASAM-based optimal weight perturbation epsilon_hat.
        
        Following AdaSAP paper Equation 5:
        epsilon_hat_i = rho_i * (T²_w * grad) / ||T²_w * grad||_2
        where T²_w = |w| + epsilon is the ASAM transformation matrix.
        
        Args:
            weight: Current weights
            grad: Gradient w.r.t. weights
            rho: Perturbation radius for each neuron
            
        Returns:
            ASAM perturbation tensor
        """
        # Reshape rho to match weight dimensions
        if len(weight.shape) == 4:  # Conv layer
            rho = rho.view(-1, 1, 1, 1)
        elif len(weight.shape) == 2:  # Linear layer
            rho = rho.view(-1, 1)
        
        # Compute L2 norm of gradient for each neuron
        if len(grad.shape) == 4:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
        elif len(grad.shape) == 2:
            grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
        
        # Avoid division by zero
        grad_norm = torch.clamp(grad_norm, min=1e-8)
        
        # ASAM transformation: T²_w = |w| + epsilon (element-wise)
        # This is the key difference from SAM - we scale by weight magnitude
        epsilon_small = 1e-8
        weight_abs = torch.abs(weight) + epsilon_small
        
        # Apply ASAM transformation to gradient: T²_w * grad
        transformed_grad = weight_abs * grad
        
        # Recompute norm with transformed gradient
        if len(transformed_grad.shape) == 4:
            transformed_grad_norm = torch.norm(transformed_grad.view(transformed_grad.size(0), -1), 
                                             p=2, dim=1, keepdim=True)
            transformed_grad_norm = transformed_grad_norm.view(-1, 1, 1, 1)
        elif len(transformed_grad.shape) == 2:
            transformed_grad_norm = torch.norm(transformed_grad, p=2, dim=1, keepdim=True)
        else:
            transformed_grad_norm = torch.norm(transformed_grad.view(transformed_grad.size(0), -1), 
                                             p=2, dim=1, keepdim=True)
        
        transformed_grad_norm = torch.clamp(transformed_grad_norm, min=epsilon_small)
        
        # Compute ASAM perturbation: epsilon_hat = rho * (T²_w * grad) / ||T²_w * grad||
        epsilon_hat = rho * transformed_grad / transformed_grad_norm
        
        return epsilon_hat
    
    def _adasap_optimization_step(self, model: nn.Module, data: torch.Tensor, 
                                target: torch.Tensor, optimizer: optim.Optimizer,
                                rho_min: float, rho_max: float, 
                                criterion: nn.Module) -> float:
        """
        Perform one AdaSAP optimization step.
        
        Args:
            model: Neural network model
            data: Input data
            target: Target labels
            optimizer: Optimizer
            rho_min: Minimum perturbation radius
            rho_max: Maximum perturbation radius
            criterion: Loss function
            
        Returns:
            Loss value
        """
        model.train()
        optimizer.zero_grad()
        
        # Store original weights
        original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:
                original_weights[name] = param.data.clone()
        
        # First forward pass to compute gradients
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Compute perturbations for each layer
        perturbations = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1 and param.grad is not None:
                # Compute importance scores
                importance_scores = self._compute_neuron_importance(param.data)
                
                # Compute adaptive rho values
                if rho_min == rho_max:
                    # Uniform perturbation (finetuning phase)
                    rho_values = torch.full_like(importance_scores, rho_max)
                else:
                    # Adaptive perturbation (warmup/pruning phase)
                    rho_values = self._compute_adaptive_rho(importance_scores)
                
                # Compute perturbation
                epsilon_hat = self._compute_perturbation(param.data, param.grad, rho_values)
                perturbations[name] = epsilon_hat
                
                # Apply perturbation
                param.data.add_(epsilon_hat)
        
        # Second forward pass with perturbed weights
        optimizer.zero_grad()
        output_perturbed = model(data)
        loss_perturbed = criterion(output_perturbed, target)
        loss_perturbed.backward()
        
        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])
        
        # Update weights with the gradient computed at perturbed point
        optimizer.step()
        
        return loss_perturbed.item()
    
    def prune_model(self, model: nn.Module, train_loader: DataLoader,
                   sparsity_ratio: float = 0.5,
                   criterion: Optional[nn.Module] = None) -> nn.Module:
        print(f"DEBUG: SPSD/AdaSAP prune_model() received sparsity_ratio={sparsity_ratio} (target_sparsity={sparsity_ratio*100:.2f}%)")
        """
        Prune model using AdaSAP method.
        
        Args:
            model: Model to prune
            train_loader: Training data loader
            sparsity_ratio: Target sparsity ratio
            criterion: Loss function
            
        Returns:
            Pruned model
        """
        print(f"DEBUG: AdaSAPPruning prune_model() received sparsity_ratio={sparsity_ratio}")
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        device = next(model.parameters()).device
        pruned_model = copy.deepcopy(model).to(device)
        optimizer = optim.SGD(pruned_model.parameters(), lr=self.learning_rate, 
                             momentum=0.9, weight_decay=1e-4)
        
        print("Starting AdaSAP pruning...")
        
        # Phase 1: Adaptive Weight Perturbations (Warmup)
        print(f"Phase 1: Warmup ({self.warmup_epochs} epochs)")
        for epoch in range(self.warmup_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                loss = self._adasap_optimization_step(
                    pruned_model, data, target, optimizer, 
                    self.rho_min, self.rho_max, criterion
                )
                total_loss += loss
            
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{self.warmup_epochs}, Loss: {avg_loss:.4f}")
        
        # Phase 2: Neuron Removal (Pruning)
        print(f"Phase 2: Pruning ({self.pruning_epochs} epochs)")
        iteration = 0
        
        for epoch in range(self.pruning_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Optimization step
                loss = self._adasap_optimization_step(
                    pruned_model, data, target, optimizer,
                    self.rho_min, self.rho_max, criterion
                )
                total_loss += loss
                
                # Pruning step
                if iteration % self.prune_frequency == 0:
                    self._prune_least_important_neurons(pruned_model, sparsity_ratio)
                
                iteration += 1
            
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{self.pruning_epochs}, Loss: {avg_loss:.4f}")
        
        # Phase 3: Robustness Encouragement (Finetuning)
        print(f"Phase 3: Finetuning ({self.finetune_epochs} epochs)")
        for epoch in range(self.finetune_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                loss = self._adasap_optimization_step(
                    pruned_model, data, target, optimizer,
                    self.rho_max, self.rho_max, criterion  # Uniform perturbation
                )
                total_loss += loss
                
                # CRITICAL: Re-enforce pruning masks after each optimization step
                self._prune_least_important_neurons(pruned_model, sparsity_ratio)
            
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{self.finetune_epochs}, Loss: {avg_loss:.4f}")
        
        # Debug final model sparsity
        from core.utils import calculate_sparsity
        final_sparsity = calculate_sparsity(pruned_model)
        print(f"DEBUG: SPSD returning model with {final_sparsity:.2f}% sparsity")
        return pruned_model
    
    def _prune_least_important_neurons(self, model: nn.Module, sparsity_ratio: float):
        """
        Perform structured pruning by removing entire neurons/channels.
        
        Following AdaSAP paper: removes neurons based on importance scores,
        implementing true structured pruning (channel-wise for conv, neuron-wise for linear).
        
        Args:
            model: Model to prune
            sparsity_ratio: Target sparsity ratio
        """
        # Calculate structured importance scores for each layer
        structured_masks = {}
        all_layer_scores = []
        layer_mapping = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Calculate importance score per output neuron/channel
                if isinstance(module, nn.Conv2d):
                    # For conv: importance per output channel
                    weights = module.weight.data  # [out_channels, in_channels, h, w]
                    importance_scores = torch.norm(weights.view(weights.size(0), -1), p=2, dim=1)
                elif isinstance(module, nn.Linear):
                    # For linear: importance per output neuron  
                    weights = module.weight.data  # [out_features, in_features]
                    importance_scores = torch.norm(weights, p=2, dim=1)
                
                # Store for global ranking
                num_neurons = len(importance_scores)
                all_layer_scores.extend(importance_scores.cpu().numpy())
                layer_mapping.extend([(name, i) for i in range(num_neurons)])
        
        if not all_layer_scores:
            print("No prunable layers found for structured pruning")
            return
        
        # Global ranking: find threshold for structured pruning
        all_scores = np.array(all_layer_scores)
        num_neurons_to_prune = int(len(all_scores) * sparsity_ratio)
        
        if num_neurons_to_prune > 0:
            # Sort scores and find threshold
            sorted_indices = np.argsort(all_scores)
            prune_indices = sorted_indices[:num_neurons_to_prune]
            
            # Create structured masks
            pruned_neurons = set()
            for idx in prune_indices:
                layer_name, neuron_idx = layer_mapping[idx]
                pruned_neurons.add((layer_name, neuron_idx))
            
            # Apply structured pruning masks
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Determine which neurons/channels to keep
                    total_neurons = module.weight.size(0)
                    keep_mask = torch.ones(total_neurons, dtype=torch.bool, device=module.weight.device)
                    
                    for neuron_idx in range(total_neurons):
                        if (name, neuron_idx) in pruned_neurons:
                            keep_mask[neuron_idx] = False
                    
                    # Apply structured mask (zero out entire neurons/channels)
                    if isinstance(module, nn.Conv2d):
                        # Zero out entire output channels
                        num_pruned = (~keep_mask).sum().item()
                        total_channels = keep_mask.numel()
                        print(f"DEBUG: SPSD zeroing {num_pruned}/{total_channels} channels in {name}")
                        module.weight.data[~keep_mask] = 0
                        if module.bias is not None:
                            module.bias.data[~keep_mask] = 0
                    elif isinstance(module, nn.Linear):
                        # Zero out entire output neurons
                        num_pruned = (~keep_mask).sum().item()
                        total_neurons = keep_mask.numel()
                        print(f"DEBUG: SPSD zeroing {num_pruned}/{total_neurons} neurons in {name}")
                        module.weight.data[~keep_mask] = 0
                        if module.bias is not None:
                            module.bias.data[~keep_mask] = 0
                    
                    structured_masks[name] = keep_mask
            
            # Print structured pruning statistics
            total_neurons = len(all_layer_scores)
            pruned_neurons_count = len(pruned_neurons)
            actual_prune_ratio = pruned_neurons_count / total_neurons
            
            print(f"  Structured pruning: {pruned_neurons_count}/{total_neurons} neurons/channels removed")
            print(f"  Actual structured sparsity: {actual_prune_ratio*100:.1f}%")
        
        return structured_masks


def magnitude_based_importance(weight: torch.Tensor) -> torch.Tensor:
    """Default importance function based on L2 norm of weights."""
    if len(weight.shape) == 4:  # Conv layer
        return torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
    elif len(weight.shape) == 2:  # Linear layer
        return torch.norm(weight, p=2, dim=1)
    else:
        return torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)