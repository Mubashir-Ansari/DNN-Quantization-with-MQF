"""
PDP: Parameter-free Differentiable Pruning (Corrected Implementation).

This implementation follows the official paper by introducing learnable scores `s`
for each weight, which are updated during a pruning-aware training process.
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple

class PDPPruner:
    def __init__(self, model: nn.Module, target_sparsity: float, train_loader, device: str,
                 warmup_epochs=5, pruning_epochs=15, lr=1e-3, tau_schedule='linear'):
        self.model = copy.deepcopy(model).to(device)
        self.target_sparsity = target_sparsity
        self.train_loader = train_loader
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.pruning_epochs = pruning_epochs
        self.lr = lr
        self.tau_schedule = tau_schedule

        # PDP's core: learnable scores, one for each weight
        # Initialize with small random values instead of zeros for better gradient flow
        self.scores = nn.ParameterDict()
        for name, param in self.model.named_parameters():
            if self._is_prunable(name, param):
                # Initialize scores with small random values centered around 0
                initial_scores = torch.randn_like(param) * 0.01
                self.scores[name.replace('.', '_')] = nn.Parameter(initial_scores)

    def _is_prunable(self, name: str, param: torch.Tensor) -> bool:
        return 'weight' in name and param.dim() > 1

    def _get_current_sparsity(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        if epoch >= self.warmup_epochs + self.pruning_epochs:
            return self.target_sparsity
        
        progress = (epoch - self.warmup_epochs) / self.pruning_epochs
        # Using the cubic schedule from the paper
        return self.target_sparsity * (1 - (1 - progress) ** 3)

    def _get_current_tau(self, epoch: int) -> float:
        # Temperature annealing for the soft mask
        if self.tau_schedule == 'linear':
            start_tau, end_tau = 1.0, 0.1
            if epoch < self.warmup_epochs:
                return start_tau
            progress = min(1.0, (epoch - self.warmup_epochs) / self.pruning_epochs)
            return start_tau - progress * (start_tau - end_tau)
        return 0.1 # Default fixed tau

    def _get_soft_masks(self, current_sparsity: float, tau: float) -> Dict[str, torch.Tensor]:
        masks = {}
        all_scores = torch.cat([s.flatten() for s in self.scores.values()])
        
        if current_sparsity > 0:
            num_to_prune = int(all_scores.numel() * current_sparsity)
            num_to_keep = all_scores.numel() - num_to_prune
            threshold = torch.kthvalue(all_scores, num_to_keep).values  # Find threshold to keep top weights
            
            for name, score in self.scores.items():
                masks[name] = torch.sigmoid((score - threshold) / tau)
        else:
            for name, score in self.scores.items():
                masks[name] = torch.ones_like(score)
        
        return masks

    def train_and_prune(self) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        print("Starting PDP training and pruning process...")
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.scores.parameters(), 'lr': self.lr * 10} # Scores learn faster
        ], lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        total_epochs = self.warmup_epochs + self.pruning_epochs
        for epoch in range(total_epochs):
            self.model.train()
            current_sparsity = self._get_current_sparsity(epoch)
            tau = self._get_current_tau(epoch)
            soft_masks = self._get_soft_masks(current_sparsity, tau)
            
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply soft masks
                original_weights = {}
                for name, param in self.model.named_parameters():
                    if self._is_prunable(name, param):
                        score_name = name.replace('.', '_')
                        original_weights[name] = param.data.clone()
                        param.data *= soft_masks[score_name]

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # Restore weights before optimizer step
                for name, param in self.model.named_parameters():
                     if self._is_prunable(name, param):
                        param.data = original_weights[name]

                optimizer.step()

                if i > 200: break # Increased training iterations for better accuracy

            print(f"Epoch {epoch+1}/{total_epochs} | Sparsity: {current_sparsity*100:.1f}% | Loss: {loss.item():.4f}")

        # Finalize pruning
        final_masks = self.get_final_masks()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if self._is_prunable(name, param):
                    param.data *= final_masks[name.replace('.', '_')]

        # Debug: Print final sparsity
        # Count only prunable parameters (same as masks)
        total_prunable_params = sum(mask.numel() for mask in final_masks.values())
        pruned_params = sum((mask == 0).sum().item() for mask in final_masks.values())
        actual_sparsity = pruned_params / total_prunable_params * 100 if total_prunable_params > 0 else 0
        print(f"PDP Final: Target={self.target_sparsity*100:.2f}%, Actual={actual_sparsity:.2f}%")
        
        return self.model, final_masks

    def get_final_masks(self) -> Dict[str, torch.Tensor]:
        binary_masks = {}
        all_scores = torch.cat([s.flatten() for s in self.scores.values()])
        num_to_prune = int(all_scores.numel() * self.target_sparsity)
        num_to_keep = all_scores.numel() - num_to_prune
        threshold = torch.kthvalue(all_scores, num_to_keep).values  # Find threshold to keep top weights
        
        for name, score in self.scores.items():
            binary_masks[name] = (score >= threshold).float()  # Keep weights >= threshold
            
        return binary_masks

class PDPPruning:
    """Wrapper to integrate the new PDPPruner into the benchmark runner."""
    def __init__(self, warmup_epochs=3, pruning_epochs=8, learning_rate=1e-3):
        self.warmup_epochs = warmup_epochs
        self.pruning_epochs = pruning_epochs
        self.lr = learning_rate

    def prune_model(self, model: nn.Module, target_sparsity: float, 
                    train_loader: nn.Module, device: str) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        
        sparsity_ratio = target_sparsity / 100.0
        
        pdp_pruner = PDPPruner(
            model=model,
            target_sparsity=sparsity_ratio,
            train_loader=train_loader,
            device=device,
            warmup_epochs=self.warmup_epochs,
            pruning_epochs=self.pruning_epochs,
            lr=self.lr
        )

        pruned_model, final_masks_internal = pdp_pruner.train_and_prune()

        # Adapt mask format for benchmark runner
        final_masks = {}
        for name, param in pruned_model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                mask_name = name.replace('.', '_')
                if mask_name in final_masks_internal:
                    final_masks[name] = final_masks_internal[mask_name]

        return pruned_model, final_masks
