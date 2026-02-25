"""DEGRAPH: Dependency Graph-based Structured Pruning implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from torch.utils.data import DataLoader
import copy
import networkx as nx


class DEGRAPHPruning:
    """DEGRAPH (Dependency Graph-based) structured pruning implementation."""
    
    def __init__(self, 
                 sparse_training_epochs: int = 10,
                 finetuning_epochs: int = 15,
                 alpha: float = 0.5,
                 learning_rate: float = 1e-3):
        """
        Initialize DEGRAPH pruning.
        
        Args:
            sparse_training_epochs: Epochs for sparse training
            finetuning_epochs: Epochs for final finetuning
            alpha: Hyperparameter balancing accuracy and efficiency
            learning_rate: Learning rate for training
        """
        self.sparse_training_epochs = sparse_training_epochs
        self.finetuning_epochs = finetuning_epochs
        self.alpha = alpha
        self.learning_rate = learning_rate
        
    def _decompose_network(self, model: nn.Module) -> Tuple[List[str], List[str]]:
        """
        Decompose network into input/output components.
        
        Args:
            model: Neural network model
            
        Returns:
            Tuple of (input_components, output_components)
        """
        input_components = []
        output_components = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                input_components.append(f"{name}_input")
                output_components.append(f"{name}_output")
        
        return input_components, output_components
    
    def _construct_dependency_graph(self, model: nn.Module) -> nx.Graph:
        """
        Construct dependency graph modeling layer inter-dependencies.
        
        Args:
            model: Neural network model
            
        Returns:
            NetworkX graph representing dependencies
        """
        graph = nx.Graph()
        
        # Get model structure
        modules = list(model.named_modules())
        module_names = [name for name, _ in modules if isinstance(_, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
        
        # Add nodes for each input/output component
        for name in module_names:
            input_node = f"{name}_input"
            output_node = f"{name}_output"
            graph.add_node(input_node)
            graph.add_node(output_node)
            
            # Add intra-layer dependency for layers with same pruning scheme
            # (e.g., BatchNorm layers)
            _, module = next((n, m) for n, m in modules if n == name)
            if isinstance(module, nn.BatchNorm2d):
                graph.add_edge(input_node, output_node)
        
        # Add inter-layer dependencies (connections between layers)
        for i, name1 in enumerate(module_names[:-1]):
            name2 = module_names[i + 1]
            
            # Connect output of current layer to input of next layer
            output_node1 = f"{name1}_output"
            input_node2 = f"{name2}_input"
            graph.add_edge(output_node1, input_node2)
        
        # Handle special connections (skip connections, etc.)
        # This is simplified - in practice, would need model-specific logic
        for name, module in model.named_modules():
            if hasattr(module, 'shortcut') or 'shortcut' in name.lower():
                # Add additional dependencies for skip connections
                pass
        
        return graph
    
    def _find_parameter_groups(self, dependency_graph: nx.Graph) -> List[Set[str]]:
        """
        Find groups of coupled parameters using connected components.
        
        Args:
            dependency_graph: Dependency graph
            
        Returns:
            List of parameter groups (sets of component names)
        """
        # Find connected components in the dependency graph
        connected_components = list(nx.connected_components(dependency_graph))
        
        # Convert to list of sets
        parameter_groups = [set(component) for component in connected_components]
        
        return parameter_groups
    
    def _calculate_group_importance(self, model: nn.Module, group: Set[str], 
                                  dataloader: DataLoader) -> float:
        """
        Calculate importance score for a parameter group.
        
        Args:
            model: Neural network model
            group: Set of component names in the group
            dataloader: Data loader for importance calculation
            
        Returns:
            Importance score for the group
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Extract layer names from group components
        layer_names = set()
        for component in group:
            layer_name = component.replace('_input', '').replace('_output', '')
            layer_names.add(layer_name)
        
        # Calculate importance based on weight magnitudes and gradients
        total_importance = 0.0
        num_samples = 0
        
        with torch.enable_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 10:  # Limit samples for efficiency
                    break
                    
                data, target = data.to(device), target.to(device)
                data.requires_grad_(True)
                
                model.zero_grad()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                # Calculate importance for layers in this group
                batch_importance = 0.0
                for name, module in model.named_modules():
                    if name in layer_names and isinstance(module, (nn.Conv2d, nn.Linear)):
                        if module.weight.grad is not None:
                            # Importance = |weight| * |gradient|
                            weight_importance = (module.weight.abs() * module.weight.grad.abs()).sum().item()
                            batch_importance += weight_importance
                
                total_importance += batch_importance
                num_samples += data.size(0)
        
        return total_importance / num_samples if num_samples > 0 else 0.0
    
    def _select_groups_to_prune(self, model: nn.Module, parameter_groups: List[Set[str]], 
                               target_speedup: float, dataloader: DataLoader) -> List[Set[str]]:
        """
        Select parameter groups to prune based on importance and target speedup.
        
        Args:
            model: Neural network model
            parameter_groups: List of parameter groups
            target_speedup: Target speedup ratio
            dataloader: Data loader for importance calculation
            
        Returns:
            List of groups selected for pruning
        """
        print("Calculating group importance scores...")
        
        # Calculate importance for each group
        group_importance = []
        for i, group in enumerate(parameter_groups):
            importance = self._calculate_group_importance(model, group, dataloader)
            group_importance.append((importance, i, group))
            print(f"  Group {i}: importance = {importance:.6f}")
        
        # Sort by importance (ascending - prune least important first)
        group_importance.sort(key=lambda x: x[0])
        
        # Select groups to prune based on target speedup
        # Simplified heuristic: prune until we reach approximately target speedup
        groups_to_prune = []
        current_speedup = 0.0
        
        for importance, idx, group in group_importance:
            if current_speedup >= target_speedup:
                break
            
            # Estimate speedup contribution of this group
            # Simplified: assume each group contributes proportionally
            estimated_contribution = 1.0 / len(parameter_groups)
            
            groups_to_prune.append(group)
            current_speedup += estimated_contribution
            
            print(f"  Selected group {idx} for pruning (importance: {importance:.6f})")
        
        return groups_to_prune
    
    def _apply_group_pruning(self, model: nn.Module, groups_to_prune: List[Set[str]]):
        """
        Apply structured pruning to selected parameter groups.
        
        Args:
            model: Neural network model
            groups_to_prune: List of groups to prune
        """
        print("Applying structured pruning to selected groups...")
        
        for group in groups_to_prune:
            # Extract layer names from group
            layer_names = set()
            for component in group:
                layer_name = component.replace('_input', '').replace('_output', '')
                layer_names.add(layer_name)
            
            # Apply pruning to layers in this group
            for name, module in model.named_modules():
                if name in layer_names:
                    if isinstance(module, nn.Conv2d):
                        # For conv layers, prune entire channels
                        num_channels = module.out_channels
                        # Simple strategy: prune 50% of channels
                        channels_to_keep = num_channels // 2
                        
                        with torch.no_grad():
                            # Keep channels with highest L1 norm
                            channel_norms = torch.norm(module.weight.view(num_channels, -1), p=1, dim=1)
                            _, keep_indices = torch.topk(channel_norms, channels_to_keep)
                            
                            # Create new weight tensor
                            new_weight = module.weight[keep_indices]
                            
                            # Update module (simplified - would need proper architecture modification)
                            module.weight.data = torch.zeros_like(module.weight.data)
                            module.weight.data[keep_indices] = new_weight
                    
                    elif isinstance(module, nn.Linear):
                        # For linear layers, prune neurons
                        num_neurons = module.out_features
                        neurons_to_keep = num_neurons // 2
                        
                        with torch.no_grad():
                            # Keep neurons with highest L1 norm
                            neuron_norms = torch.norm(module.weight, p=1, dim=1)
                            _, keep_indices = torch.topk(neuron_norms, neurons_to_keep)
                            
                            # Zero out pruned neurons
                            mask = torch.zeros(num_neurons, dtype=torch.bool)
                            mask[keep_indices] = True
                            module.weight.data[~mask] = 0
                            if module.bias is not None:
                                module.bias.data[~mask] = 0
    
    def _sparse_training(self, model: nn.Module, train_loader: DataLoader, 
                        criterion: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Perform sparse training phase.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
        """
        print(f"Starting sparse training ({self.sparse_training_epochs} epochs)...")
        
        model.train()
        for epoch in range(self.sparse_training_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:  # Limit batches for efficiency
                    break
                
                device = next(model.parameters()).device
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"  Epoch {epoch+1}/{self.sparse_training_epochs}, Loss: {avg_loss:.4f}")
    
    def prune_model(self, model: nn.Module, train_loader: DataLoader, 
                   sparsity_ratio: float = 0.5) -> nn.Module:
        """
        Prune model using DEGRAPH method.
        
        Args:
            model: Model to prune
            train_loader: Training data loader
            sparsity_ratio: Target sparsity ratio (converted to speedup)
            
        Returns:
            Pruned model
        """
        print("Starting DEGRAPH (Dependency Graph-based) pruning...")
        
        # Create copy of model
        pruned_model = copy.deepcopy(model)
        device = next(pruned_model.parameters()).device
        
        # Phase 1: Network decomposition
        print("Phase 1: Network decomposition...")
        input_components, output_components = self._decompose_network(pruned_model)
        print(f"  Identified {len(input_components)} input and {len(output_components)} output components")
        
        # Phase 2: Construct dependency graph
        print("Phase 2: Constructing dependency graph...")
        dependency_graph = self._construct_dependency_graph(pruned_model)
        print(f"  Dependency graph has {dependency_graph.number_of_nodes()} nodes and {dependency_graph.number_of_edges()} edges")
        
        # Phase 3: Find parameter groups
        print("Phase 3: Finding parameter groups...")
        parameter_groups = self._find_parameter_groups(dependency_graph)
        print(f"  Found {len(parameter_groups)} parameter groups")
        
        # Phase 4: Select groups to prune
        print("Phase 4: Selecting groups for pruning...")
        target_speedup = sparsity_ratio  # Use sparsity ratio as proxy for speedup
        groups_to_prune = self._select_groups_to_prune(
            pruned_model, parameter_groups, target_speedup, train_loader
        )
        print(f"  Selected {len(groups_to_prune)} groups for pruning")
        
        # Phase 5: Apply structured pruning
        print("Phase 5: Applying structured pruning...")
        self._apply_group_pruning(pruned_model, groups_to_prune)
        
        # Phase 6: Sparse training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=self.learning_rate)
        self._sparse_training(pruned_model, train_loader, criterion, optimizer)
        
        # Phase 7: Final evaluation and cleanup
        print("DEGRAPH pruning completed")
        
        return pruned_model
    
    def get_pruning_masks(self, model: nn.Module, train_loader: DataLoader, 
                         sparsity_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Get pruning masks using DEGRAPH method.
        
        Args:
            model: Model to analyze
            train_loader: Training data loader
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Dictionary of pruning masks
        """
        # For structured pruning, masks are more complex
        # This is a simplified version
        masks = {}
        
        # Construct dependency graph and find groups
        dependency_graph = self._construct_dependency_graph(model)
        parameter_groups = self._find_parameter_groups(dependency_graph)
        groups_to_prune = self._select_groups_to_prune(
            model, parameter_groups, sparsity_ratio, train_loader
        )
        
        # Create masks for selected groups
        layer_names_to_prune = set()
        for group in groups_to_prune:
            for component in group:
                layer_name = component.replace('_input', '').replace('_output', '')
                layer_names_to_prune.add(layer_name)
        
        # Generate masks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in layer_names_to_prune:
                    # Create mask that zeros out pruned elements
                    mask = torch.ones_like(module.weight, dtype=torch.bool)
                    
                    if isinstance(module, nn.Conv2d):
                        # Prune channels
                        num_channels = module.out_channels
                        channels_to_keep = num_channels // 2
                        channel_norms = torch.norm(module.weight.view(num_channels, -1), p=1, dim=1)
                        _, keep_indices = torch.topk(channel_norms, channels_to_keep)
                        
                        # Create channel mask
                        channel_mask = torch.zeros(num_channels, dtype=torch.bool)
                        channel_mask[keep_indices] = True
                        mask = channel_mask.view(-1, 1, 1, 1).expand_as(module.weight)
                    
                    elif isinstance(module, nn.Linear):
                        # Prune neurons
                        num_neurons = module.out_features
                        neurons_to_keep = num_neurons // 2
                        neuron_norms = torch.norm(module.weight, p=1, dim=1)
                        _, keep_indices = torch.topk(neuron_norms, neurons_to_keep)
                        
                        # Create neuron mask
                        neuron_mask = torch.zeros(num_neurons, dtype=torch.bool)
                        neuron_mask[keep_indices] = True
                        mask = neuron_mask.view(-1, 1).expand_as(module.weight)
                else:
                    # Keep all weights for layers not selected for pruning
                    mask = torch.ones_like(module.weight, dtype=torch.bool)
                
                masks[name] = mask
        
        return masks


def create_degraph_dataloader(train_loader: DataLoader, num_samples: int) -> DataLoader:
    """Create a smaller dataloader for DEGRAPH importance calculation."""
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