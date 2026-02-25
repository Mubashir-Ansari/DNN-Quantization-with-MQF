import torch
import torch.nn as nn
import argparse
import os
import sys
import copy

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loaders import load_model
from evaluation.pipeline import get_cifar10_dataloader, get_cifar100_dataloader, evaluate_accuracy

# --- INLINED MAGNITUDE PRUNING (To avoid sklearn dependency) ---
class MagnitudePruning:
    """Implementation of magnitude-based pruning methods (Inlined)."""
    
    def __init__(self):
        pass
    
    def _is_prunable_layer(self, name, param):
        """Check if layer is prunable."""
        return (
            param.requires_grad and 
            len(param.shape) > 1 and  # Not bias terms
            'weight' in name and
            ('conv' in name.lower() or 'linear' in name.lower() or 
             'features' in name or 'classifier' in name)
        )

    def prune_model(self, model, target_sparsity, global_pruning=True):
        """
        Complete pruning pipeline.
        Args:
            model: Model to prune
            target_sparsity: Target sparsity percentage (0-100)
        """
        sparsity_ratio = target_sparsity / 100.0
        
        # 1. Collect all weights
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

        if not all_weights:
            return model, {}
            
        # 2. Find Threshold
        all_weights_cat = torch.cat(all_weights)
        num_params_to_prune = int(len(all_weights_cat) * sparsity_ratio)
        
        if num_params_to_prune > 0:
            threshold = torch.kthvalue(all_weights_cat, num_params_to_prune)[0]
            
            # 3. Apply Masks
            pruned_model = copy.deepcopy(model)
            with torch.no_grad():
                for layer in layer_info:
                    start, end = layer['start_idx'], layer['end_idx']
                    layer_weights = all_weights_cat[start:end]
                    mask = (layer_weights > threshold).float().view(layer['shape'])
                    
                    # Apply mask to model
                    if layer['name'] in dict(pruned_model.named_parameters()):
                        param = dict(pruned_model.named_parameters())[layer['name']]
                        param.data.mul_(mask.to(param.device))
        else:
            pruned_model = copy.deepcopy(model)

        return pruned_model, {}

# --- BATCH RUNNER ---
MODELS_TO_PRUNE = [
    {
        'name': 'levit',
        'checkpoint': './models/best3_levit_model_cifar10.pth',
        'dataset': 'cifar10',
        'input_size': 224
    },
    {
        'name': 'vgg11_bn',
        'checkpoint': './models/vgg11_bn.pt',
        'dataset': 'cifar10',
        'input_size': 32
    },
    {
        'name': 'swin',
        'checkpoint': './models/best_swin_model_cifar_changed.pth',
        'dataset': 'cifar100',
        'input_size': 224
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Prune Models (50% Sparsity)')
    parser.add_argument('--sparsity', type=float, default=50.0, help='Target sparsity percentage (default: 50.0)')
    args = parser.parse_args()
    
    print(f"=== Specifying Batch Pruning (Target: {args.sparsity}%) ===")
    
    for config in MODELS_TO_PRUNE:
        print("\n" + "="*50)
        print(f"Processing Model: {config['name']}")
        print("="*50)
        
        # 1. Load Model
        try:
            num_classes = 100 if config['dataset'] == 'cifar100' else 10
            model = load_model(config['name'], checkpoint_path=config['checkpoint'], num_classes=num_classes)
        except Exception as e:
            print(f"Skipping {config['name']}: Could not load checkpoint ({e})")
            continue
            
        # 2. Load Data
        data_root = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/Paper2/data'
        if config['dataset'] == 'cifar100':
            val_loader = get_cifar100_dataloader(train=False, input_size=config['input_size'], root=data_root)
        else:
            val_loader = get_cifar10_dataloader(train=False, input_size=config['input_size'], root=data_root)
            
        # 3. Baseline Eval
        print("Measuring Baseline Accuracy...")
        acc_base = evaluate_accuracy(model, val_loader)
        print(f"Baseline: {acc_base:.2f}%")
        
        # 4. Prune
        print(f"Applying Global Pruning ({args.sparsity}%)...")
        pruner = MagnitudePruning()
        pruned_model, _ = pruner.prune_model(model, target_sparsity=args.sparsity, global_pruning=True)
        
        # 5. Pruned Eval
        print("Measuring Pruned Accuracy...")
        acc_pruned = evaluate_accuracy(pruned_model, val_loader)
        print(f"Pruned: {acc_pruned:.2f}% (Drop: {acc_base - acc_pruned:.2f}%)")
        
        # 6. Save
        output_name = f"models/{config['name']}_pruned_{int(args.sparsity)}.pth"
        torch.save(pruned_model.state_dict(), output_name)
        print(f"Saved to: {output_name}")
        
    print("\nBatch Pruning Complete.")
