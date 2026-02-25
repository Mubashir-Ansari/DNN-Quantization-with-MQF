"""
Hardware-Aware Mixed-Precision Search
Assigns bit-widths from user-specified choices based on layer sensitivity.
Architecture-agnostic: Works for VGG, ResNet, LeViT, Swin, and any other models.
"""

import argparse
import json
import csv
import torch
import torch.nn as nn
import sys
import os
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from evaluation.pipeline import get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader, get_fashionmnist_dataloader, evaluate_accuracy


def get_nested_attr(obj, attr):
    """
    Resolve nested attributes like 'features[0]' or 'conv1[0]'.
    """
    for part in attr.split("."):
        if "[" in part:
            p, i = part.split("[")
            obj = getattr(obj, p)[int(i[:-1])]
        else:
            obj = getattr(obj, part)
    return obj


def is_conv_layer(model, layer_name):
    """
    Check if a layer is a convolutional layer (sensitive to extreme quantization).
    Uses module type inspection instead of name matching for architecture-agnostic detection.
    
    Args:
        model: PyTorch model
        layer_name: Layer name to check
    
    Returns:
        True if Conv2d/Conv1d/Conv3d, False otherwise
    """
    for name, module in model.named_modules():
        if name == layer_name:
            # Conv layers are sensitive to 2-bit quantization
            return isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d))
    return False


def is_accumulator_safe(model, layer_name, bits, register_width=16):
    """
    Checks if a bit-width is safe for a 16-bit accumulator given K.
    S = ceil(log2(K * MaxProd)) + 1
    """
    for name, module in model.named_modules():
        if name == layer_name:
            if isinstance(module, nn.Conv2d):
                K = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                K = module.in_features
            else:
                return True # Default safe for pools etc.
            
            # S calculation
            max_val = (1 << (bits - 1))
            max_prod = max_val * max_val
            max_sum = K * max_prod
            S = math.ceil(math.log2(max_sum)) + 1 if max_sum > 0 else 1
            
            return S <= register_width
    return True


def load_sensitivity_profile(profile_csv):
    """
    Load sensitivity scores from CSV.
    Returns dict: {layer_name: {bit_width: sensitivity_score}}
    """
    layer_data = {}
    
    with open(profile_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_name = row['layer']
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            
            # Read sensitivity for different bit-widths
            for key in row.keys():
                if key.startswith('sensitivity_') and 'bit' in key:
                    # Extract bit-width from column name (e.g., 'sensitivity_2bit' -> 2)
                    try:
                        bits = int(key.split('_')[1].replace('bit', ''))
                        sensitivity = float(row[key])
                        layer_data[layer_name][bits] = sensitivity
                    except (ValueError, IndexError):
                        continue
    
    return layer_data


def estimate_register_usage(model, config, register_width=16, input_size=227):
    """
    Estimate total register usage for the model (Weights + Activations).
    Assumes B-bit values are packed into W-bit registers.
    
    Args:
        model: PyTorch model
        config: Dict mapping layer names to bit-widths
        register_width: HW register width (default: 16)
        input_size: Size of input images (default: 227)
        
    Returns:
        total_registers: Total count of registers required
    """
    total_registers = 0
    device = next(model.parameters()).device
    
    # Track activation sizes using a dummy forward pass
    activation_sizes = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # output is (batch, channels, H, W) or (batch, features)
            # We estimate for batch_size=1
            activation_sizes[name] = output[0].numel()
        return hook
        
    for name, module in model.named_modules():
        if name in config:
            hooks.append(module.register_forward_hook(hook_fn(name)))
            
    # Dummy forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, input_size, input_size).to(device)
        try:
            model(dummy_input)
        except Exception:
            # Fallback for models that need different input channels (e.g. RGB)
            dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
            try:
                model(dummy_input)
            except Exception:
                pass
            
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    # Calculate registers
    for name, module in model.named_modules():
        if name in config:
            bits = config[name]
            vals_per_reg = register_width // bits
            
            # 1. Weight Registers
            num_weights = module.weight.numel()
            reg_weights = math.ceil(num_weights / vals_per_reg)
            
            # 2. Activation Registers (at the output of this layer)
            num_acts = activation_sizes.get(name, 0)
            reg_acts = math.ceil(num_acts / vals_per_reg)
            
            total_registers += (reg_weights + reg_acts)
            
    return total_registers


def estimate_accuracy_impact(layer_sensitivities, current_config, layer_name, new_bits, baseline_bits=32):
    """
    Estimate accuracy impact using the W+A sensitivity profile.
    """
    if layer_name not in layer_sensitivities:
        return 5.0 # Conservative fallback
        
    # Profile loader stores bits as integers
    if new_bits in layer_sensitivities[layer_name]:
        return layer_sensitivities[layer_name][new_bits]
    
    return 10.0 # Conservative


def greedy_search(model_name, checkpoint_path, dataset, sensitivity_profile, 
                  bit_choices, target_drop=3.0, baseline_acc=None, 
                  register_aware=False, register_width=16):
    """
    Greedy search for optimal mixed-precision configuration.
    Architecture-agnostic: Uses module type inspection for layer classification.
    
    Strategy:
    1. Start with all layers at highest bit-width (safest)
    2. Iteratively reduce bit-widths of least sensitive layers
    3. Filter out 2-bit for Conv layers (they can't handle it)
    4. Stop when target accuracy drop is reached
    
    Optional Register-Aware Mode:
    - Calculates register usage (weights + activations assumed same bits)
    - Prioritizes reductions that save the most registers for the least accuracy drop.
    
    Args:
        model_name: Model architecture name
        checkpoint_path: Path to checkpoint
        dataset: Dataset name
        sensitivity_profile: {layer_name: {bits: sensitivity}}
        bit_choices: User-specified bit-widths (e.g., [2, 4, 8])
        target_drop: Maximum acceptable accuracy drop (default: 3.0%)
        baseline_acc: Baseline FP32 accuracy (if None, will measure)
        register_aware: Whether to optimize for register usage (default: False)
        register_width: HW register width in bits (default: 16)
    
    Returns:
        config: Dict mapping layer names to bit-widths
        final_acc: Estimated final accuracy
    """
    print(f"\n{'='*60}")
    if register_aware:
        print(f"REGISTER-AWARE GREEDY SEARCH ({register_width}-bit Registers)")
    else:
        print(f"HARDWARE-AWARE GREEDY SEARCH")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"User bit-widths: {bit_choices}")
    print(f"Target accuracy drop: {target_drop}%")
    if register_aware:
        print(f"Optimization: Weighted Register Savings / Accuracy Drop")
    print(f"{'='*60}\n")
    
    # Sort bit choices (high to low for initialization)
    sorted_bits = sorted(bit_choices, reverse=True)
    highest_bits = sorted_bits[0]
    
    # Initialize: All layers start at highest bit-width
    config = {}
    
    # We allow $S > 16$ because Hierarchical-MAC splits the work
    # So we simply initialize all layers at the highest bit choice
    for layer_name in sensitivity_profile.keys():
        config[layer_name] = highest_bits
    
    print(f"Initialized all layers at {highest_bits}-bit (Hierarchical-MAC enabled).")
    
    # Load model for layer type checking and baseline measurement
    print(f"\nLoading model for analysis...")
    num_classes = 100 if dataset == 'cifar100' else 43 if dataset == 'gtsrb' else 10
    if dataset == 'fashionmnist': num_classes = 10
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    
    # Determine input size early for register estimation
    if model_name in ['vgg11_bn', 'resnet']:
        input_size = 32  # CIFAR-style models (including for FashionMNIST)
    elif dataset == 'gtsrb':
        input_size = 224  # GTSRB uses 224x224 for all models
    elif model_name == 'alexnet' or dataset == 'fashionmnist':
        input_size = 227  # AlexNet/Vanilla FashionMNIST
    else:
        input_size = 224  # levit, swin, etc.

    # Measure baseline if not provided
    if baseline_acc is None:
        print(f"Measuring baseline accuracy...")

        # Set batch size based on model type to avoid GPU OOM
        if model_name == 'swin':
            batch_size = 16
        elif model_name == 'levit':
            batch_size = 32
        else:
            batch_size = 128

        if dataset == 'cifar100':
            loader = get_cifar100_dataloader(train=False, input_size=input_size, batch_size=batch_size)
        elif dataset == 'gtsrb':
            loader = get_gtsrb_dataloader(train=False, input_size=input_size, batch_size=batch_size)
        elif dataset == 'fashionmnist':
            loader = get_fashionmnist_dataloader(train=False, input_size=input_size, batch_size=batch_size)
        else:
            loader = get_cifar10_dataloader(train=False, input_size=input_size, batch_size=batch_size)
        
        baseline_acc = evaluate_accuracy(model, loader, max_samples=1000)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    target_acc = baseline_acc - target_drop
    print(f"Target accuracy: {target_acc:.2f}% (max drop: {target_drop}%)\n")
    
    initial_registers = estimate_register_usage(model, config, register_width, input_size)
    if register_aware:
        print(f"Initial register usage: {initial_registers:,} registers (Weights + Activations)")
    
    # Create list of (layer, current_bits, possible_reductions)
    # Sort by sensitivity (least sensitive first for aggressive reduction)
    layer_priorities = []
    
    print("Analyzing layer types...")
    conv_count = 0
    linear_count = 0
    
    for layer_name in config.keys():
        current_bits = config[layer_name]
        
        # Find all lower bit-widths we can try
        possible_reductions = [b for b in sorted_bits if b < current_bits]
        
        # ARCHITECTURE-AGNOSTIC: Filter 2-bit for Conv layers
        if is_conv_layer(model, layer_name):
            if 2 in possible_reductions:
                possible_reductions = [b for b in possible_reductions if b >= 4]
                # print(f"  Conv layer {layer_name}: excluding 2-bit")
            conv_count += 1
        else:
            linear_count += 1
        
        if not possible_reductions:
            continue  # Already at lowest bit-width
        
        # Get average sensitivity across available bit-widths
        sensitivities = []
        for bits in possible_reductions:
            if bits in sensitivity_profile[layer_name]:
                sensitivities.append(sensitivity_profile[layer_name][bits])
        
        if sensitivities:
            avg_sensitivity = sum(sensitivities) / len(sensitivities)
        else:
            avg_sensitivity = 0.5  # Default if no data
            
        # Register savings estimation for sorting
        if register_aware:
            # Get combined footprint for prioritization
            # We estimate the reduction in registers if we drop this layer's bits
            # For simplicity in sorting, we use (Weights + Activations) total count
            layer_config = {layer_name: current_bits}
            layer_regs = estimate_register_usage(model, layer_config, register_width, input_size)
            layer_priorities.append((layer_name, avg_sensitivity, possible_reductions, layer_regs))
        else:
            layer_priorities.append((layer_name, avg_sensitivity, possible_reductions))
    
    print(f"Detected: {conv_count} Conv layers (min 4-bit), {linear_count} Linear layers (can use 2-bit)\n")
    
    # Sorting logic
    if register_aware:
        # Sort by: sensitivity / num_params (Lower is better: low drop, high weight count)
        layer_priorities.sort(key=lambda x: x[1] / max(1, x[3]))
    else:
        # Sort by sensitivity (ascending - least sensitive first)
        layer_priorities.sort(key=lambda x: x[1])
    
    print(f"Layer prioritization (least → most sensitive):")
    for i, p in enumerate(layer_priorities[:5]):
        layer, sens = p[0], p[1]
        print(f"  {i+1}. {layer}: sensitivity={sens:.4f}")
    if len(layer_priorities) > 5:
        print(f"  ... ({len(layer_priorities)} total layers)\n")
    
    # Greedy reduction phase
    cumulative_drop = 0.0
    moves = 0
    
    print("Starting greedy bit-width reduction...\n")
    
    for layer_info in layer_priorities:
        layer_name = layer_info[0]
        avg_sensitivity = layer_info[1]
        possible_reductions = layer_info[2]
        
        current_bits = config[layer_name]
        
        # Try each lower bit-width (from highest to lowest)
        for new_bits in sorted(possible_reductions, reverse=True):
            # CARRY-SAFE FILTER: Only try if it fits in 16-bit register
            if not is_accumulator_safe(model, layer_name, new_bits, register_width):
                continue

            # Estimate accuracy impact
            estimated_drop = estimate_accuracy_impact(
                sensitivity_profile, config, layer_name, new_bits
            )
            
            # Check if we can afford this reduction
            if cumulative_drop + estimated_drop <= target_drop:
                # Accept the reduction
                old_bits = config[layer_name]
                config[layer_name] = new_bits
                cumulative_drop += estimated_drop
                moves += 1
                
                if moves % 5 == 0 or moves <= 10:  # Show first 10 moves + every 5th
                    print(f"  Move {moves}: {layer_name} "
                          f"{old_bits}→{new_bits}bit, "
                          f"est_drop={estimated_drop:.2f}%, "
                          f"cumulative={cumulative_drop:.2f}%")
                
                break  # Move to next layer after successful reduction
            else:
                # Can't reduce further without exceeding target
                if moves == 0:  # Debug info for first layer only
                    print(f"  [DEBUG] Cannot reduce {layer_name}: "
                          f"estimated_drop={estimated_drop:.2f}% would exceed "
                          f"budget (cumulative={cumulative_drop:.2f}%, target={target_drop:.2f}%)")
                continue
        
        # Stop if we're at target
        if cumulative_drop >= target_drop * 0.95:  # Within 95% of target
            print(f"\n  Reached target drop ({cumulative_drop:.2f}% ≈ {target_drop}%)")
            break
    
    # Calculate final statistics
    estimated_final_acc = baseline_acc - cumulative_drop
    
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Moves made: {moves}")
    print(f"Estimated accuracy: {estimated_final_acc:.2f}%")
    print(f"Estimated drop: {cumulative_drop:.2f}%")
    
    # Bit-width distribution
    bit_distribution = {}
    for bits in bit_choices:
        count = sum(1 for b in config.values() if b == bits)
        bit_distribution[bits] = count
    
    print(f"\nBit-width distribution:")
    for bits in sorted(bit_distribution.keys(), reverse=True):
        count = bit_distribution[bits]
        percentage = (count / len(config)) * 100
        print(f"  {bits}-bit: {count:3d} layers ({percentage:5.1f}%)")
    
    # Calculate compression ratio
    total_bits_original = len(config) * 32  # FP32 baseline
    total_bits_quantized = sum(config.values())
    compression_ratio = total_bits_original / total_bits_quantized
    
    print(f"\nEstimated compression: {compression_ratio:.2f}x")
    print(f"{'='*60}\n")
    
    return config, estimated_final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hardware-Aware Mixed-Precision Search')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--profile', type=str, required=True, help='Sensitivity profile CSV')
    parser.add_argument('--output', type=str, required=True, help='Output config JSON')
    parser.add_argument('--bits', type=int, nargs='+', required=True, 
                        help='User-specified bit-widths (e.g., --bits 2 4 8)')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='Dataset (cifar10/cifar100/gtsrb)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                        help='Target accuracy drop percentage (default: 3.0)')
    parser.add_argument('--baseline-acc', type=float, default=None,
                        help='Baseline accuracy (if known, to skip measurement)')
    parser.add_argument('--register-aware', action='store_true', help='Optimize for register usage')
    parser.add_argument('--register-width', type=int, default=16, help='HW Register width in bits')
    
    args = parser.parse_args()
    
    # Load sensitivity profile
    print(f"Loading sensitivity profile: {args.profile}")
    sensitivity_profile = load_sensitivity_profile(args.profile)
    print(f"Loaded {len(sensitivity_profile)} layers\n")
    
    # Run greedy search
    config, final_acc = greedy_search(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        sensitivity_profile=sensitivity_profile,
        bit_choices=args.bits,
        target_drop=args.target_drop,
        baseline_acc=args.baseline_acc,
        register_aware=args.register_aware,
        register_width=args.register_width
    )
    
    # Save configuration
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {args.output}")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Validate PTQ accuracy:")
    print(f"   python quantization_framework/experiments/validate_config.py \\")
    print(f"     --model {args.model} \\")
    print(f"     --checkpoint {args.checkpoint} \\")
    print(f"     --config {args.output} \\")
    print(f"     --dataset {args.dataset}")
    print(f"\n2. If PTQ fails, run QAT:")
    print(f"   python quantization_framework/experiments/qat_training.py \\")
    print(f"     --model {args.model} \\")
    print(f"     --checkpoint {args.checkpoint} \\")
    print(f"     --config {args.output} \\")
    print(f"     --dataset {args.dataset}")
    print(f"{'='*60}\n")
