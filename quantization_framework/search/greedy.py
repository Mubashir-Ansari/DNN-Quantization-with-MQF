import torch
import torch.nn as nn
import copy
import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
import pandas as pd

def load_sensitivity_profile(csv_path):
    df = pd.read_csv(csv_path)
    
    # Detect format
    if 'granule_index' in df.columns:
        is_granular = True
        # Granular: layer,type,granule_index,bit_width,accuracy,drop
        profile = {}
        for _, row in df.iterrows():
            layer = row['layer']
            idx = int(row['granule_index'])
            bits = int(row['bit_width'])
            acc = float(row['accuracy'])
            
            if layer not in profile: profile[layer] = {}
            if idx not in profile[layer]: profile[layer][idx] = {}
            profile[layer][idx][bits] = acc
            
    else:
        is_granular = False
        # Layer-wise: layer_name,layer_type,bit_width,accuracy,accuracy_drop
        profile = {}
        target_col = 'layer_name' if 'layer_name' in df.columns else 'layer'
        
        for _, row in df.iterrows():
            layer = row[target_col]
            bits = int(row['bit_width'])
            acc = float(row['accuracy'])
            
            if layer not in profile: profile[layer] = {}
            profile[layer][bits] = acc
            
    return profile, is_granular

def get_layer_size_mb(model, layer_name):
    """Calculate size of a layer's weights in MB (assuming FP32 baseline size)."""
    for name, module in model.named_modules():
        if name == layer_name:
            if hasattr(module, 'weight'):
                # num_params * 4 bytes / 1024^2
                return module.weight.numel() * 4 / (1024 * 1024)
    return 0
    
def get_granule_count(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
             if hasattr(module, 'weight'):
                # Conv: [Out, In, k, k], Linear: [Out, In]
                # Granules = Dim 0
                return module.weight.shape[0]
    return 1

def greedy_search_constrained(model, sensitivity_profile, is_granular, target_acc_drop=1.0, constraints=None):
    """
    Greedy Accuracy-Aware Search (Supports Granular)
    Args:
        constraints: dict of {regex_pattern: min_bits}
    """
    
    # 1. State Initialization
    packet = {} 
    layer_names = list(sensitivity_profile.keys())
    
    baseline_acc = 0
    
    # Find available bit-widths in the profile
    available_bits = set()
    if is_granular:
        for layer in layer_names:
            for idx in sensitivity_profile[layer]:
                available_bits.update(sensitivity_profile[layer][idx].keys())
    else:
        for layer in layer_names:
            available_bits.update(sensitivity_profile[layer].keys())
    
    max_bit = max(available_bits) if available_bits else 8
    steps = sorted(available_bits, reverse=True)  # e.g., [4, 2] or [8, 6, 4, 2]
    print(f"Available bit-widths in profile: {steps}")
    
    # Initialize all to max available bit-width (not necessarily 8)
    if is_granular:
        for layer in layer_names:
            indices = list(sensitivity_profile[layer].keys())
            packet[layer] = {idx: max_bit for idx in indices} 
            
            # Baseline = accuracy when drop is near 0 at highest bit-width
            # Or equivalently: accuracy + drop = baseline
            for idx in indices:
                if max_bit in sensitivity_profile[layer][idx]:
                    # The stored accuracy IS the model accuracy when that granule is quantized
                    # Baseline ≈ max(all accuracies at highest bit) since drop should be ~0
                    baseline_acc = max(baseline_acc, sensitivity_profile[layer][idx][max_bit])
    else:
        for l in layer_names:
            packet[l] = max_bit
            if max_bit in sensitivity_profile[l]:
                 baseline_acc = max(baseline_acc, sensitivity_profile[l][max_bit])
    
    print(f"Baseline Accuracy ({max_bit}-bit): {baseline_acc:.2f}%")
    print(f"Target Accuracy: {baseline_acc - target_acc_drop:.2f}% (Drop tolerance: {target_acc_drop}%)")
    
    current_acc = baseline_acc 
    
    # Pre-compute layer sizes (OPTIMIZATION: avoid repeated model lookups)
    layer_sizes = {}
    granule_counts = {}
    for layer in layer_names:
        layer_sizes[layer] = get_layer_size_mb(model, layer)
        granule_counts[layer] = get_granule_count(model, layer)
    
    # Optimization Loop
    move_count = 0
    while True:
        best_move = None
        best_metric = float('inf') 
        
        # Flattened search space: Iterate over all (Layer, Unit)
        # If Layer-wise, Unit is just "Layer"
        # If Granular, Unit is "Layer:Index"
        
        candidates = []
        if is_granular:
             for layer in layer_names:
                for idx in packet[layer]:
                    candidates.append((layer, idx))
        else:
             candidates = [(l, None) for l in layer_names]
             
        for layer, granule_idx in candidates:
            # Determine current/next bits
            if is_granular:
                current_bits = packet[layer][granule_idx]
            else:
                current_bits = packet[layer]
            
            if current_bits == min(steps): continue  # Already at min bit-width
            
            try:
                next_bits = steps[steps.index(current_bits) + 1]
            except (IndexError, ValueError): continue

            # CHECK CONSTRAINTS
            if constraints:
                min_b = 2
                for pattern, mb in constraints.items():
                    if pattern in layer:
                        min_b = max(min_b, mb)
                if next_bits < min_b: continue
            
            # Check data availability
            if is_granular:
                if next_bits not in sensitivity_profile[layer][granule_idx]: continue
                acc_current = sensitivity_profile[layer][granule_idx][current_bits]
                acc_next = sensitivity_profile[layer][granule_idx][next_bits]
            else:
                if next_bits not in sensitivity_profile[layer]: continue
                acc_current = sensitivity_profile[layer][current_bits]
                acc_next = sensitivity_profile[layer][next_bits]
                
            # 1. Size Saving (using cached values)
            total_layer_size = layer_sizes[layer]
            if is_granular:
                num_g = granule_counts[layer]
                granule_size = total_layer_size / max(1, num_g)
                size_reduc = (granule_size / 32) * (current_bits - next_bits)
            else:
                size_reduc = (total_layer_size / 32) * (current_bits - next_bits)
                
            if size_reduc <= 0: continue
            
            # 2. Acc Drop
            delta_drop = acc_current - acc_next
            
            metric = delta_drop / size_reduc
            
            if metric < best_metric:
                best_metric = metric
                best_move = (layer, granule_idx, next_bits, delta_drop, size_reduc)
        
        if best_move is None:
            print(f"\\nNo valid moves left after {move_count} moves.")
            break
            
        layer, g_idx, bits, drop, size_saved = best_move
        
        if (current_acc - drop) < (baseline_acc - target_acc_drop):
            print(f"Stopping: Next best move drops acc to {current_acc - drop:.2f}%, violating target.")
            break
            
        # Apply move
        if is_granular:
            packet[layer][g_idx] = bits
            display_name = f"{layer}[:{g_idx}]"
        else:
            packet[layer] = bits
            display_name = layer
            
        current_acc -= drop
        move_count += 1
        
        # Progress reporting
        if move_count % 500 == 0:
            print(f"  [{move_count} moves] Est Acc: {current_acc:.2f}%")
        
    # Final cleanup: If granular, convert dict {idx: bits} to list [bits, bits...]
    if is_granular:
        final_config = {}
        for layer in packet:
            # We need to sort indices to form a list
            sorted_indices = sorted(packet[layer].keys())
            # Warning: If profile didn't cover all indices (e.g. strided), we have gaps.
            # We should assume 8-bit for missing ones.
            # Get true count
            num_g = get_granule_count(model, layer)
            bit_list = [8] * num_g
            for idx in sorted_indices:
                if idx < num_g:
                    bit_list[idx] = packet[layer][idx]
            final_config[layer] = bit_list
        return final_config
    else:
        return packet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--profile', type=str, required=True, help='Path to sensitivity csv')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--target_drop', type=float, default=1.0, help='Max allowed accuracy drop (%)')
    parser.add_argument('--output', type=str, default='optimized_config.json')
    args = parser.parse_args()
    
    # Load Model
    print(f"Loading {args.model}...")
    num_classes = 100 if 'swin' in args.model.lower() or 'cifar100' in (args.checkpoint or '') else 10
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    
    # Load Profile
    print(f"Loading sensitivity profile: {args.profile}")
    profile, is_granular = load_sensitivity_profile(args.profile)
    print(f"Mode: {'Granular' if is_granular else 'Layer-wise'}")
    
    # Run Search
    print("Starting Greedy Search...")
    config = greedy_search(model, profile, is_granular, target_acc_drop=args.target_drop)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Optimized configuration saved to {args.output}")
