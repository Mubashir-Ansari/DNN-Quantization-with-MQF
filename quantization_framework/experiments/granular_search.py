import os
import sys
import json
import csv
import argparse
import torch
import torch.nn as nn
import numpy as np

def load_filter_sensitivity(csv_path):
    """
    Load per-filter sensitivity data.
    Returns: {layer_name: {filter_idx: {bits: drop}}}
    """
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            idx = int(row['filter_idx'])
            bits = int(row['bits'])
            drop = float(row['drop'])
            
            if layer not in data:
                data[layer] = {}
            if idx not in data[layer]:
                data[layer][idx] = {}
            data[layer][idx][bits] = drop
    return data

def main():
    parser = argparse.ArgumentParser(description='Granular MQF Search')
    parser.add_argument('--sensitivity', type=str, default='alexnet_filter_sensitivity.csv')
    parser.add_argument('--output', type=str, default='alexnet_granular_config.json')
    parser.add_argument('--target-drop', type=float, default=5.0)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sensitivity):
        print(f"Error: {args.sensitivity} not found.")
        return

    print(f"\n[GRANULAR SEARCH] Optimizing bits per neuron (Target Drop: {args.target_drop}%)")
    sensitivity_data = load_filter_sensitivity(args.sensitivity)
    
    config = {}
    
    # We apply a thresholding strategy
    # If 2-bit drop is very small (< 0.1% for this neuron), use 2-bit
    # Else if 4-bit drop is small (< 0.1%), use 4-bit
    # Otherwise use 8-bit
    
    # This is a heuristic that works well for per-filter mixed precision
    # In a full researcher POC, we could use a global greedy search across ALL filters,
    # but per-layer thresholding is more hardware-intuitive.

    # Load model to get actual filter counts
    sys.path.append(os.path.join(os.getcwd(), 'quantization_framework'))
    from models.model_loaders import load_model
    model = load_model('alexnet', checkpoint_path='models/qalex-0-7.pth', num_classes=10)

    for layer_name, analyzed_filters in sensitivity_data.items():
        print(f"Processing {layer_name}...")
        
        # Get actual channel count from model
        target_module = dict(model.named_modules()).get(layer_name)
        is_linear = isinstance(target_module, nn.Linear)
        
        if target_module and hasattr(target_module, 'out_channels'):
            real_total = target_module.out_channels
        elif target_module and hasattr(target_module, 'out_features'):
            real_total = target_module.out_features
        else:
            real_total = max(analyzed_filters.keys()) + 1
            
        layer_bits = [8] * real_total # Default everything to 8-bit
        
        # Determine if we used a stride (for interpolation)
        analyzed_indices = sorted(analyzed_filters.keys())
        stride = analyzed_indices[1] - analyzed_indices[0] if len(analyzed_indices) > 1 else 1
        
        # Apply findings for analyzed filters AND their neighbor blocks
        for idx in analyzed_indices:
            bit_drops = analyzed_filters[idx]
            drop_2 = bit_drops.get(2, 99.0)
            drop_4 = bit_drops.get(4, 99.0)
            
            # RESEARCH STRATEGY: "Prudent Surgical" (Winning Map)
            # 1. FC layers must be 8-bit for cumulative noise stability (Backbone)
            # 2. Conv layers use surgical 4-bit (if drop < 0.05) or 8-bit.
            
            if is_linear:
                best_b = 8
            else:
                if drop_4 < 0.05:
                    best_b = 4
                else:
                    best_b = 8
                
            # Assign to the block [idx, idx + stride)
            for j in range(idx, min(idx + stride, real_total)):
                layer_bits[j] = best_b
        
        config[layer_name] = layer_bits
        
        # Stats for the layer
        b2 = layer_bits.count(2)
        b4 = layer_bits.count(4)
        b8 = layer_bits.count(8)
        print(f"  Result: 2-bit: {b2}, 4-bit: {b4}, 8-bit: {b8} (Total: {real_total})")

    # Add default layer-wise precision for layers not in granular analysis
    # (e.g., fc layers or later convs if we skipped them)
    # We'll use 8-bit for safety
    other_layers = ["conv3.0", "conv4.0", "conv5.0", "fc1", "fc2", "fc3"]
    for l in other_layers:
        if l not in config:
            config[l] = 8

    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nGranular config saved to {args.output}")

if __name__ == "__main__":
    main()
