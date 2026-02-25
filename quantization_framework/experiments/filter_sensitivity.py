import os
import sys
import torch
import torch.nn as nn
import csv
import argparse
from tqdm import tqdm

# Add quantization_framework to path
sys.path.append(os.path.join(os.getcwd(), 'quantization_framework'))

from models.model_loaders import load_model
from evaluation.pipeline import get_fashionmnist_dataloader, evaluate_accuracy

def quantize_filter(weight_tensor, channel_idx, bits):
    """
    Quantize only a specific filter (output channel) within a weight tensor.
    """
    with torch.no_grad():
        q_weight = weight_tensor.clone()
        filter_w = q_weight[channel_idx]
        
        q_max = (1 << (bits - 1)) - 1
        w_max = torch.max(torch.abs(filter_w))
        scale = w_max / q_max if w_max > 0 else 1.0
        
        # Quantize and dequantize to simulate precision loss
        filter_q = torch.round(filter_w / scale).clamp(-q_max-1, q_max)
        filter_dq = filter_q * scale
        
        q_weight[channel_idx] = filter_dq
    return q_weight

def main():
    parser = argparse.ArgumentParser(description='Granular Filter Sensitivity Analysis')
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--checkpoint', type=str, default='models/qalex-0-7.pth')
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--output', type=str, default='alexnet_fast_granular_sensitivity.csv')
    parser.add_argument('--layers', nargs='+', default=['conv1.0', 'conv2.0', 'conv3.0', 'conv4.0', 'conv5.0', 'fc1', 'fc2', 'fc3'])
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[GRANULAR ANALYSIS] Measuring Filter/Neuron Sensitivity for {args.model}")
    print(f"Target Layers: {args.layers}")
    
    # Load base model
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=10)
    model.to(device)
    model.eval()

    # Load data
    loader = get_fashionmnist_dataloader(train=False, input_size=227, batch_size=128)
    
    print("Measuring Baseline Accuracy...")
    baseline_acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
    print(f"Baseline: {baseline_acc:.2f}%\n")

    results = []
    
    # Analyze each layer
    for layer_name in args.layers:
        print(f"Analyzing {layer_name}...")
        
        # Find the module
        module = None
        for name, m in model.named_modules():
            if name == layer_name:
                module = m
                break
        
        if module is None or not isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"  Skipping {layer_name} (Not a Conv/Linear module)")
            continue

        # No limit - process every filter/neuron in the layer
        if isinstance(module, nn.Conv2d):
            num_filters = module.out_channels
        else: # Linear
            num_filters = module.out_features
            
        original_weights = module.weight.data.clone()
        
        # Test 2-bit and 4-bit for each filter
        for bits in [2, 4]:
            print(f"  Testing {bits}-bit quantization per neuron ({num_filters} total)...")
            # POC Aggression: Stride 1 for conv1 (small/critical), Stride 16 for others
            stride = 1 if layer_name == 'conv1.0' else 16 
            
            for i in tqdm(range(0, num_filters, stride)):
                # Apply quantization to specific filter/neuron
                module.weight.data = quantize_filter(original_weights, i, bits)
                
                # Measure drop
                acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
                drop = baseline_acc - acc
                
                results.append({
                    'layer': layer_name,
                    'filter_idx': i,
                    'bits': bits,
                    'accuracy': acc,
                    'drop': drop
                })
                
                # Restore weights
                module.weight.data = original_weights.clone()

                # Save result incrementally
                with open(args.output, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['layer', 'filter_idx', 'bits', 'accuracy', 'drop'])
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(results[-1])

    print(f"\nFilter sensitivity data saved to {args.output}")

if __name__ == "__main__":
    main()
