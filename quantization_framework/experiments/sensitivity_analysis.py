import torch
import torch.nn as nn
import argparse
import pandas as pd
import copy
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.primitives import quantize_tensor
from models.model_loaders import load_model
from evaluation.pipeline import evaluate_accuracy, get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader

def get_quantizable_layers(model):
    """
    Identify Linear and Conv2d layers that can be quantized.
    Returns a list of (name, module) tuples.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers

def measure_sensitivity(model, dataloader, bit_widths=[2, 4, 6, 8], output_file='sensitivity_results.csv', device='cuda'):
    print(f"Starting sensitivity analysis on device: {device}")
    model.to(device)
    
    # 1. Baseline Accuracy
    print("Measuring baseline accuracy...")
    baseline_acc = evaluate_accuracy(model, dataloader, device=device)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    results = []
    layers = get_quantizable_layers(model)
    print(f"Found {len(layers)} quantizable layers.")
    
    # 2. Iterate layers
    for i, (name, layer) in enumerate(layers):
        print(f"[{i+1}/{len(layers)}] analyzing layer: {name}...")
        
        # Save original weights
        original_weight = layer.weight.data.clone()
        
        for bits in bit_widths:
            # Quantize weight
            # Note: For sensitivity analysis, we often just quantize weights. 
            # Activations are usually kept at 32-bit or fixed 8-bit.
            # Here we follow the plan: "quantize_only_layer(L, bit_width)"
            
            q_weight, scale, zero = quantize_tensor(original_weight, bit_width=bits, method='symmetric')
            
            # Dequantize immediately to simulate specific bit-width inference (Fake Quantization equivalent for analysis)
            # q_weight is integer-like (float), so we convert back to float scale
            # In primitives.py, quantize_tensor returns (x_quant, scale, zero)
            # x_quant is scaled integer values. To run in PyTorch FP32 engine, we dequantize:
            # x_dequant = (x_quant - zero) * scale (if asymmetric) roughly. 
            # Symmetric: x_dequant = x_quant * scale (if mean is 0).
            # Let's rely on standard dequant logic.
            
            # Reconstruct estimated FP values from quantized representation
            if zero.numel() > 1: # Per channel/tensor
                 # Simple scalar logic for now based on primitives.py
                 pass
            
            # primitives.py returns x_quant (integer values as float). 
            # We need to dequantize to run inference (Simulated Quantization / Fake Quant)
            if bits == 32:
                # Skip
                acc = baseline_acc
            else:
                # Fake Quantize: Dequantize back to float
                # Symmetric: x_out = x_quant
                # Primitives returns scaled x_quant. We don't have an explicit dequant function exposed there yet?
                # Actually, primitives returns x_quant which is "x_int * scale".
                # Wait, looking at primitives.py: 
                # x_quant = x_int * scale. 
                # So x_quant IS ALREADY the dequantized approximation (Fake Quantized value).
                # Excellent.
                
                layer.weight.data = q_weight.to(device)
                
                # Evaluate
                acc = evaluate_accuracy(model, dataloader, device=device)
                
            drop = baseline_acc - acc
            print(f"  Bits: {bits} | Acc: {acc:.2f}% | Drop: {drop:.2f}%")
            
            results.append({
                'layer_name': name,
                'layer_type': type(layer).__name__,
                'bit_width': bits,
                'accuracy': acc,
                'accuracy_drop': drop
            })
            
        # Restore original weights
        layer.weight.data = original_weight
        
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Sensitivity analysis complete. Results saved to {output_file}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='levit', choices=['levit', 'swin', 'vgg11_bn', 'vgg', 'resnet'], help='Model to analyze')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint file (optional)')
    parser.add_argument('--output', type=str, default='sensitivity_results.csv', help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'gtsrb'], help='Dataset (cifar10/cifar100/gtsrb)')
    parser.add_argument('--input-size', type=int, default=None, help='Input resolution (e.g. 32 or 224). If None, auto-detected.')
    
    args = parser.parse_args()
    
    # Load Model
    print(f"Loading {args.model}...")
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10
        
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    
    if model is None:
        print("Model load failed.")
        sys.exit(1)
        
    # Determine input size
    if args.input_size is not None:
        input_size = args.input_size
    elif args.model in ['levit', 'swin']: # Transformers usually need 224
        input_size = 224
    else:
        # ResNet and VGG on CIFAR are usually 32
        input_size = 32
        
    # Load Data
    print(f"Loading {args.dataset} Dataset with resolution {input_size}x{input_size}...")
    if args.dataset == 'cifar100':
        train_loader = get_cifar100_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        train_loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    else:
        train_loader = get_cifar10_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    
    # Run Analysis
    measure_sensitivity(model, train_loader, output_file=args.output, device=args.device)
