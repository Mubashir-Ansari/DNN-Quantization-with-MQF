"""
Layer-Level Sensitivity Analysis (Matched Weight + Activation)
Measures accuracy drop when quantizing entire layers (W+A).
Outputs format compatible with hardware_aware_search.py
"""

import argparse
import csv
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model, get_model_size_info
from quantization.primitives import quantize_tensor
from evaluation.pipeline import evaluate_accuracy, get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader, get_fashionmnist_dataloader
from experiments.validate_config import insert_activation_quantizers, calibrate_activation_quantizers

def run_layer_sensitivity(model_name, checkpoint_path, dataset='cifar10', 
                          output_csv='layer_profile.csv', bit_widths=[2, 4, 8],
                          quantize_activations=True):
    """
    Measure layer-level sensitivity: quantize entire layer (W+A), measure accuracy drop.
    """
    print(f"\n{'='*60}")
    print(f"LAYER-LEVEL SENSITIVITY ANALYSIS (Matched W+A)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Quantize Activations: {quantize_activations}")
    print(f"Testing bit-widths: {bit_widths}")
    print(f"{'='*60}\n")
    
    # Load model
    if dataset == 'fashionmnist':
        num_classes = 10
    else:
        num_classes = 100 if dataset == 'cifar100' else 43 if dataset == 'gtsrb' else 10
    
    model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Determine input size early for register estimation
    if model_name in ['vgg11_bn', 'resnet']:
        input_size = 32
    elif dataset == 'gtsrb':
        input_size = 224
    elif model_name == 'alexnet' or dataset == 'fashionmnist':
        input_size = 227
    else:
        input_size = 224
    
    model.to(device)
    model.eval()
    
    # Load data
    batch_size = 128
    
    if dataset == 'fashionmnist':
        loader = get_fashionmnist_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        loader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    
    # Measure baseline
    print("Measuring baseline accuracy...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        baseline_acc = evaluate_accuracy(model, loader, device=device, max_samples=1000)
        print(f"Baseline: {baseline_acc:.2f}%\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to measure baseline accuracy: {e}")
        sys.exit(1)
    
    # Get quantizable layers
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layers.append(name)
    
    print(f"Analyzing {len(layers)} layers...\n")
    
    # Run analysis
    results = []
    
    for idx, layer_name in enumerate(layers):
        print(f"[{idx+1}/{len(layers)}] Testing {layer_name}...")
        layer_result = {'layer': layer_name, 'baseline_acc': baseline_acc}
        
        for bits in sorted(bit_widths):
            # Create fresh model copy for each bit-width to avoid interference
            test_model = load_model(model_name, checkpoint_path=checkpoint_path, num_classes=num_classes)
            test_model.to(device)
            test_model.eval()
            
            # 1. Quantize Weights
            for name, module in test_model.named_modules():
                if name == layer_name:
                    attr = '_data' if hasattr(module.weight, '_data') else 'data'
                    w = getattr(module.weight, attr)
                    q_w, _, _ = quantize_tensor(w, bit_width=bits)
                    setattr(module.weight, attr, q_w)
            
            # 2. Quantize Activations (if enabled)
            if quantize_activations:
                # Insert quantizer for THIS layer only
                test_model, quantizers = insert_activation_quantizers(
                    test_model, config={layer_name: bits}, quantize_activations=True
                )
                # Calibrate on a few batches
                calibrate_activation_quantizers(test_model, quantizers, loader, device=device, num_batches=5)
            
            # 3. Evaluate
            try:
                acc = evaluate_accuracy(test_model, loader, device=device, max_samples=1000)
                drop = baseline_acc - acc
                layer_result[f'accuracy_{bits}bit'] = round(acc, 2)
                layer_result[f'sensitivity_{bits}bit'] = round(drop, 2)
                
                status = "W+A" if quantize_activations else "W-only"
                print(f"  {bits}-bit ({status}): Acc={acc:.2f}%, Drop={drop:.2f}%")
            except Exception as e:
                print(f"  {bits}-bit: ERROR ({e})")
                layer_result[f'sensitivity_{bits}bit'] = 100.0
            
            del test_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results.append(layer_result)
        print()
    
    # Save to CSV
    if results:
        fieldnames = ['layer', 'baseline_acc']
        for bit_width in sorted(bit_widths):
            fieldnames.extend([f'accuracy_{bit_width}bit', f'sensitivity_{bit_width}bit'])
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"{'='*60}")
        print(f"✓ Sensitivity profile (W+A) saved to {output_csv}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Layer-Level Sensitivity Analysis')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8], help='Bit-widths to test')
    parser.add_argument('--no-activations', action='store_false', dest='activations', help='Disable activation quantization during profiling')
    
    args = parser.parse_args()
    
    run_layer_sensitivity(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        output_csv=args.output,
        bit_widths=args.bits,
        quantize_activations=args.activations
    )
