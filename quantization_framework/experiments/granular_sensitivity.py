import argparse
import pandas as pd
import torch
import torch.nn as nn
import copy
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor
from evaluation.pipeline import evaluate_accuracy, get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader

def measure_granular_sensitivity(model, dataloader, bit_widths=[2, 4, 8], device='cuda', samples=128):
    """
    Measures sensitivity of individual Output Channels (Conv) or Heads (Attention).
    Strategy:
    1. Identify all quantizable layers.
    2. For Conv2d: Iterate over each output channel.
       - Quantize ONLY that channel to specified bit-widths.
       - Measure Acc Drop.
    3. For Transformers: This is trickier. We need to identify Heads.
       - We will implement 'Per-Channel' for Linear layers as a proxy for now.
       - In PyTorch, Linear weight is (Out, In). So channel_dim=0 corresponds to output neurons.
       - For QKV layers, Output Neurons = All Heads concatenated.
       - So measuring per-output-neuron sensitivity gives us sub-head granularity.
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        bit_widths: List of bit-widths to test (e.g., [2, 4, 8])
        device: Device to run on
        samples: Number of samples for evaluation
    """
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    # 1. Baseline Accuracy (FP32 or INT8 baseline?)
    # Let's assume baseline is current state (FP32)
    print("Measuring baseline accuracy...")
    baseline_acc = evaluate_accuracy(model, dataloader, device=device, max_samples=samples)
    print(f"Baseline Acc: {baseline_acc:.2f}%")
    
    results = []
    
    # helper for clean names
    def get_layer_type(module):
        if isinstance(module, nn.Conv2d): return "Conv2d"
        if isinstance(module, nn.Linear): return "Linear"
        return "Unknown"

    print(f"Testing bit-widths: {bit_widths}")
    print("Starting Granular Sensitivity Analysis...")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # We focus on weights
            if not hasattr(module, 'weight'): continue
            
            # Determine Granularity
            # Conv2d: weight (Out, In, k, k) -> Dim 0 is Out Channels (Filters)
            # Linear: weight (Out, In) -> Dim 0 is Out Neurons
            num_granules = module.weight.shape[0]
            
            # Sub-sampling granules if too many?
            # VGG features.0 has 64 filters. Easy.
            # VGG features.40 fc has 4096 neurons. Hard.
            # We might need to stride if too large.
            step = 1
            if num_granules > 512: 
                step = num_granules // 128 # Cap at ~128 probes per layer to save time
            
            print(f"Scanning {name} ({num_granules} granules)...")
            
            # Backup weight
            original_weight = module.weight.data.clone()
            
            for i in range(0, num_granules, step):
                # We analyze granule 'i'
                
                for bits in bit_widths:
                    # 1. Quantize ONLY granule i
                    # How? We can use the scale vector!
                    # Make a scale vector of 1s (no quant) 
                    # Set scale for i to quant scale.
                    # Actually easier: Quantize the whole tensor with per-channel logic
                    # BUT that quantizes ALL channels.
                    # We want to measure sensitivity of ONE channel.
                    # So: Quantize channel i to K bits. Keep others FP32 (effectively).
                    
                    # Create a mixed-precision weight for probing
                    # This is complex to implement efficiently.
                    # Simplification: Quantize STRIP i to 'bits', keep rest original.
                    
                    # Get Channel i
                    w_i = original_weight[i].unsqueeze(0) # (1, In...)
                    
                    # Quantize it
                    q_w_i, _, _ = quantize_tensor(w_i, bit_width=bits, method='symmetric')
                    
                    # Construct probe weight
                    probe_weight = original_weight.clone()
                    probe_weight[i] = q_w_i.squeeze(0)
                    
                    # Apply
                    module.weight.data = probe_weight
                    
                    # Eval
                    acc = evaluate_accuracy(model, dataloader, device=device, max_samples=samples)
                    drop = baseline_acc - acc
                    
                    results.append({
                        'layer': name,
                        'type': get_layer_type(module),
                        'granule_index': i,
                        'bit_width': bits,
                        'accuracy': acc,
                        'drop': drop
                    })
                    
                    # Restore for next bit-width
                    
                # Restore original for next granule
                module.weight.data = original_weight
                
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--output', type=str, default='granular_sensitivity.csv')
    parser.add_argument('--samples', type=int, default=128) # Fast probe
    parser.add_argument('--input-size', type=int, default=None)
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 8],
                        help='Bit-widths to test (e.g., --bits 2 4 8)')
    
    args = parser.parse_args()
    
    # Constants
    if args.dataset == 'cifar100': num_classes = 100
    elif args.dataset == 'gtsrb': num_classes = 43
    else: num_classes = 10
    
    # Load Model
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    
    # Load Data
    input_size = args.input_size if args.input_size else (32 if args.model not in ['levit', 'swin'] else 224)
    if args.dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=128, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(batch_size=128, train=False, input_size=input_size)
    else:
        loader = get_cifar10_dataloader(batch_size=128, train=False, input_size=input_size)
        
    print(f"Running Granular Sensitivity for {args.model} on {args.dataset}...")
    df = measure_granular_sensitivity(model, loader, bit_widths=args.bits, samples=args.samples)
    
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
