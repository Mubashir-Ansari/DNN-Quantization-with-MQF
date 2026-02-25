import os
import sys
import torch
import json
import argparse
import math
from tqdm import tqdm

# Add quantization_framework to path
sys.path.append(os.path.join(os.getcwd(), 'quantization_framework'))

from models.model_loaders import load_model, get_model_size_info
from evaluation.pipeline import get_fashionmnist_dataloader, get_cifar10_dataloader, evaluate_accuracy
from quantization.register_aware_executor import wrap_model_for_mqf


def run_strategic_simulation(config, args):
    print("\n" + "="*80)
    print(f"STRATEGIC REGISTER-MISMATCH SIMULATION")
    print("="*80)
    print(f"Register Width: {args.register_width}-bit")
    print(f"Target Model:   {args.model}")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load and Wrap Model
    print(f"\n[STEP 1] Loading and Wrapping Model...")
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=10)
    
    # Inject Research Wrappers
    model = wrap_model_for_mqf(model, config, register_width=args.register_width, joint=args.joint)
    
    # Trigger Audit for the first convolutional layer
    if "conv1.0" in config:
        try:
            conv1_wrapper = dict(model.named_modules())["conv1.0"]
            conv1_wrapper.do_audit = True
            print(f"[AUDIT] Math audit enabled for conv1.0 (Joint={args.joint})")
        except KeyError:
            pass
            
    model.to(device)
    model.eval()

    # 2. Evaluate
    print(f"\n[STEP 2] Measuring Accuracy (FashionMNIST)...")
    input_size = 227 if args.model == 'alexnet' else 32
    loader = get_fashionmnist_dataloader(train=False, input_size=input_size, batch_size=64)
    
    acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
    
    print(f"\n[RESULTS] Final Accuracy: {acc:.2f}%")
    
    # 3. Report Efficiency
    print(f"\n[ALGO REPORT: REGISTER-MISMATCH ANALYSIS]")
    print(f"{'Layer':<12} | {'2b %':<6} | {'4b %':<6} | {'8b %':<6} | {'Pack (A)':<8} | {'Status'}")
    print("-" * 80)
    
    total_regs = 0
    total_wasted_bits = 0
    
    for name, bits_info in config.items():
        try:
            wrapper = dict(model.named_modules())[name]
        except KeyError:
            continue
            
        actual_layer = wrapper.layer
        w = actual_layer.weight.data
        out_ch = w.shape[0]
        params_per_ch = w[0].numel()
        
        # Granular Stats Summation
        layer_regs = 0
        layer_wasted = 0
        bit_counts = {2: 0, 4: 0, 8: 0}
        
        # We calculate per output channel (neuron)
        for i in range(out_ch):
            bits = wrapper.bit_widths[i].item()
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
            pack_dens = wrapper.storage_density[i].item()
            
            # Regs for this neuron
            reg_count = math.ceil(params_per_ch / pack_dens)
            layer_regs += reg_count
            
            # Wasted bits
            wasted_bits_per_reg = args.register_width % bits
            layer_wasted += reg_count * wasted_bits_per_reg + (reg_count * pack_dens - params_per_ch) * bits
            
        total_regs += layer_regs
        total_wasted_bits += layer_wasted
        
        # Calculate percentages
        p2 = (bit_counts.get(2, 0) / out_ch) * 100
        p4 = (bit_counts.get(4, 0) / out_ch) * 100
        p8 = (bit_counts.get(8, 0) / out_ch) * 100
        
        avg_pack = torch.mean(wrapper.storage_density.float()).item()
        
        # Status for the layer
        status = "GRANULAR"
        print(f"{name:<12} | {p2:>5.1f}% | {p4:>5.1f}% | {p8:>5.1f}% | {avg_pack:<8.1f} | {status}")

    total_bits = total_regs * args.register_width
    efficiency = (1 - (total_wasted_bits / total_bits)) * 100 if total_bits > 0 else 0
    
    print("-" * 80)
    print(f"  Register Count (Total):        {total_regs:,}")
    print(f"  Storage Efficiency (Stage A):  {efficiency:.2f}%")
    print(f"  Quantization Mode:             {'JOINT W=A' if args.joint else 'WEIGHT-ONLY'}")
    print(f"  Algorithm: MQF Carry-Aware Packing (Accordion Flow).")
    print("="*80 + "\n")

    return acc, efficiency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MQF Strategic Research Hub')
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--checkpoint', type=str, default='models/qalex-0-7.pth')
    parser.add_argument('--register-width', type=int, default=16)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--config', type=str, help='Path to JSON config')
    parser.add_argument('--joint', action='store_true', default=True, help='Enable Joint Weight-Activation matching')
    
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "conv1.0": 4, "conv2.0": 2, "conv3.0": 2, "conv4.0": 2, "conv5.0": 2,
            "fc1": 2, "fc2": 2, "fc3": 2
        }

    run_strategic_simulation(config, args)
