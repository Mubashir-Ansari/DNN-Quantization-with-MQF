import torch
import json
import os
import sys
import argparse
import csv
from tqdm import tqdm
import math

# Add framework to path
sys.path.append(os.path.join(os.getcwd(), 'quantization_framework'))

from models.model_loaders import load_model
from evaluation.pipeline import get_fashionmnist_dataloader, evaluate_accuracy
from quantization.register_aware_executor import wrap_model_for_mqf

def run_joint_surgical_search(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print(f"\n[STEP 1] Loading FashionMNIST...")
    # AlexNet expects 227x227
    loader = get_fashionmnist_dataloader(train=False, input_size=227, batch_size=128)
    
    # 2. Get Baseline (Joint 8-bit)
    print(f"[STEP 2] Measuring Baseline (8-bit Joint)...")
    layers_to_optimize = ["conv1.0", "conv2.0", "conv3.0", "conv4.0", "conv5.0", "fc1", "fc2", "fc3"]
    
    # Initial config: All 8-bit
    best_config = {layer: 8 for layer in layers_to_optimize}
    
    model = load_model(args.model, checkpoint_path=args.checkpoint)
    model = wrap_model_for_mqf(model, best_config, register_width=args.register_width, joint=True)
    model.to(device)
    baseline_acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    # 3. Surgical Search logic
    log_path = "full_model_surgical_hardware_proof.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Group', 'FilterRange', 'TargetBits', 'Acc', 'Drop', 'Status', 'RegsSaved'])
    
    print(f"\n[STEP 3] Starting Full Model Sequential Search...")
    print(f"Register Width: {args.register_width}, Max Global Drop: {args.max_drop}%")
    
    for layer_name in layers_to_optimize:
        print(f"\n>>> Optimizing Layer: {layer_name}")
        
        # Get number of filters/neurons in this layer
        temp_model = load_model(args.model, checkpoint_path=args.checkpoint)
        target_layer = dict(temp_model.named_modules())[layer_name]
        num_filters = target_layer.out_channels if hasattr(target_layer, 'out_channels') else target_layer.out_features
        del temp_model
        
    for layer_name in layers_to_optimize:
        print(f"\n>>> Optimizing Layer: {layer_name}")
        
        # Get number of filters/neurons in this layer
        temp_model = load_model(args.model, checkpoint_path=args.checkpoint)
        target_layer = dict(temp_model.named_modules())[layer_name]
        num_filters = target_layer.out_channels if hasattr(target_layer, 'out_channels') else target_layer.out_features
        del temp_model
        
        # Initialize bits for this layer to 8
        if layer_name not in best_config or not isinstance(best_config[layer_name], list):
            layer_bits = [8] * num_filters
        else:
            layer_bits = best_config[layer_name]
    
        # Strict Filter-by-Filter / Neuron-by-Neuron search
        for i in tqdm(range(num_filters), desc=f"Surgical Pulse: {layer_name}"):
            # Greedy Optimization: Try 2-bit first, then 4-bit
            for trial_bits in [2, 4]:
                temp_layer_bits = list(layer_bits)
                temp_layer_bits[i] = trial_bits
                
                # Test Model with updated bit for this specific filter/neuron
                test_config = dict(best_config)
                test_config[layer_name] = temp_layer_bits
                
                torch.cuda.empty_cache()
                model = load_model(args.model, checkpoint_path=args.checkpoint)
                # Wrap ensuring Joint Weight+Activation and Handoff Locks (verified in MQF engine)
                model = wrap_model_for_mqf(model, test_config, register_width=args.register_width, joint=True)
                model.to(device)
                model.eval()
                
                # Real-time Hardware-Fidelity Accuracy check
                acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
                total_drop = baseline_acc - acc
                
                if total_drop <= args.max_drop:
                    # ACCEPTED: Signal integrity maintained in this node
                    layer_bits[i] = trial_bits
                    best_config[layer_name] = layer_bits
                    
                    status = f"ACCEPTED ({trial_bits}b)"
                    regs_at_8 = 1 / (args.register_width / 8)
                    regs_at_trial = 1 / (args.register_width / trial_bits)
                    regs_saved = regs_at_8 - regs_at_trial
                    
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([layer_name, "N/A", i, trial_bits, acc, total_drop, "ACCEPTED", regs_saved])
                    
                    # Periodic checkpoint
                    if i % 10 == 0:
                        with open('alexnet_full_surgical_search_result.json', 'w') as f_out:
                            json.dump(best_config, f_out, indent=4)
                    break 
                else:
                    # REJECTED: Staying at 8-bit or moving to next trial
                    if trial_bits == 4:
                        with open(log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([layer_name, "N/A", i, "REJECTED", acc, total_drop, "REJECTED", 0])
        
        # Save after each layer completes
        with open('alexnet_full_surgical_search_result.json', 'w') as f_out:
            json.dump(best_config, f_out, indent=4)

    # Final Summary
    with open('alexnet_full_surgical_search_result.json', 'w') as f:
        json.dump(best_config, f, indent=4)
        
    print(f"\n[FINAL] Full Model Optimal Surgical Config saved to alexnet_full_surgical_search_result.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--checkpoint', type=str, default='models/qalex-0-7.pth')
    parser.add_argument('--register-width', type=int, default=16)
    parser.add_argument('--max-drop', type=float, default=5.0) # More reasonable for research
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()
    
    run_joint_surgical_search(args)
