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
        
        # Dynamic Grouping: Conv (16) vs FC (1024) for speed
        if "conv" in layer_name:
            group_size = 16
        else:
            group_size = 1024 # Target 15M register result faster
            
        num_groups = math.ceil(num_filters / group_size)
    
        layer_bits = [8] * num_filters
        
        for g in tqdm(range(num_groups), desc=f"Searching {layer_name}"):
            start_f = g * group_size
            end_f = min((g + 1) * group_size, num_filters)
            
            # Greedy: Try 2-bit, then 4-bit
            for trial_bits in [2, 4]:
                temp_layer_bits = list(layer_bits)
                for i in range(start_f, end_f):
                    temp_layer_bits[i] = trial_bits
                
                # Update current best config with these trial bits for this layer
                test_config = dict(best_config)
                test_config[layer_name] = temp_layer_bits
                
                # Test Model
                torch.cuda.empty_cache()
                model = load_model(args.model, checkpoint_path=args.checkpoint)
                # Wrap with current test config
                model = wrap_model_for_mqf(model, test_config, register_width=args.register_width, joint=True)
                model.to(device)
                
                acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
                total_drop = baseline_acc - acc
                
                if total_drop <= args.max_drop:
                    # Accepted!
                    layer_bits = temp_layer_bits
                    best_config[layer_name] = layer_bits
                    
                    # Register Savings math: 2 filters per reg at 8-bit, 4 at 4-bit, 8 at 2-bit (for reg_w=16)
                    regs_at_8 = (end_f - start_f) / (args.register_width / 8)
                    regs_at_trial = (end_f - start_f) / (args.register_width / trial_bits)
                    regs_saved = int(regs_at_8 - regs_at_trial)
                    
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([layer_name, g, f"{start_f}-{end_f-1}", trial_bits, acc, total_drop, "ACCEPTED", regs_saved])
                    break # Success for this group
                else:
                    if trial_bits == 4:
                        with open(log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([layer_name, g, f"{start_f}-{end_f-1}", trial_bits, acc, total_drop, "REJECTED", 0])

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
