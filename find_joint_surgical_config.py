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
    # Focus on Convolutional Backbone (FC layers are too fragile for 2-bit without QAT)
    layers_to_optimize = ["conv1.0", "conv2.0", "conv3.0", "conv4.0", "conv5.0"]
    # We keep FC layers at 8-bit to protect the model's "brain"
    best_config = {layer: 8 for layer in ["conv1.0", "conv2.0", "conv3.0", "conv4.0", "conv5.0", "fc1", "fc2", "fc3"]}
    
    model = load_model(args.model, checkpoint_path=args.checkpoint)
    model = wrap_model_for_mqf(model, best_config, register_width=args.register_width, joint=True)
    model.to(device)
    baseline_acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
    print(f"Original Baseline Accuracy (All 8-bit): {baseline_acc:.2f}%")
    
    current_best_acc = baseline_acc
    
    # 3. Surgical Search logic
    log_path = "full_model_surgical_hardware_proof.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Group', 'FilterRange', 'TargetBits', 'Acc', 'Drop', 'Status', 'RegsSaved'])
    
    print(f"\n[STEP 3] Starting Robust Sequential Convolutional Search...")
    print(f"Register Width: {args.register_width}, Max Layer-End Drop Allowed: {args.max_drop}%")
    
    for layer_name in layers_to_optimize:
        print(f"\n>>> Optimizing Layer: {layer_name}")
        
        # Snapshot of config before optimizing this layer
        pre_layer_config = {k: (list(v) if isinstance(v, list) else v) for k, v in best_config.items()}
        
        # Get number of filters in this layer
        temp_model = load_model(args.model, checkpoint_path=args.checkpoint)
        target_layer = dict(temp_model.named_modules())[layer_name]
        num_filters = target_layer.out_channels
        del temp_model
        
        layer_bits = [8] * num_filters
    
        # Surgical Pulse: Iterate through every single Filter
        for i in tqdm(range(num_filters), desc=f"Surgical Pulse: {layer_name}"):
            # Greedy Optimization: Try 2-bit, then 4-bit
            for trial_bits in [2, 4]:
                temp_layer_bits = list(layer_bits)
                temp_layer_bits[i] = trial_bits
                
                test_config = {k: (list(v) if isinstance(v, list) else v) for k, v in best_config.items()}
                test_config[layer_name] = temp_layer_bits
                
                torch.cuda.empty_cache()
                model = load_model(args.model, checkpoint_path=args.checkpoint)
                model = wrap_model_for_mqf(model, test_config, register_width=args.register_width, joint=True)
                model.to(device)
                model.eval()
                
                acc = evaluate_accuracy(model, loader, device=device, max_samples=args.samples)
                total_drop = baseline_acc - acc
                
                # Check for collisions in node i
                collision = False
                if os.path.exists("mqf_math_audit.csv"):
                    with open("mqf_math_audit.csv", 'r') as audit_f:
                        reader = csv.DictReader(audit_f)
                        for row in reader:
                            if row['Layer'] == layer_name and int(row['FilterID']) == i:
                                if row['Collision'] == 'True':
                                    collision = True
                                break
                
                if not collision and total_drop <= args.max_drop:
                    # ACCEPTED internally for this pulse
                    layer_bits[i] = trial_bits
                    best_config[layer_name] = layer_bits
                    
                    regs_saved = (1 / (args.register_width / 8)) - (1 / (args.register_width / trial_bits))
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([layer_name, "N/A", i, trial_bits, acc, total_drop, "ACCEPTED", regs_saved])
                    break 
                else:
                    if trial_bits == 4:
                        with open(log_path, 'a', newline='') as f:
                            csv.writer(f).writerow([layer_name, "N/A", i, "REJECTED", acc, total_drop, "REJECTED", 0])

        # --- AFTER LAYER MASTER CHECK ("Final Exam") ---
        print(f"\n[LAYER END] Running Master Check for {layer_name} (2000 samples)...")
        torch.cuda.empty_cache()
        model = load_model(args.model, checkpoint_path=args.checkpoint)
        model = wrap_model_for_mqf(model, best_config, register_width=args.register_width, joint=True)
        model.to(device)
        
        master_acc = evaluate_accuracy(model, loader, device=device, max_samples=2000)
        master_drop = baseline_acc - master_acc
        
        if master_drop > args.max_drop:
            print(f"!!! CRITICAL: Layer {layer_name} failed Master Check (Drop: {master_drop:.2f}%). Reverting layer.")
            best_config = pre_layer_config
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([layer_name, 'LAYER_TOTAL', 'N/A', 'N/A', master_acc, master_drop, "REVERTED", 0])
        else:
            print(f"SUCCESS: Layer {layer_name} passed Master Check (Accuracy: {master_acc:.2f}%). Saving config.")
            current_best_acc = master_acc
            with open('alexnet_full_surgical_search_result.json', 'w') as f_out:
                json.dump(best_config, f_out, indent=4)

    print(f"\n[FINAL] Robust Surgical Config saved to alexnet_full_surgical_search_result.json")
    print(f"Final Estimated Accuracy: {current_best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--checkpoint', type=str, default='models/qalex-0-7.pth')
    parser.add_argument('--register-width', type=int, default=16)
    parser.add_argument('--max-drop', type=float, default=5.0) # More reasonable for research
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()
    
    run_joint_surgical_search(args)
