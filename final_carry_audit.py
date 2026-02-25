import os
import sys
import torch
import json
import math

# Add quantization_framework to path
sys.path.append(os.path.join(os.getcwd(), 'quantization_framework'))

from models.model_loaders import load_model
from quantization.register_aware_executor import wrap_model_for_mqf, MQFLayerWrapper

def perform_carry_audit(config_path, register_width=16):
    print("\n" + "="*80)
    print(f"MQF HARDWARE CARRY-SAFE AUDIT (Register Width: {register_width}-bit)")
    print("="*80)

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Load Model Structure (to get K)
    model = load_model('alexnet', checkpoint_path='models/qalex-0-7.pth')
    model = wrap_model_for_mqf(model, config, register_width=register_width, joint=True)
    
    print(f"{'Layer':<12} | {'S_req':<8} | {'Density':<8} | {'LaneW':<6} | {'Status'}")
    print("-" * 80)
    
    total_regs = 0
    total_params = 0
    collisions = 0

    for name, module in model.named_modules():
        if isinstance(module, MQFLayerWrapper):
            # We check the worst-case S for the layer
            s_req = torch.max(module.S).item()
            density = torch.min(module.storage_density).item()
            lane_w = register_width // int(density)
            
            status = "PASS"
            if s_req > lane_w:
                status = "COLLISION"
                collisions += 1
            elif s_req > register_width:
                status = "OVERFLOW" # Even a single lane overflows
                collisions += 1
                
            print(f"{name:<12} | {s_req:>7.1f} | {int(density):>7} | {lane_w:>5} | {status}")
            
            # Efficiency Stats
            w = module.layer.weight.data
            out_ch = w.shape[0]
            params_per_ch = w[0].numel()
            
            layer_regs = 0
            for i in range(out_ch):
                d = module.storage_density[i].item()
                layer_regs += math.ceil(params_per_ch / d)
            
            total_regs += layer_regs
            total_params += w.numel()

    print("-" * 80)
    print(f"Total 16-bit Registers: {total_regs:,}")
    print(f"Total Parameters:       {total_params:,}")
    print(f"Carry Collisions:       {collisions}")
    print(f"Hardware Logic:         {'STRICT' if collisions == 0 else 'UNSAFE'}")
    print("="*80 + "\n")

    if collisions > 0:
        print("WARNING: Collisions detected. This means the accumulated sum exceeds the")
        print("allocated lane width in the packed register. Precision will be lost or")
        print("overflow will occur on FPGA hardware.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='alexnet_full_surgical_search_result.json')
    args = parser.parse_args()
    perform_carry_audit(args.config)
