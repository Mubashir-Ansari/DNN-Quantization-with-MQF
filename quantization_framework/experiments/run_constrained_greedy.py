"""
Constrained Greedy Search for LeViT
Generates an "architecture-aware" quantization config.

Constraints:
- stem.* layers: Can go to 2-bit (robust)
- attn.* layers: Minimum 4-bit (sensitive)
- mlp.* layers: Minimum 4-bit (sensitive)
- head layer: 8-bit (critical)
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.greedy import greedy_search_constrained, load_sensitivity_profile
from models.model_loaders import load_model

# CONSTRAINTS: {pattern_in_layer_name: min_bits}
LEVIT_CONSTRAINTS = {
    "attn": 4,      # Attention layers: min 4-bit
    "mlp": 4,       # MLP layers: min 4-bit
    "head": 8,      # Classifier head: 8-bit
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Constrained Greedy Search')
    parser.add_argument('--model', type=str, default='levit')
    parser.add_argument('--checkpoint', type=str, default='./models/best3_levit_model_cifar10.pth')
    parser.add_argument('--profile', type=str, default='levit_granular_profile.csv')
    parser.add_argument('--target_drop', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='levit_smart_config.json')
    args = parser.parse_args()

    print("="*60)
    print("CONSTRAINED GREEDY SEARCH FOR LEVIT")
    print("="*60)
    print(f"Constraints: {LEVIT_CONSTRAINTS}")
    print("="*60)

    # Load Model
    print(f"Loading {args.model}...")
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=10)

    # Load Profile
    print(f"Loading sensitivity profile: {args.profile}")
    profile, is_granular = load_sensitivity_profile(args.profile)
    print(f"Mode: {'Granular' if is_granular else 'Layer-wise'}")

    # Run Constrained Search
    print("\nStarting Constrained Greedy Search...")
    config = greedy_search_constrained(
        model, 
        profile, 
        is_granular, 
        target_acc_drop=args.target_drop,
        constraints=LEVIT_CONSTRAINTS
    )

    # Save
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nSmart configuration saved to {args.output}")
    print("="*60)
    print("NEXT STEP: Run QAT with the smart config:")
    print(f"  python quantization_framework/experiments/qat_training.py \\")
    print(f"    --model levit \\")
    print(f"    --checkpoint ./models/best3_levit_model_cifar10.pth \\")
    print(f"    --config {args.output} \\")
    print(f"    --dataset cifar10 \\")
    print(f"    --epochs 10 --patience 3")
    print("="*60)
