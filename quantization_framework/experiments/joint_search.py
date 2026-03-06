"""
Joint W=A Greedy Search with Co-optimization Constraint
========================================================

Generates optimal mixed-precision configuration where EVERY layer
has matching weight and activation bit-widths (W=A constraint).

Algorithm:
  1. Load joint sensitivity profile (from joint_sensitivity.py)
  2. Initialize all layers to highest bit-width (e.g., W8/A8)
  3. Greedily reduce bit-widths of least sensitive layers
  4. CONSTRAINT: Always assign same bits to W and A in each layer
  5. Stop when target accuracy drop is reached

Usage:
    python joint_search.py \
        --model levit \
        --checkpoint models/best3_levit_model_cifar10.pth \
        --profile levit_joint_sensitivity.csv \
        --dataset cifar10 \
        --bits 2 4 6 8 \
        --target-drop 3.0

    # Outputs: levit_config_2_4_6_8.json (+ _weight.json, _activation.json)
"""

import argparse
import csv
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model


def load_joint_sensitivity(profile_csv):
    """
    Load joint sensitivity profile from CSV.

    Returns:
        dict: {layer_name: {bits: sensitivity_score}}
    """
    sensitivity = {}

    with open(profile_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = row['layer']
            sensitivity[layer] = {'baseline_acc': float(row['baseline_acc'])}

            # Extract sensitivity scores
            for key, value in row.items():
                if key.startswith('sensitivity_') and 'bit' in key:
                    bits = int(key.replace('sensitivity_', '').replace('bit', ''))
                    sensitivity[layer][bits] = float(value)

    return sensitivity


def greedy_joint_search(sensitivity, bit_choices, target_drop=3.0, baseline_acc=None):
    """
    Greedy search with W=A constraint.

    Args:
        sensitivity: Joint sensitivity dict from load_joint_sensitivity()
        bit_choices: Available bit-widths (e.g., [2, 4, 6, 8])
        target_drop: Target accuracy drop (%)
        baseline_acc: Baseline accuracy (optional, auto-detected from sensitivity)

    Returns:
        config: {layer: {"weight": bits, "activation": bits}} with W=A enforced
        stats: Search statistics
    """
    # Get baseline accuracy
    if baseline_acc is None:
        sample_layer = next(iter(sensitivity.values()))
        baseline_acc = sample_layer.get('baseline_acc', 100.0)

    # Sort bit choices (highest to lowest)
    bit_choices_sorted = sorted(bit_choices, reverse=True)
    max_bits = bit_choices_sorted[0]

    print("\n" + "="*70)
    print("GREEDY JOINT SEARCH (W=A Constraint)")
    print("="*70)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Target drop: {target_drop}%")
    print(f"Bit choices: {bit_choices_sorted}")
    print(f"Max bits: {max_bits}")
    print(f"Constraint: W=A enforced for all layers")
    print("="*70)

    # Initialize all layers to max bits
    config = {}
    for layer in sensitivity.keys():
        config[layer] = max_bits  # Single value (W=A)

    # Build layer priority list (least sensitive first)
    # Use highest available bit-width's sensitivity for ranking
    layer_priority = []
    for layer, scores in sensitivity.items():
        # Use max_bits sensitivity as baseline sensitivity
        if max_bits in scores:
            avg_sensitivity = scores[max_bits]
        else:
            # Fallback: use average of all available sensitivities
            available_sens = [v for k, v in scores.items() if isinstance(k, int)]
            avg_sensitivity = sum(available_sens) / len(available_sens) if available_sens else 100.0

        layer_priority.append((layer, avg_sensitivity))

    # Sort by ascending sensitivity (least sensitive first)
    layer_priority.sort(key=lambda x: x[1])

    print(f"\nLayer priority (least to most sensitive):")
    for i, (layer, sens) in enumerate(layer_priority[:5], 1):
        print(f"  {i}. {layer:40s} sensitivity: {sens:.2f}%")
    print(f"  ...")
    for i, (layer, sens) in enumerate(layer_priority[-3:], len(layer_priority)-2):
        print(f"  {i}. {layer:40s} sensitivity: {sens:.2f}%")

    # Greedy reduction
    print(f"\n{'='*70}")
    print("GREEDY REDUCTION PHASE")
    print(f"{'='*70}")

    estimated_drop = 0.0
    moves = []

    for layer, _ in layer_priority:
        current_bits = config[layer]

        # Try lower bit-widths
        best_bits = current_bits
        for bits in bit_choices_sorted:
            if bits >= current_bits:
                continue  # Skip if not lower

            # Estimate drop for this bit-width
            if bits in sensitivity[layer]:
                layer_drop = sensitivity[layer][bits]
            else:
                # If exact bit-width not tested, skip
                continue

            # Check if we can afford this drop
            potential_drop = estimated_drop + layer_drop

            if potential_drop <= target_drop:
                # Accept this reduction
                best_bits = bits
                estimated_drop = potential_drop
                moves.append({
                    'layer': layer,
                    'from': current_bits,
                    'to': bits,
                    'layer_drop': layer_drop,
                    'cumulative_drop': estimated_drop
                })
                print(f"[Move {len(moves)}] {layer:40s}: W{current_bits}/A{current_bits} → W{bits}/A{bits} "
                      f"(drop: {layer_drop:.2f}%, cumulative: {estimated_drop:.2f}%)")
                break  # Take first acceptable reduction

        config[layer] = best_bits

    # Convert to standard format {layer: {"weight": bits, "activation": bits}}
    final_config = {}
    for layer, bits in config.items():
        final_config[layer] = {
            'weight': bits,
            'activation': bits  # W=A constraint enforced
        }

    # Calculate statistics
    bit_distribution = {}
    for bits in config.values():
        bit_distribution[bits] = bit_distribution.get(bits, 0) + 1

    avg_bits = sum(config.values()) / len(config)

    stats = {
        'total_layers': len(config),
        'bit_distribution': bit_distribution,
        'average_bits': round(avg_bits, 2),
        'estimated_drop': round(estimated_drop, 2),
        'estimated_accuracy': round(baseline_acc - estimated_drop, 2),
        'moves_made': len(moves),
        'target_drop': target_drop,
        'baseline_accuracy': baseline_acc
    }

    print(f"\n{'='*70}")
    print("SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total layers: {stats['total_layers']}")
    print(f"Moves made: {stats['moves_made']}")
    print(f"Estimated drop: {stats['estimated_drop']:.2f}%")
    print(f"Estimated accuracy: {stats['estimated_accuracy']:.2f}%")
    print(f"Average bits (W=A): {stats['average_bits']:.2f}")
    print(f"\nBit distribution (W=A pairs):")
    for bits in sorted(bit_distribution.keys(), reverse=True):
        count = bit_distribution[bits]
        pct = (count / stats['total_layers']) * 100
        print(f"  W{bits}/A{bits}: {count} layers ({pct:.1f}%)")
    print(f"{'='*70}\n")

    return final_config, stats


def save_config(config, output_path, stats):
    """
    Save configuration to JSON files.

    Generates TWO separate config files for compatibility with validate_config.py:
    1. *_weight_config.json - Weight bit-widths {layer: int}
    2. *_activation_config.json - Activation bit-widths {layer: int}

    Both files have IDENTICAL values (W=A constraint enforced).
    """
    # Extract weight and activation configs (they're identical due to W=A constraint)
    weight_config = {}
    activation_config = {}

    for layer, bits_dict in config.items():
        weight_config[layer] = bits_dict['weight']
        activation_config[layer] = bits_dict['activation']

    # Determine output paths
    # Remove .json extension and append _weight.json or _activation.json
    base_name = output_path.rsplit('.json', 1)[0]
    weight_path = f"{base_name}_weight.json"
    activation_path = f"{base_name}_activation.json"

    # Save weight config
    with open(weight_path, 'w') as f:
        json.dump(weight_config, f, indent=2)

    # Save activation config
    with open(activation_path, 'w') as f:
        json.dump(activation_config, f, indent=2)

    # Also save the joint config for reference
    output_data = {
        'config': config,
        'metadata': {
            'constraint': 'W=A (co-optimized)',
            'total_layers': stats['total_layers'],
            'bit_distribution': stats['bit_distribution'],
            'average_bits': stats['average_bits'],
            'estimated_drop': stats['estimated_drop'],
            'estimated_accuracy': stats['estimated_accuracy'],
            'baseline_accuracy': stats['baseline_accuracy'],
            'target_drop': stats['target_drop']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Configuration saved:")
    print(f"  - Weight config:      {weight_path}")
    print(f"  - Activation config:  {activation_path}")
    print(f"  - Joint config (ref): {output_path}")
    print(f"✓ Format: {{layer: bits_int}} (compatible with validate_config.py)")
    print(f"✓ Constraint: W=A enforced (all {stats['total_layers']} layers have matching bits)")

    return weight_path, activation_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint W=A Greedy Search with Co-optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-generates config filename)
  python joint_search.py \\
      --model levit \\
      --checkpoint models/best3_levit_model_cifar10.pth \\
      --profile levit_joint_sensitivity.csv \\
      --dataset cifar10

  # Custom target drop and output
  python joint_search.py \\
      --model vgg11_bn \\
      --checkpoint checkpoints/vgg11_bn.pt \\
      --profile vgg_joint_sensitivity.csv \\
      --dataset cifar10 \\
      --target-drop 2.0 \\
      --output results/vgg_joint_config.json
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture name')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--profile', type=str, required=True,
                       help='Joint sensitivity CSV file')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'gtsrb'],
                       help='Dataset name')

    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output config JSON path (default: {model}_config_{bits}.json)')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 6, 8],
                       help='Available bit-widths (default: 2 4 6 8)')
    parser.add_argument('--target-drop', type=float, default=3.0,
                       help='Target accuracy drop %% (default: 3.0)')
    parser.add_argument('--baseline-acc', type=float, default=None,
                       help='Baseline accuracy (auto-detected if not provided)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided (include bit-widths)
    if args.output is None:
        bits_str = "_".join(map(str, sorted(args.bits)))
        args.output = f"{args.model}_config_{bits_str}.json"
        print(f"No output file specified, using: {args.output}")

    # Validate inputs
    if not os.path.exists(args.profile):
        print(f"Error: Profile not found: {args.profile}")
        exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)

    print("="*70)
    print("SETUP")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Profile: {args.profile}")
    print(f"Dataset: {args.dataset}")
    print(f"Bit choices: {args.bits}")
    print(f"Target drop: {args.target_drop}%")
    print("="*70)

    # Load sensitivity profile
    print(f"\nLoading joint sensitivity profile...")
    sensitivity = load_joint_sensitivity(args.profile)
    print(f"✓ Loaded sensitivity data for {len(sensitivity)} layers")

    # Run greedy search
    config, stats = greedy_joint_search(
        sensitivity=sensitivity,
        bit_choices=args.bits,
        target_drop=args.target_drop,
        baseline_acc=args.baseline_acc
    )

    # Save configuration
    save_config(config, args.output, stats)

    # Get the generated config paths (already returned from save_config)
    base_name_display = args.output.rsplit('.json', 1)[0]
    weight_path_display = f"{base_name_display}_weight.json"
    activation_path_display = f"{base_name_display}_activation.json"

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"1. Validate the config:")
    print(f"   python validate_config.py \\")
    print(f"       --model {args.model} \\")
    print(f"       --checkpoint {args.checkpoint} \\")
    print(f"       --config {weight_path_display} \\")
    print(f"       --activation-config {activation_path_display} \\")
    print(f"       --dataset {args.dataset}")
    print(f"\n2. If accuracy is good, you're done!")
    print(f"   If accuracy drops too much, run QAT:")
    print(f"   python qat_training.py \\")
    print(f"       --model {args.model} \\")
    print(f"       --checkpoint {args.checkpoint} \\")
    print(f"       --config {weight_path_display} \\")
    print(f"       --activation-config {activation_path_display} \\")
    print(f"       --dataset {args.dataset}")
    print(f"{'='*70}\n")
