"""
Verify W=A Constraint Compliance
=================================

Checks if a quantization config satisfies the W=A constraint
(every layer has matching weight and activation bit-widths).

Usage:
    python verify_wa_constraint.py --config levit_joint_config.json
"""

import argparse
import json
import sys


def verify_wa_constraint(config_path):
    """
    Verify that all layers have W=A constraint satisfied.

    Args:
        config_path: Path to config JSON file

    Returns:
        bool: True if all layers satisfy W=A, False otherwise
    """
    print("="*70)
    print("W=A CONSTRAINT VERIFICATION")
    print("="*70)
    print(f"Config file: {config_path}\n")

    # Load config
    with open(config_path, 'r') as f:
        data = json.load(f)

    # Handle both formats:
    # 1. {"config": {...}, "metadata": {...}}
    # 2. {"layer": {"weight": bits, "activation": bits}}
    if 'config' in data:
        config = data['config']
        metadata = data.get('metadata', {})
    else:
        config = data
        metadata = {}

    print(f"Total layers in config: {len(config)}")

    # Check each layer
    mismatches = []
    matches = []

    for layer_name, layer_config in config.items():
        # Handle different formats
        if isinstance(layer_config, dict):
            w_bits = layer_config.get('weight', None)
            a_bits = layer_config.get('activation', None)
        else:
            # Old format: single integer
            w_bits = layer_config
            a_bits = layer_config

        if w_bits is None or a_bits is None:
            print(f"⚠️  Warning: Layer '{layer_name}' missing W or A bits")
            continue

        if w_bits == a_bits:
            matches.append((layer_name, w_bits, a_bits))
        else:
            mismatches.append((layer_name, w_bits, a_bits))

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    if mismatches:
        print(f"❌ FAILED: Found {len(mismatches)} mismatches (W≠A)\n")
        print("Layers violating W=A constraint:")
        for layer, w, a in mismatches[:20]:  # Show first 20
            print(f"  {layer:50s} W{w}/A{a}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")

        print(f"\n⚠️  COMPLIANCE: {len(matches)}/{len(config)} layers ({len(matches)/len(config)*100:.1f}%)")
        print("="*70)
        return False
    else:
        print(f"✅ PASSED: All {len(matches)} layers satisfy W=A constraint!\n")

        # Show bit distribution
        bit_distribution = {}
        for _, w, a in matches:
            pair = f"W{w}/A{a}"
            bit_distribution[pair] = bit_distribution.get(pair, 0) + 1

        print("Bit-width distribution (W=A pairs):")
        for pair in sorted(bit_distribution.keys(), key=lambda x: int(x[1]), reverse=True):
            count = bit_distribution[pair]
            pct = (count / len(matches)) * 100
            print(f"  {pair}: {count:3d} layers ({pct:5.1f}%)")

        # Calculate average
        avg_bits = sum(w for _, w, _ in matches) / len(matches)
        print(f"\nAverage bits per layer: {avg_bits:.2f}")

        print(f"\n✅ COMPLIANCE: 100% ({len(matches)}/{len(config)} layers)")

        # Show metadata if available
        if metadata:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                if key != 'bit_distribution':  # Skip dict
                    print(f"  {key}: {value}")

        print("="*70)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify W=A Constraint Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a joint config
  python verify_wa_constraint.py --config levit_joint_config.json

  # Verify multiple configs
  python verify_wa_constraint.py --config vgg_config.json
  python verify_wa_constraint.py --config resnet_config.json
        """
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to quantization config JSON file')

    args = parser.parse_args()

    # Verify
    try:
        compliant = verify_wa_constraint(args.config)
        exit_code = 0 if compliant else 1
        sys.exit(exit_code)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in config file: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
