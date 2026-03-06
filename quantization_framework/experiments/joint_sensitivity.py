"""
Joint Weight-Activation Sensitivity Analysis (W=A Constraint)
===============================================================

Co-optimizes quantization by testing BOTH weights AND activations together
at the SAME bit-width for each layer.

Methodology:
  - For EACH layer:
      1. Keep all other layers at FP32
      2. Quantize BOTH weights AND activations of that layer to the SAME bit-width
      3. Test all W=A pairs: (W2,A2), (W4,A4), (W6,A6), (W8,A8)
      4. Measure accuracy drop for each pair
      5. Restore to FP32, move to next layer

Output: CSV with per-layer joint sensitivity scores for W=A pairs

Usage:
    python joint_sensitivity.py \
        --model levit \
        --checkpoint models/best3_levit_model_cifar10.pth \
        --dataset cifar10 \
        --bits 2 4 6 8

    # Auto-generates: levit_sensitivity_2_4_6_8.csv (includes bit-widths)
"""

import argparse
import csv
import time
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.pipeline import (
    evaluate_accuracy,
    get_cifar10_dataloader,
    get_cifar100_dataloader,
    get_gtsrb_dataloader
)
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor


class JointQuantizer:
    """Applies W=A quantization to a single layer."""

    def __init__(self, layer_module, bit_width):
        self.layer_module = layer_module
        self.bit_width = bit_width
        self.original_weight = None
        self.hook_handle = None

    def apply_weight_quantization(self):
        """Quantize and replace layer weights."""
        if hasattr(self.layer_module, 'weight') and self.layer_module.weight is not None:
            # Save original weights
            self.original_weight = self.layer_module.weight.data.clone()

            # Quantize weights
            w = self.layer_module.weight.data
            q_w, _, _ = quantize_tensor(w, bit_width=self.bit_width, method='symmetric')
            self.layer_module.weight.data = q_w

    def setup_activation_quantization(self):
        """Register forward hook to quantize activations."""
        def quantize_hook(module, input, output):
            q_output, _, _ = quantize_tensor(output, bit_width=self.bit_width, method='symmetric')
            return q_output

        self.hook_handle = self.layer_module.register_forward_hook(quantize_hook)

    def restore(self):
        """Restore original weights and remove activation hook."""
        # Restore weights
        if self.original_weight is not None:
            self.layer_module.weight.data = self.original_weight
            self.original_weight = None

        # Remove activation hook
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def measure_joint_sensitivity(model, dataloader, bit_widths=[2, 4, 6, 8],
                               device='cuda', max_samples=2000):
    """
    Measure per-layer joint W=A sensitivity.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        bit_widths: List of bit-widths to test (same for W and A)
        device: Device to run on
        max_samples: Number of samples for evaluation

    Returns:
        results: List of dicts with per-layer joint sensitivity results
        baseline_acc: Baseline FP32 accuracy
    """
    print("\n" + "="*70)
    print("JOINT WEIGHT-ACTIVATION SENSITIVITY ANALYSIS (W=A Constraint)")
    print("="*70)
    print(f"Testing bit-widths: {bit_widths}")
    print(f"Co-optimization: Weight and Activation use SAME bits per layer")
    print(f"Max samples per test: {max_samples}")
    print("="*70)

    model.to(device)
    model.eval()

    # Measure baseline accuracy (FP32)
    print("\n[Step 1/2] Measuring baseline accuracy (FP32)...")
    baseline_start = time.time()
    baseline_acc = evaluate_accuracy(model, dataloader, device=device, max_samples=max_samples)
    baseline_time = time.time() - baseline_start

    print(f"Baseline: {baseline_acc:.2f}% (took {baseline_time:.1f}s)")

    # Get all quantizable layers
    layers_to_test = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                layers_to_test.append((name, module))

    print(f"\n[Step 2/2] Testing {len(layers_to_test)} layers with W=A co-optimization...")
    print("="*70)

    results = []
    total_tests = len(layers_to_test) * len(bit_widths)
    current_test = 0

    for layer_idx, (layer_name, layer_module) in enumerate(layers_to_test):
        print(f"\n[{layer_idx+1}/{len(layers_to_test)}] Layer: {layer_name}")
        print(f"  Type: {type(layer_module).__name__}")

        layer_results = {
            'layer': layer_name,
            'layer_type': type(layer_module).__name__,
            'baseline_acc': baseline_acc
        }

        # Test each bit-width with W=A constraint
        for bits in sorted(bit_widths):
            current_test += 1
            progress = (current_test / total_tests) * 100

            print(f"  Testing W{bits}/A{bits} (W=A)... ", end="", flush=True)

            test_start = time.time()

            # Create joint quantizer
            quantizer = JointQuantizer(layer_module, bits)

            try:
                # Apply BOTH weight AND activation quantization
                quantizer.apply_weight_quantization()
                quantizer.setup_activation_quantization()

                # Evaluate with joint quantization
                acc = evaluate_accuracy(model, dataloader, device=device, max_samples=max_samples)
                drop = baseline_acc - acc

                # Store results
                layer_results[f'accuracy_{bits}bit'] = round(acc, 2)
                layer_results[f'sensitivity_{bits}bit'] = round(drop, 2)

                test_time = time.time() - test_start

                # Status indicator
                if drop < 1.0:
                    status = "OK"
                elif drop < 5.0:
                    status = "MODERATE"
                elif drop < 20.0:
                    status = "HIGH"
                else:
                    status = "CRITICAL"

                print(f"Acc: {acc:.2f}% (drop: {drop:.2f}%) [{status}] ({test_time:.1f}s)")

            except Exception as e:
                print(f"Error: {e}")
                layer_results[f'accuracy_{bits}bit'] = 0.0
                layer_results[f'sensitivity_{bits}bit'] = 100.0

            finally:
                # Always restore (critical!)
                quantizer.restore()
                torch.cuda.empty_cache()

        results.append(layer_results)

    print("\n" + "="*70)
    print("JOINT SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)

    return results, baseline_acc


def save_results(results, output_file, bit_widths):
    """Save results to CSV."""
    if not results:
        print("No results to save!")
        return

    # Build fieldnames
    fieldnames = ['layer', 'layer_type', 'baseline_acc']
    for bits in sorted(bit_widths):
        fieldnames.extend([f'accuracy_{bits}bit', f'sensitivity_{bits}bit'])

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


def print_summary(results, bit_widths):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if not results:
        print("No results!")
        return

    # Find most sensitive layers at lowest bit-width
    min_bits = min(bit_widths)

    # Sort by sensitivity at min_bits
    sorted_results = sorted(
        results,
        key=lambda x: x.get(f'sensitivity_{min_bits}bit', 0),
        reverse=True
    )

    print(f"\nTop 10 Most Sensitive Layers (W{min_bits}/A{min_bits} joint quantization):")
    for i, result in enumerate(sorted_results[:10], 1):
        layer = result['layer']
        sensitivity = result.get(f'sensitivity_{min_bits}bit', 0)
        print(f"  {i:2d}. {layer:40s}: {sensitivity:6.2f}% drop")

    # Find most robust layers
    print(f"\nTop 10 Most Robust Layers (W{min_bits}/A{min_bits} joint quantization):")
    robust_sorted = sorted(sorted_results, key=lambda x: x.get(f'sensitivity_{min_bits}bit', 100))
    for i, result in enumerate(robust_sorted[:10], 1):
        layer = result['layer']
        sensitivity = result.get(f'sensitivity_{min_bits}bit', 0)
        print(f"  {i:2d}. {layer:40s}: {sensitivity:6.2f}% drop")

    # Statistics
    sensitivities = [r.get(f'sensitivity_{min_bits}bit', 0) for r in results]
    avg_sensitivity = sum(sensitivities) / len(sensitivities)
    max_sensitivity = max(sensitivities)
    min_sensitivity = min(sensitivities)

    print(f"\nStatistics (W{min_bits}/A{min_bits}):")
    print(f"  Average joint sensitivity: {avg_sensitivity:.2f}%")
    print(f"  Max joint sensitivity: {max_sensitivity:.2f}%")
    print(f"  Min joint sensitivity: {min_sensitivity:.2f}%")

    # Count by category
    critical = sum(1 for s in sensitivities if s > 20.0)
    high = sum(1 for s in sensitivities if 5.0 < s <= 20.0)
    moderate = sum(1 for s in sensitivities if 1.0 < s <= 5.0)
    robust = sum(1 for s in sensitivities if s <= 1.0)

    print(f"\nLayer Categories (W=A constraint):")
    print(f"  Critical (>20% drop): {critical} layers")
    print(f"  High (5-20% drop): {high} layers")
    print(f"  Moderate (1-5% drop): {moderate} layers")
    print(f"  Robust (<1% drop): {robust} layers")

    # Bit-width recommendations
    print(f"\nRecommended Bit-Width Allocation (W=A):")
    for bits in sorted(bit_widths, reverse=True):
        suitable = sum(1 for r in results if r.get(f'sensitivity_{bits}bit', 100) < 5.0)
        print(f"  W{bits}/A{bits}: {suitable} layers can tolerate (<5% drop)")

    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint Weight-Activation Sensitivity Analysis (W=A Constraint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-generates filename)
  python joint_sensitivity.py \\
      --model levit \\
      --checkpoint models/best3_levit_model_cifar10.pth \\
      --dataset cifar10

  # Custom output and bit-widths
  python joint_sensitivity.py \\
      --model vgg11_bn \\
      --checkpoint checkpoints/vgg11_bn.pt \\
      --dataset cifar10 \\
      --bits 4 6 8 \\
      --output results/vgg_joint_sensitivity.csv
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture (vgg11_bn, resnet, levit, swin)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar10', 'cifar100', 'gtsrb'],
                       help='Dataset name')

    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: {model}_sensitivity_{bits}.csv)')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 6, 8],
                       help='Bit-widths to test (default: 2 4 6 8)')
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Max samples per evaluation (default: 2000)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Auto-generate output filename if not provided (include bit-widths)
    if args.output is None:
        bits_str = "_".join(map(str, sorted(args.bits)))
        args.output = f"{args.model}_sensitivity_{bits_str}.csv"
        print(f"No output file specified, using: {args.output}")

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)

    # Setup
    print("="*70)
    print("SETUP")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Bit-widths: {args.bits}")
    print(f"W=A Constraint: ENFORCED (co-optimization)")
    print("="*70)

    # Determine number of classes and input size
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    # Input size based on model
    if args.model in ['swin', 'levit']:
        input_size = 224
    else:
        input_size = 32

    # Batch size based on model
    if args.model == 'swin':
        batch_size = 16
    elif args.model == 'levit':
        batch_size = 32
    else:
        batch_size = 128

    # Load dataloader
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'cifar100':
        dataloader = get_cifar100_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        dataloader = get_gtsrb_dataloader(batch_size=batch_size, train=False, input_size=input_size)
    else:
        dataloader = get_cifar10_dataloader(batch_size=batch_size, train=False, input_size=input_size)

    # Load model
    print(f"Loading model...")
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)

    if model is None:
        print(f"Error: Failed to load model")
        exit(1)

    print(f"Model loaded successfully")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Run joint sensitivity analysis
    start_time = time.time()

    results, baseline_acc = measure_joint_sensitivity(
        model=model,
        dataloader=dataloader,
        bit_widths=args.bits,
        device=args.device,
        max_samples=args.max_samples
    )

    total_time = time.time() - start_time

    # Save results
    save_results(results, args.output, args.bits)

    # Print summary
    print_summary(results, args.bits)

    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Output: {args.output}")
    print("\nNext step: Use this CSV with joint_search.py to generate W=A config")
