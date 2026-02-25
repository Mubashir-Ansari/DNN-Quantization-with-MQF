import argparse
import json
import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from quantization.primitives import quantize_tensor
from quantization.activations import ActivationQuantizer
from evaluation.pipeline import evaluate_accuracy, get_cifar10_dataloader, get_cifar100_dataloader, get_gtsrb_dataloader, get_fashionmnist_dataloader
import torch.nn as nn


def insert_activation_quantizers(model, config=None, default_bit_width=8, quantize_activations=True):
    """
    Insert ActivationQuantizer modules after key layers.
    Modifies model in-place by wrapping activations using forward hooks.
    
    Args:
        model: PyTorch model to modify
        config: Optional dict mapping layer names to bit-widths
        default_bit_width: Bit-width for activation quantization if not in config (default: 8)
        quantize_activations: Whether to enable activation quantization (default: True)
    
    Returns:
        model: Modified model with activation quantizers attached
        quantizers: List of quantizer instances for calibration
    """
    if not quantize_activations:
        print("[INFO] Activation quantization disabled.")
        return model, []
    
    # IMPORTANT: Clean up existing hooks to prevent stacking
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
    
    # Track which layers to quantize
    layers_to_wrap = []
    
    # Check for custom activation mapping (e.g. for AlexNet to find post-ReLU/MaxPool)
    act_mapping = {}
    if hasattr(model, 'get_activation_mapping'):
        act_mapping = model.get_activation_mapping()
        # Mapping is { weight_module_name: activation_module_name }
        # Reverse mapping to find which bit_width to use for which activation module
        inv_mapping = {v: k for k, v in act_mapping.items()}
        
        for name, module in model.named_modules():
            if name in inv_mapping:
                # This module is an activation module we want to wrap
                weight_name = inv_mapping[name]
                layers_to_wrap.append((name, module, weight_name))
    else:
        for name, module in model.named_modules():
            # Fallback: Quantize activations after Conv, Linear layers
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers_to_wrap.append((name, module, name))
    
    # Insert quantizers using forward hooks
    quantizers = {}
    
    # Detect model's device to ensure quantizers match
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device('cpu')
    
    for name, module, weight_name in layers_to_wrap:
        # Use bit-width from config if available (keyed by weight name), else default
        # CRITICAL: For Matched-Precision MQF, we enforce bit_width = weight_bits
        bit_width = default_bit_width
        if config and weight_name in config:
            bit_width = config[weight_name]
            if isinstance(bit_width, list):
                # For granular configs, we take the max bit-width for the activation quantizer
                bit_width = max(bit_width)
                
        quantizer = ActivationQuantizer(bit_width=bit_width)
        quantizer.to(model_device)  # Move to model's device
        quantizers[name] = quantizer
        
        # Create closure to capture quantizer instance
        def make_hook(q):
            def hook(module, input, output):
                return q(output)
            return hook
        
        module.register_forward_hook(make_hook(quantizer))
    
    print(f"[ACTIVATION MQF] Inserted {len(quantizers)} matched-precision quantizers.")
    return model, quantizers


def calibrate_activation_quantizers(model, quantizers, dataloader, device=None, num_batches=10):
    """
    Calibrate activation quantizers by running inference on calibration data.
    Device-agnostic: Auto-detects model's device if not specified.
    
    Args:
        model: Model with quantizers attached
        quantizers: List of ActivationQuantizer instances
        dataloader: Calibration data loader
        device: Device to run on (auto-detected if None)
        num_batches: Number of batches to use for calibration (default: 10)
    """
    if not quantizers:
        return
    
    # Auto-detect device if not provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    
    print(f"[CALIBRATION] Calibrating activation quantizers with {num_batches} batches on {device}...")
    
    # Handle both list and dict for backward compatibility
    q_all = quantizers.values() if isinstance(quantizers, dict) else quantizers
    
    # Put quantizers in training mode to collect statistics
    for q in q_all:
        q.train()
        q.to(device)  # Ensure quantizers are on correct device
    
    # Put model in eval mode
    model.eval()
    
    # Run calibration
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    
    # Put quantizers in eval mode (use collected statistics)
    for q in q_all:
        q.eval()
    
    print(f"[CALIBRATION] Complete. Quantizers calibrated.")


def apply_mixed_precision(model, config, device=None, 
                          quantize_weights=True, quantize_activations=True,
                          act_bit_width=8):
    """
    Apply the bit-width configuration to model weights and activations.
    Supports both:
    - Layer-wise: config[layer] = int (e.g., 4)
    - Granular: config[layer] = list of ints (e.g., [2, 4, 2, 8, ...])
    
    Args:
        model: PyTorch model
        config: Bit-width configuration dict
        device: Device to place quantized weights (auto-detected if None)
        quantize_weights: Enable weight quantization (default: True)
        quantize_activations: Enable activation quantization (default: True)
        act_bit_width: Bit-width for activations (default: 8)
    
    Returns:
        model: Modified model
        quantizers: List of activation quantizers (empty if disabled)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if quantize_weights:
        print("Applying mixed-precision configuration to weights...")
        count = 0
        granular_count = 0
        
        for name, module in model.named_modules():
            if name in config:
                bits = config[name]
                if hasattr(module, 'weight'):
                    w = module.weight.data
                    
                    if isinstance(bits, list):
                        # GRANULAR MODE: Per-channel quantization
                        # bits is a list like [2, 4, 2, 8, ...]
                        # We need to quantize each output channel with its own bit-width
                        
                        q_w = w.clone()
                        for i, b in enumerate(bits):
                            if i < w.shape[0]:  # Ensure index is valid
                                # Quantize channel i with bit-width b
                                channel_w = w[i:i+1]  # Keep dims for proper broadcasting
                                q_channel, _, _ = quantize_tensor(channel_w, bit_width=b)
                                q_w[i] = q_channel.squeeze(0)
                        
                        module.weight.data = q_w.to(device)
                        granular_count += 1
                    else:
                        # LAYER-WISE MODE: Single bit-width for entire layer
                        q_w, scale, zero = quantize_tensor(w, bit_width=bits)
                        module.weight.data = q_w.to(device)
                    
                    count += 1
                    
        print(f"Applied quantization to {count} layers ({granular_count} granular).")
    else:
        print("[INFO] Weight quantization disabled.")
    
    # Apply activation quantization
    quantizers = []
    if quantize_activations:
        # Pass config to ensure activation bits STRICTLY match weight bits per layer
        model, quantizers = insert_activation_quantizers(model, config, quantize_activations=True)
    
    return model, quantizers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, required=True, help='Path to optimized config json')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'gtsrb', 'fashionmnist'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--input-size', type=int, default=None)
    
    args = parser.parse_args()

    # Auto-adjust batch size for memory-intensive models
    if args.model == 'swin' and args.batch_size > 16:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 16 for Swin Transformer (GPU memory)")
        args.batch_size = 16
    elif args.model == 'levit' and args.batch_size > 32:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 32 for LeViT (GPU memory)")
        args.batch_size = 32

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # VALIDATE CONFIG FORMAT
    print("Validating config format...")
    granular_count = 0
    layer_wise_count = 0
    
    for layer_name, bits in config.items():
        if isinstance(bits, list):
            granular_count += 1
            print(f"  WARNING: Layer '{layer_name}' has granular config (list of {len(bits)} values)")
        elif isinstance(bits, int):
            layer_wise_count += 1
        else:
            raise ValueError(f"Invalid config for layer '{layer_name}': {bits}")
    
    print(f"Config summary: {layer_wise_count} layer-wise, {granular_count} granular")
    
    if granular_count > 0:
        print("\n⚠️  WARNING: Config contains granular (per-channel) quantization!")
        print("This may cause severe accuracy degradation.")
        print("Recommended: Use hardware_aware_search.py for layer-wise configs.\n")
        
    # 2. Load Model
    print(f"Loading {args.model}...")
    if args.dataset == 'cifar100': num_classes = 100
    elif args.dataset == 'gtsrb': num_classes = 43
    elif args.dataset == 'fashionmnist': num_classes = 10
    else: num_classes = 10
    
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)
    
    device = args.device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # 3. Apply Config
    model, quantizers = apply_mixed_precision(model, config, device=args.device,
                                   quantize_weights=True,
                                   quantize_activations=True,
                                   act_bit_width=8)
    
    # 4. Load Data
    if args.input_size:
        input_size = args.input_size
    elif args.dataset == 'gtsrb':
        input_size = 224  # GTSRB uses 224x224 for all models
    elif args.model == 'alexnet' or args.dataset == 'fashionmnist':
        input_size = 227
    elif args.model in ['levit', 'swin']:
        input_size = 224
    else:
        input_size = 32  # CIFAR-10/100 default
    
    print(f"Loading {args.dataset} (Input: {input_size}x{input_size})...")
    if args.dataset == 'cifar100':
        loader = get_cifar100_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'fashionmnist':
        loader = get_fashionmnist_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    else:
        loader = get_cifar10_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    
    # 5. Calibrate activation quantizers if enabled
    if quantizers:
        calibrate_activation_quantizers(model, quantizers, loader, device=args.device, num_batches=10)
        
    # 6. Evaluate
    print("Measuring accuracy of Mixed-Precision Model...")
    acc = evaluate_accuracy(model, loader, device=args.device)
    print(f"Final Mixed-Precision Accuracy: {acc:.2f}%")
