"""
Quantization-Aware Training (QAT) for Mixed-Precision Models.

This module implements:
1. Fake Quantization modules with Straight-Through Estimator (STE)
2. QAT training loop that fine-tunes models with quantization simulation
3. Support for per-layer and per-channel bit-width configurations
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loaders import load_model
from evaluation.pipeline import (
    evaluate_accuracy, 
    get_cifar10_dataloader, 
    get_cifar100_dataloader, 
    get_gtsrb_dataloader,
    get_fashionmnist_dataloader
)

# ============================================================================
# Fake Quantization with Straight-Through Estimator (STE)
# ============================================================================

class FakeQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for fake quantization.
    Forward: Quantize -> Dequantize (simulate quantization error)
    Backward: Pass gradients straight through (ignore quantization)
    """
    @staticmethod
    def forward(ctx, x, bit_width, symmetric=True):
        if symmetric:
            q_min = -(2 ** (bit_width - 1))
            q_max = 2 ** (bit_width - 1) - 1
            max_abs = torch.max(torch.abs(x))
            scale = max_abs / q_max if q_max > 0 else torch.tensor(1.0)
            scale = scale.clamp(min=1e-8)  # Avoid division by zero
            
            x_int = torch.round(x / scale).clamp(q_min, q_max)
            x_quant = x_int * scale
        else:
            q_min = 0
            q_max = 2 ** bit_width - 1
            min_val = torch.min(x)
            max_val = torch.max(x)
            scale = (max_val - min_val) / (q_max - q_min)
            scale = scale.clamp(min=1e-8)
            zero_point = torch.round(-min_val / scale).clamp(q_min, q_max)
            
            x_int = torch.round(x / scale + zero_point).clamp(q_min, q_max)
            x_quant = (x_int - zero_point) * scale
            
        return x_quant
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient straight through
        return grad_output, None, None

def fake_quantize(x, bit_width, symmetric=True):
    """Wrapper for FakeQuantize autograd function."""
    return FakeQuantize.apply(x, bit_width, symmetric)

# ============================================================================
# QAT Model Wrapper
# ============================================================================

class QATWrapper(nn.Module):
    """
    Wraps model to apply fake quantization during training.
    Supports ONLY layer-wise quantization (not granular).
    """
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
        # Validate config is layer-wise (not granular)
        for layer_name, bits in config.items():
            if isinstance(bits, list):
                raise ValueError(
                    f"QAT does not support granular quantization! "
                    f"Layer '{layer_name}' has granular config: {bits}. "
                    f"Please use hardware_aware_search.py which outputs layer-wise configs."
                )
        
        print(f"QAT wrapper initialized for {len(config)} layers")
    
    def forward(self, x):
        # Apply fake quantization to weights before forward pass
        original_weights = {}
        
        for name, module in self.model.named_modules():
            if name in self.config and hasattr(module, 'weight'):
                bits = self.config[name]
                
                # Store original weight
                original_weights[name] = module.weight.data.clone()
                
                # Apply fake quantization (layer-wise)
                w = module.weight.data
                q_w = fake_quantize(w, bits, symmetric=True)
                module.weight.data = q_w
        
        # Forward pass with quantized weights
        output = self.model(x)
        
        # Restore original weights (gradient flows through FakeQuantize)
        for name, original_w in original_weights.items():
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    module.weight.data = original_w
        
        return output

# ============================================================================
# QAT Training Loop
# ============================================================================

def train_qat(model, config, train_loader, val_loader, 
              epochs=5, lr=1e-4, patience=3, device=None, best_model_path='qat_best_checkpoint.pth'):
    """
    Fine-tune model with Quantization-Aware Training.
    
    Args:
        model: Base model to fine-tune
        config: Bit-width configuration (layer -> bits or [bits])
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of fine-tuning epochs
        lr: Learning rate
        patience: Epochs to wait for improvement before early stopping
        device: Device to use (auto-detected if None)
    
    Returns:
        Fine-tuned model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.train()
    
    # Wrap model with QAT
    qat_model = QATWrapper(model, config)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward with fake quantization
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate_accuracy(model, val_loader, device=device)
        print(f"Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best: {best_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{patience})")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    print(f"\nQAT Training Complete. Best Val Acc: {best_acc:.2f}%")
    return model

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantization-Aware Training')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Bit-width config JSON')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--input-size', type=int, default=None)
    parser.add_argument('--output', type=str, default='qat_model.pth')
    
    args = parser.parse_args()

    # Auto-adjust batch size for memory-intensive models
    if args.model == 'swin' and args.batch_size > 16:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 16 for Swin Transformer (GPU memory)")
        args.batch_size = 16
    elif args.model == 'levit' and args.batch_size > 32:
        print(f"WARNING: Reducing batch size from {args.batch_size} to 32 for LeViT (GPU memory)")
        args.batch_size = 32

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load model
    print(f"Loading {args.model}...")
    if args.dataset == 'cifar100': num_classes = 100
    elif args.dataset == 'gtsrb': num_classes = 43
    elif args.dataset == 'fashionmnist': num_classes = 10
    else: num_classes = 10
    
    model = load_model(args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)

    # Load data - determine input size based on dataset and model
    if args.input_size:
        input_size = args.input_size
    elif args.dataset == 'gtsrb':
        input_size = 224  # GTSRB uses 224x224 for all models
    elif args.model in ['levit', 'swin']:
        input_size = 224
    elif args.model == 'alexnet' or args.dataset == 'fashionmnist':
        input_size = 224
    else:
        input_size = 32  # CIFAR-10/100 default
    
    print(f"Loading {args.dataset} (Input: {input_size}x{input_size})...")
    if args.dataset == 'cifar100':
        train_loader = get_cifar100_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_cifar100_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'gtsrb':
        train_loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_gtsrb_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    elif args.dataset == 'fashionmnist':
        train_loader = get_fashionmnist_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_fashionmnist_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    else:
        train_loader = get_cifar10_dataloader(batch_size=args.batch_size, train=True, input_size=input_size)
        val_loader = get_cifar10_dataloader(batch_size=args.batch_size, train=False, input_size=input_size)
    
    # Run QAT
    print(f"\nStarting QAT for {len(config)} layers...")
    best_path = f"{args.model}_qat_best.pth"
    model = train_qat(model, config, train_loader, val_loader, 
                      epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device,
                      best_model_path=best_path)
    
    # Save final model
    torch.save(model.state_dict(), args.output)
    print(f"Saved QAT model to {args.output}")
