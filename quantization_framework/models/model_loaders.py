import os
import torch

# Default to looking for 'models' relative to the project root, or use env var
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(PROJECT_ROOT, '../models'))

# --- GLOBAL SHIM FOR QUANTO (Fixes SSH/VM registration issues) ---
import sys
try:
    # 1. Try to import real 'quanto' (from local path in requirements.txt)
    import quanto
    # Verify it's actually loaded and not just an empty namespace
    if hasattr(quanto, 'QBitsTensor'):
        print("✓ Successfully loaded local 'quanto' (modified version).")
    else:
        raise ImportError
except (ImportError, AttributeError):
    # 2. Fallback shim for optimum-quanto if necessary
    try:
        import optimum.quanto as quanto
        sys.modules['quanto'] = quanto
        import optimum.quanto.tensor
        sys.modules['quanto.tensor'] = sys.modules['optimum.quanto.tensor']
        print("✓ Mapped 'optimum.quanto' to 'quanto' globally for VM compatibility")
    except ImportError:
        print("⚠ WARNING: 'quanto' library not found. Quantization loading may fail.")
# -----------------------------------------------------------------

def get_model_size_info(model, checkpoint_path=None):
    """
    Calculate model size in MB.
    Returns dict with size information.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate memory size (assuming FP32)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # Get file size if checkpoint provided
    file_size_mb = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    return {
        'parameters': total_params,
        'size_mb': size_mb,
        'file_size_mb': file_size_mb
    }


def print_model_size(model, checkpoint_path=None, label="Model"):
    """Print model size information in a clean format."""
    info = get_model_size_info(model, checkpoint_path)
    
    print(f"{'='*60}")
    print(f"{label} Size:")
    print(f"  Parameters: {info['parameters']:,} | Memory: {info['size_mb']:.2f} MB", end="")
    if info['file_size_mb']:
        print(f" | File: {info['file_size_mb']:.2f} MB")
    else:
        print()
    print(f"{'='*60}")

def load_model(model_name, checkpoint_path=None, num_classes=10):
    """
    Load a model by name, optionally loading weights from a checkpoint.
    """
    import sys
    model = None
    
    # Pre-inject class into __main__ if it's a known custom name to allow pickling
    if model_name == 'alexnet':
        from .alexnet import AlexNet, fasion_mnist_alexnet
        if '__main__' in sys.modules:
            main_mod = sys.modules['__main__']
            setattr(main_mod, 'fasion_mnist_alexnet', fasion_mnist_alexnet)
            setattr(main_mod, 'AlexNet', AlexNet)

    if model_name == 'vgg11_bn':
        from .vgg import vgg11_bn
        model = vgg11_bn(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'vgg11_bn.pt')
    elif model_name == 'levit':
        from .levit import levit_cifar
        model = levit_cifar(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'best3_levit_model_cifar10.pth')
    elif model_name == 'swin':
        from .swin import swin_tiny_patch4_window7_224
        model = swin_tiny_patch4_window7_224(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'best_swin_model_cifar_changed.pth')
    elif model_name == 'resnet':
        from .resnet import ResNet18
        model = ResNet18(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'road_0.9994904891304348.pth')
    elif model_name == 'alexnet':
        from .alexnet import alexnet
        model = alexnet(num_classes=num_classes)
        default_ckpt = os.path.join(MODELS_DIR, 'alexnet_fashionmnist.pth')
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    ckpt_to_load = checkpoint_path if checkpoint_path else default_ckpt
    
    if os.path.exists(ckpt_to_load):
        print(f"Loading checkpoint from {ckpt_to_load}")
        try:
            # Load with weights_only=False because custom classes (fasion_mnist_alexnet) are common here
            loaded = torch.load(ckpt_to_load, map_location='cpu', weights_only=False)
            
            # CASE 1: Checkpoint IS a full model object (e.g. quanto/Pickled model)
            if isinstance(loaded, torch.nn.Module):
                print(f"Detected full model object ({type(loaded).__name__}). Extracting weights.")
                # We extract the state_dict to avoid using the old class definition from the pickle
                loaded = loaded.state_dict()
                
            # CASE 2: Checkpoint is a dictionary (state_dict or mixed)
            if isinstance(loaded, dict):
                if 'state_dict' in loaded:
                    state_dict = loaded['state_dict']
                elif 'model_state_dict' in loaded:
                    state_dict = loaded['model_state_dict']
                elif 'model' in loaded:
                    state_dict = loaded['model']
                else:
                    state_dict = loaded
                
                # Check for 'quanto' keys that might need special mapping or just pass through
                # Clean prefix from keys
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if name.startswith('module.'): name = name[7:]
                    if name.startswith('model.'): name = name[6:]
                    cleaned_state_dict[name] = v
                
                final_state_dict = {}
                
                # First, identify and dequantize 8-bit weights for comparison
                quanto_bases = set()
                for k in cleaned_state_dict.keys():
                    if k.endswith('.weight._data'):
                        quanto_bases.add(k[:-13])
                
                if quanto_bases:
                    print(f"Restoring 8-bit weights for {len(quanto_bases)} layers (for MQF Baseline)...")
                    for k, v in cleaned_state_dict.items():
                        is_processed = False
                        for base in quanto_bases:
                            if k == f"{base}.weight._data":
                                w_int = v
                                scale = cleaned_state_dict[f"{base}.weight._scale"]
                                zp = cleaned_state_dict[f"{base}.weight._zeropoint"]
                                # Dequantize: (W - Z) * S
                                w_float = (w_int.float() - zp) * scale
                                final_state_dict[f"{base}.weight"] = w_float
                                is_processed = True
                                break
                            elif k.startswith(base) and ('.weight.' in k or k.endswith('.weight_qtype')):
                                is_processed = True # Skip metadata
                                break
                        
                        if not is_processed:
                            final_state_dict[k] = v
                else:
                    final_state_dict = cleaned_state_dict
                    
                missing, unexpected = model.load_state_dict(final_state_dict, strict=False)
                if len(missing) > 0 or len(unexpected) > 0:
                    print(f"Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
                    if len(unexpected) > 0:
                        print(f"First 5 unexpected keys: {list(unexpected)[:5]}")
            else:
                print(f"Unknown checkpoint type: {type(loaded)}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Returning randomly initialized model.")
    else:
        print(f"Checkpoint not found at {ckpt_to_load}")
        
    print_model_size(model, ckpt_to_load, label=f"{model_name}")
    return model
