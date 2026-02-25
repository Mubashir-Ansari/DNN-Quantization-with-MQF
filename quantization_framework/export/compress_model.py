import torch
import numpy as np
import os
import json

def pack_weights_to_bits(weight_tensor, bit_width):
    """
    Pack FP32 weights into actual bit-width representation.
    Returns packed bytes and metadata for unpacking.
    """
    # Quantize to integers in the bit-width range
    w_min, w_max = weight_tensor.min().item(), weight_tensor.max().item()
    if w_max == w_min:
        scale = 1.0
        zero_point = 0
    else:
        scale = (w_max - w_min) / (2**bit_width - 1)
        zero_point = -w_min / scale
    
    # Quantize to integers
    if scale == 0:
        w_int = torch.zeros_like(weight_tensor, dtype=torch.uint8)
    else:
        w_int = torch.round((weight_tensor - w_min) / scale).to(torch.uint8)
    
    # Pack into bytes (for 4-bit, pack 2 values per byte)
    if bit_width == 4:
        # Pack two 4-bit values into one byte
        w_packed = torch.zeros(w_int.numel() // 2 + w_int.numel() % 2, dtype=torch.uint8)
        # We can use vectorized operations for speed instead of loop
        # But keeping the loop from the request for safety/correctness matching
        w_flat = w_int.view(-1)
        # Handle even pairs
        n_pairs = w_flat.numel() // 2
        
        # vectorized packing attempt for speed? 
        # sticking to the provided logic but optimizing slightly for huge tensors if possible
        # Actually provided code had explicit loop. Let's stick to provided logic 
        # but correctly implement it. Iterating tensor elements in python is SLOW.
        # I'll use a slightly vectorized approach for performance.
        
        w_flat = w_int.flatten()
        if w_flat.numel() % 2 != 0:
            w_flat = torch.cat([w_flat, torch.tensor([0], dtype=torch.uint8, device=w_flat.device)])
        
        # Reshape to (N/2, 2)
        w_pairs = w_flat.view(-1, 2)
        # Shift first col by 4 and OR with second
        w_packed = (w_pairs[:, 0] << 4) | w_pairs[:, 1]
        
    elif bit_width == 8:
        w_packed = w_int.view(-1)
        
    elif bit_width == 2:
        # Pack four 2-bit values into one byte
        w_flat = w_int.flatten()
        padding = (4 - (w_flat.numel() % 4)) % 4
        if padding > 0:
            w_flat = torch.cat([w_flat, torch.tensor([0] * padding, dtype=torch.uint8, device=w_flat.device)])
        
        w_quads = w_flat.view(-1, 4)
        w_packed = (w_quads[:, 0] << 6) | (w_quads[:, 1] << 4) | (w_quads[:, 2] << 2) | w_quads[:, 3]
        
    else:
        raise ValueError(f"Bit-width {bit_width} not supported for packing")
    
    metadata = {
        'scale': float(scale),
        'zero_point': float(zero_point),
        'min': float(w_min),
        'max': float(w_max),
        'shape': list(weight_tensor.shape),
        'bit_width': bit_width
    }
    
    return w_packed.cpu().numpy().tobytes(), metadata


def compress_model(model, config, output_path):
    """
    Compress model weights according to config and save to disk.
    Returns actual compressed file size in MB.
    """
    compressed_data = {}
    total_bytes = 0
    
    print(f"\nCompressing model to {output_path}...")
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            
            if name in config:
                bit_width = config[name]
                
                if isinstance(bit_width, list):
                    # Granular: compress each channel separately
                    compressed_data[name] = {'type': 'granular', 'channels': []}
                    for i, bits in enumerate(bit_width):
                        if i < weight.shape[0]:
                            packed, meta = pack_weights_to_bits(weight[i], bits)
                            compressed_data[name]['channels'].append({
                                'data': packed,
                                'metadata': meta
                            })
                            total_bytes += len(packed)
                else:
                    # Layer-wise: compress entire weight tensor
                    packed, meta = pack_weights_to_bits(weight, bit_width)
                    compressed_data[name] = {
                        'type': 'layer',
                        'data': packed,
                        'metadata': meta
                    }
                    total_bytes += len(packed)
            else:
                # Keep as FP32
                compressed_data[name] = {
                    'type': 'fp32',
                    'data': weight.cpu().numpy().tobytes(),
                    'shape': list(weight.shape)
                }
                total_bytes += weight.numel() * 4
    
    # Save to file
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    
    actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    theoretical_size_mb = total_bytes / (1024 * 1024)
    
    print(f"✓ Compressed model saved")
    print(f"  File size: {actual_size_mb:.2f} MB")
    print(f"  Theoretical: {theoretical_size_mb:.2f} MB")
    
    return actual_size_mb
