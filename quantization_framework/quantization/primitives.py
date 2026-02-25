import torch

def quantize_tensor_symmetric(x, bit_width=8, min_val=None, max_val=None, channel_dim=None):
    """
    Symmetric quantization: Maps [-max_abs, max_abs] to [-2^(b-1), 2^(b-1)-1].
    Scale = max_abs / (2^(b-1) - 1)
    """
    q_min = -(2 ** (bit_width - 1))
    q_max = 2 ** (bit_width - 1) - 1

    if min_val is None or max_val is None:
        if channel_dim is None:
            max_abs = torch.max(torch.abs(x))
        else:
            # Calculate max_abs along specific dimensions
            # For shapes like (N, C, H, W) and channel_dim=1:
            # We want to keep dim 1, and reduce all others.
            # Easiest way: flatten all except channel_dim, then max.
            
            # Move channel dim to front, flatten rest
            x_t = x.transpose(0, channel_dim) # (C, N, H, W...)
            x_flat = x_t.reshape(x_t.shape[0], -1)
            max_abs = torch.max(torch.abs(x_flat), dim=1)[0] # (C,)
            
            # Reshape scale for broadcasting back to x
            # shape: [1, C, 1, 1...]
            shape = [1] * x.ndim
            shape[channel_dim] = x.shape[channel_dim]
            max_abs = max_abs.view(*shape)
    else:
        max_abs = max(abs(min_val), abs(max_val))
        
    scale = max_abs / q_max
    
    # Handle zero scales
    scale = torch.where(scale == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), scale)
        
    x_int = torch.round(x / scale).clamp(q_min, q_max)
    x_quant = x_int * scale
    
    return x_quant, scale, torch.tensor(0.0)

def quantize_tensor_asymmetric(x, bit_width=8, min_val=None, max_val=None, channel_dim=None):
    """
    Asymmetric quantization: Maps [min, max] to [0, 2^b - 1].
    Scale = (max - min) / (2^b - 1)
    Zero_point = -round(min / scale)
    """
    q_min = 0
    q_max = 2 ** bit_width - 1
    
    if min_val is None or max_val is None:
        if channel_dim is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            x_t = x.transpose(0, channel_dim)
            x_flat = x_t.reshape(x_t.shape[0], -1)
            
            min_val = torch.min(x_flat, dim=1)[0]
            max_val = torch.max(x_flat, dim=1)[0]
            
            shape = [1] * x.ndim
            shape[channel_dim] = x.shape[channel_dim]
            min_val = min_val.view(*shape)
            max_val = max_val.view(*shape)
    
    scale = (max_val - min_val) / (q_max - q_min)
    scale = torch.where(scale == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), scale)
        
    zero_point = torch.round(-min_val / scale)
    zero_point = zero_point.clamp(q_min, q_max)

    x_int = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    x_quant = (x_int - zero_point) * scale
    
    return x_quant, scale, zero_point

def quantize_tensor(x, bit_width=8, method='symmetric', min_val=None, max_val=None, channel_dim=None):
    if method == 'symmetric':
        return quantize_tensor_symmetric(x, bit_width, min_val, max_val, channel_dim)
    elif method == 'asymmetric':
        return quantize_tensor_asymmetric(x, bit_width, min_val, max_val, channel_dim)
    else:
        raise ValueError(f"Unknown quantization method: {method}")
