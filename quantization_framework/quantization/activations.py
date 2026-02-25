import torch
import torch.nn as nn

class ActivationQuantizer(nn.Module):
    """
    Module to quantize activations during Forward Pass.
    Used for QAT (Fake Quantization) and Calibration.
    """
    def __init__(self, bit_width=8, method='asymmetric', momentum=0.9):
        super().__init__()
        self.bit_width = bit_width
        self.method = method
        self.momentum = momentum
        
        # State buffers
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        
        # Flags
        self.initialized = False
        
    def forward(self, x):
        # CRITICAL: Ensure all buffers are on the same device as input
        target_device = x.device
        if self.running_min.device != target_device:
            self.running_min = self.running_min.to(target_device)
            self.running_max = self.running_max.to(target_device)
            self.scale = self.scale.to(target_device)
            self.zero_point = self.zero_point.to(target_device)
        
        if self.training:
            # Update ranges
            current_min = x.detach().min()
            current_max = x.detach().max()
            
            if not self.initialized:
                self.running_min.copy_(current_min)
                self.running_max.copy_(current_max)
                self.initialized = True
            else:
                self.running_min.mul_(self.momentum).add_(current_min * (1 - self.momentum))
                self.running_max.mul_(self.momentum).add_(current_max * (1 - self.momentum))
        
        # Calculate Scale/ZP based on running stats
        q_min = 0
        q_max = 2 ** self.bit_width - 1
        
        scale = (self.running_max - self.running_min) / (q_max - q_min)
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        
        zero_point = -self.running_min / scale
        zero_point = torch.round(zero_point).clamp(q_min, q_max)
        
        # Update buffers in-place to preserve registration and device
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        
        # Fake Quantize using the registered buffers
        # x_int = round(x / s + z)
        # x_dequant = (x_int - z) * s
        
        x_int = torch.round(x / self.scale + self.zero_point).clamp(q_min, q_max)
        x_dq = (x_int - self.zero_point) * self.scale
        
        return x_dq

    def extra_repr(self):
        return f"bit_width={self.bit_width}, method={self.method}"
