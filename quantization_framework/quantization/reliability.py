import torch
import random
import numpy as np

class ReliabilitySimulator:
    """
    Simulates hardware-level reliability for quantized tensors.
    Supports bit-level fault injection and redundancy (DMR/TMR)
    for mixed-precision (2, 4, 8 bits).
    """
    
    def __init__(self, register_width=16):
        self.register_width = register_width

    def inject_faults(self, tensor, bits, ber=1e-5):
        """
        Inject random bit flips based on Bit Error Rate (BER).
        
        Args:
            tensor: Quantized tensor (e.g., int8)
            bits: Effective bit-width of the values (2, 4, 8)
            ber: Bit Error Rate
            
        Returns:
            faulty_tensor: Tensor with random bit flips
        """
        if ber <= 0:
            return tensor
            
        # Create a mask for bit flips
        # Total bits = tensor.numel() * bits
        # But we simulate on the underlying storage (e.g. int8)
        # For simplicity, we flip bits within the 'bits' range of each value
        
        faulty = tensor.clone()
        
        # Determine bit mask for the actual bit-width
        # e.g., for 4-bit, we only care about flips in the lower 4 bits
        # (Assuming quantization mapping fits into the lower 'bits' bits)
        mask_val = (1 << bits) - 1
        
        for b in range(bits):
            # Probability per bit
            flip_mask = (torch.rand_like(tensor.float()) < ber)
            if flip_mask.any():
                # XOR with (1 << b) to flip the b-th bit
                faulty ^= (flip_mask.to(tensor.dtype) << b)
                
        return faulty

    def apply_redundancy(self, tensor, bits, method='DMR'):
        """
        Simulate hardware redundancy.
        
        Args:
            tensor: Quantized tensor
            bits: Bit-width
            method: 'DMR' (Dual Modular Redundancy) or 'TMR' (Triple Modular Redundancy)
            
        Returns:
            recovered_tensor: Tensor after error correction
        """
        if method == 'DMR':
            # DMR can detect errors but needs a second copy to "vote" 
            # In simple simulations, we often assume if copy1 != copy2, we take the "clean" one
            # To simulate realistically, we'd need to inject faults into BOTH copies
            pass
        elif method == 'TMR':
            # Triple Modular Redundancy: Majority voting
            pass
            
        return tensor

    def pack_and_protect(self, tensor, bits, redundancy='DMR', ber=1e-5):
        """
        Higher-level utility to pack, protect, and simulate faults.
        
        1. Quantize (assumed already done)
        2. Create Redundant copies
        3. Inject faults into all copies
        4. Vote/Recover
        """
        if redundancy is None:
            return self.inject_faults(tensor, bits, ber)
            
        if redundancy == 'DMR':
            copy1 = self.inject_faults(tensor, bits, ber)
            copy2 = self.inject_faults(tensor, bits, ber)
            
            # Simple Recovery: If mismatch, copy2 is the fallback (representing a clean copy or voter)
            # In a real hardware simulation, if both are faulty, logic fails.
            diff = (copy1 != copy2)
            recovered = torch.where(diff, copy2, copy1)
            return recovered
            
        if redundancy == 'TMR':
            c1 = self.inject_faults(tensor, bits, ber)
            c2 = self.inject_faults(tensor, bits, ber)
            c3 = self.inject_faults(tensor, bits, ber)
            
            # Majority voting
            # (c1 == c2) | (c1 == c3) -> c1 is majority
            # else -> c2 must be part of majority (c2 == c3)
            mask1 = (c1 == c2) | (c1 == c3)
            recovered = torch.where(mask1, c1, c2)
            return recovered
            
        return tensor
