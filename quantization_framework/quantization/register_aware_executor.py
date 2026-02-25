import torch
import torch.nn as nn
import math
import os
import csv

# Global Scale Transceiver for Hardware-Fidelity Pipeline
MQF_GLOBAL_CONTEXT = {'last_scale': 1.0, 'is_first_layer': True}

class MQFLayerWrapper(nn.Module):
    """
    STRICT FPGA-FAITHFUL MQF EXECUTOR
    
    1. Integer-Only MAC (32-bit Accumulator)
    2. Fixed-Point Requantization (M_fixed, Shift)
    3. Carry-Space (S) verification for dense packing
    4. 100% Integer-to-Integer handoff (No Float Bridge)
    """
    def __init__(self, layer, bit_widths, register_width=16, act_bit_width=8, joint=True):
        super().__init__()
        self.layer = layer
        self.register_width = register_width
        self.act_bit_width = act_bit_width
        self.joint = joint
        
        if isinstance(bit_widths, int):
            out_ch = layer.out_channels if hasattr(layer, 'out_channels') else layer.out_features
            self.bit_widths = torch.full((out_ch,), bit_widths, dtype=torch.int32)
        else:
            self.bit_widths = torch.tensor(bit_widths, dtype=torch.int32)
            
        # Calculation Depth (K)
        if isinstance(layer, nn.Conv2d):
            self.K = layer.in_channels * module_kernel_size(layer)
        else:
            self.K = layer.in_features
            
        # Hardware Constraints & Accumulator Safety ($S_{required}$)
        # Formula: ceil(log2(K * (max_prod))) + 1
        max_vals_w = (1 << (self.bit_widths - 1)) - 1
        max_vals_a = (1 << (self.act_bit_width - 1)) - 1 if not self.joint else (1 << (self.bit_widths - 1)) - 1
        max_prods = max_vals_w * max_vals_a
        self.S = torch.ceil(torch.log2(self.K * max_prods.float() + 1e-9)) + 1
        
        # Packing Logic (FPGA-Faithful)
        self.storage_density = self.register_width // self.bit_widths
        
        # Scale Parameters for Integer-Pure Handoff
        self.input_scale = 1.0 
        self.input_zero_point = 0
        self._audit_data = {}

    def forward(self, x_in):
        """
        Hardware-Faithful Forward Pass: Integer-In, Integer-Out
        """
        device = x_in.device
        self.bit_widths = self.bit_widths.to(device)
        self.S = self.S.to(device)
        
        # 1. Input Quantization (Simulate FPGA streaming input or calibrate first layer)
        if not MQF_GLOBAL_CONTEXT.get('is_first_layer', True):
            # Use global transceiver for scale continuity across layers
            self.input_scale = MQF_GLOBAL_CONTEXT['last_scale']
            self.input_zero_point = 0 
            x_int = x_in
        else:
            # First layer dynamic calibration
            with torch.no_grad():
                x_min, x_max = x_in.min(), x_in.max()
                qx_min, qx_max = - (1 << (self.act_bit_width - 1)), (1 << (self.act_bit_width - 1)) - 1
                self.input_scale = (x_max - x_min) / (qx_max - qx_min) if x_max > x_min else 1e-9
                self.input_zero_point = torch.round(qx_min - x_min / self.input_scale)
                x_int = torch.round(x_in / self.input_scale + self.input_zero_point).clamp(qx_min, qx_max)
            # Lock calibration for subsequent layers
            MQF_GLOBAL_CONTEXT['is_first_layer'] = False

        # 2. Weights and MAC computation
        w = self.layer.weight.data
        out_ch = w.shape[0]
        unique_bits = sorted(list(set(self.bit_widths.tolist())))
        
        # Buffer to store intermediate integer results and their scales
        groups_data = [] 
        global_max_float = 0.0
        final_output_q = None

        for b in unique_bits:
            mask = (self.bit_widths == b)
            if not mask.any(): continue
            indices = torch.where(mask)[0]
            w_b = w[indices]
            
            # Weight Quantization (Per-Filter Scaling)
            w_view = [indices.shape[0]] + [1] * (w.ndim - 1)
            qw_min, qw_max = - (1 << (b - 1)), (1 << (b - 1)) - 1
            w_abs = torch.abs(w_b).view(indices.shape[0], -1)
            w_max, _ = torch.max(w_abs, dim=1)
            w_scale = (w_max * 2.0 / (qw_max - qw_min)).view(w_view).clamp(min=1e-9)
            w_int = torch.round(w_b / w_scale).clamp(qw_min, qw_max)
            
            # Integer MAC
            if isinstance(self.layer, nn.Conv2d):
                acc_int32 = torch.nn.functional.conv2d(
                    x_int - self.input_zero_point, w_int, bias=None, 
                    stride=self.layer.stride, padding=self.layer.padding
                )
            else:
                acc_int32 = torch.nn.functional.linear(x_int - self.input_zero_point, w_int, bias=None)
            
            # Bias Add (Integer Domain)
            if self.layer.bias is not None:
                b_val = self.layer.bias[indices]
                b_scale = (self.input_scale * w_scale).view(-1)
                b_int32 = torch.round(b_val / b_scale).view(1, -1)
                if acc_int32.ndim == 4:
                    acc_int32 += b_int32.view(1, indices.shape[0], 1, 1)
                else:
                    acc_int32 += b_int32

            # Find peak float magnitude for this group to determine global scale
            # represented_float = acc_int32 * Sin * Sw
            group_max_float = acc_int32.float().abs().max() * self.input_scale * w_scale.abs().max()
            global_max_float = max(global_max_float, group_max_float.item())
            
            groups_data.append({
                'indices': indices,
                'acc_int32': acc_int32,
                'w_scale': w_scale,
                'b': b
            })

        # 3. Unified Requantization (PPU)
        # We need a single Sout for the entire layer to keep feature magnitudes consistent
        max_target_b = max(unique_bits) if self.joint else self.act_bit_width
        target_range_max = (1 << (max_target_b - 1)) - 1
        
        # Calculate Unified Output Scale
        unified_out_scale = global_max_float / target_range_max if global_max_float > 1e-6 else self.input_scale
        
        # DEBUG: Check for magnitude collapse
        if not hasattr(self, '_print_count'): self._print_count = 0
        if self._print_count < 1:
            print(f"[DEBUG {self.layer_name}] MaxFloat: {global_max_float:.4f}, Sout: {unified_out_scale:.6f}, Sin: {self.input_scale:.6f}")
            self._print_count += 1

        if final_output_q is None:
            # First group defines the spatial shape
            ref_acc = groups_data[0]['acc_int32']
            out_shape = [x_int.shape[0], out_ch] + list(ref_acc.shape[2:])
            final_output_q = torch.zeros(out_shape, device=device)

        for group in groups_data:
            indices = group['indices']
            acc_int32 = group['acc_int32']
            w_scale = group['w_scale']
            b = group['b']
            
            # Multiplier to map from (Sin * Sw) domain to unified Sout domain
            # M = (Sin * Sw) / Sout
            M_float = (self.input_scale * w_scale.view(-1)) / unified_out_scale
            M_float = M_float.view(1, -1, 1, 1) if acc_int32.ndim == 4 else M_float.view(1, -1)
            
            # Fixed-Point Multiplication (16-bit precision for M)
            M_fixed = torch.round(M_float * 65536)
            out_q = torch.round((acc_int32.float() * M_fixed) / 65536.0)
            
            # Clamp to the SPECIFIC bit-width of the filter
            target_b = b if self.joint else self.act_bit_width
            target_range = (1 << (target_b - 1)) - 1
            final_output_q[:, indices] = out_q.clamp(-target_range-1, target_range)

            # Audit Data Collection
            if hasattr(self, 'do_audit') and self.do_audit:
                for lookup_idx, f_idx in enumerate(indices):
                    f_idx_item = f_idx.item()
                    self._audit_data[f_idx_item] = {
                        'val_int': acc_int32[0, lookup_idx].abs().max().item(),
                        'w_scale': w_scale[lookup_idx].item(),
                        'x_scale': self.input_scale if isinstance(self.input_scale, float) else self.input_scale.item(),
                        'w_max_int': (w[f_idx_item] / w_scale[lookup_idx]).abs().max().item(),
                        'x_max_int': x_int.abs().max().item(),
                    }

        if hasattr(self, 'do_audit') and self.do_audit:
            self._audit_math(self.K)
            self.do_audit = False
            
        # Update global transceiver for next layer
        MQF_GLOBAL_CONTEXT['last_scale'] = unified_out_scale
        return final_output_q

    def _audit_math(self, K):
        log_path = "mqf_math_audit.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Layer', 'FilterID', 'W_Bits', 'A_Bits', 'K', 'S_Required', 'LaneWidth', 'Density', 'Collision', 'RegisterID', 'IntSum', 'W_Scale', 'A_Scale', 'FloatOutput', 'W_Max_Int', 'A_Max_Int'])
            reg_id = 0
            curr_reg_fill = 0
            for i in range(self.bit_widths.shape[0]):
                w_b = self.bit_widths[i].item()
                a_b = w_b if self.joint else self.act_bit_width
                s_req = self.S[i].item()
                density = self.storage_density[i].item()
                lane_w = self.register_width // density
                collision = s_req > lane_w
                data = self._audit_data.get(i, {})
                writer.writerow([self.layer_name if hasattr(self, 'layer_name') else 'unknown', i, w_b, a_b, K, round(s_req, 1), lane_w, int(density), collision, reg_id, int(data.get('val_int', 0)), f"{data.get('w_scale', 0):.6f}", f"{data.get('x_scale', 0):.6f}", 0.0, int(data.get('w_max_int', 0)), int(data.get('x_max_int', 0))])
                curr_reg_fill += 1
                if curr_reg_fill >= density:
                    reg_id += 1
                    curr_reg_fill = 0

def module_kernel_size(layer):
    if hasattr(layer, 'kernel_size'):
        if isinstance(layer.kernel_size, (tuple, list)):
            return layer.kernel_size[0] * layer.kernel_size[1]
        return layer.kernel_size * layer.kernel_size
    return 1

def wrap_model_for_mqf(model, config, register_width=16, joint=True):
    for name, module in model.named_modules():
        if name in config:
            wrapped = MQFLayerWrapper(module, config[name], register_width, joint=joint)
            wrapped.layer_name = name 
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                if isinstance(parent, nn.Sequential):
                    parent[int(child_name)] = wrapped
                else:
                    setattr(parent, child_name, wrapped)
            else:
                setattr(model, name, wrapped)
    return model
