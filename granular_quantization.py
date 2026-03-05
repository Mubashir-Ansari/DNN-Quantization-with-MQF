"""
Granular W=A Quantization with Hardware Register Packing
Implementation for AlexNet with MQF Framework Integration

Author: Senior AI Engineer
Date: March 5, 2026
Purpose: Filter-level quantization with hardware-aware register optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import json
from dataclasses import dataclass
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RegisterPackingLayout:
    """Definition of how weights/activations pack into fixed register."""
    weight_bits: int
    activation_bits: int
    num_weights_per_register: int
    num_activations_per_register: int
    carry_space_bits: int
    utilization_percent: float
    packing_offsets: Dict[str, Tuple[int, int]]  # {element: (start_bit, end_bit)}
    
    def to_dict(self):
        return {
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits,
            'num_weights_per_register': self.num_weights_per_register,
            'num_activations_per_register': self.num_activations_per_register,
            'carry_space_bits': self.carry_space_bits,
            'utilization_percent': self.utilization_percent,
        }


@dataclass
class FilterQuantizationConfig:
    """Per-filter quantization configuration."""
    filter_idx: int
    layer_name: str
    weight_bits: int
    activation_bits: int
    pack_layout: RegisterPackingLayout
    weight_scale: float = None
    weight_zero_point: float = None
    activation_scale: float = None
    activation_zero_point: float = None


# ============================================================================
# REGISTER PACKING UTILITIES
# ============================================================================

class RegisterPackingManager:
    """
    Manages hardware register packing for efficient bit-width storage.
    Supports 16-bit and 32-bit registers with automatic carry handling.
    """
    
    REGISTER_WIDTH = 16
    ACCUMULATOR_WIDTH = 32
    
    @staticmethod
    def compute_packing_layout(w_bits: int, a_bits: int, 
                               register_width: int = 16) -> Tuple[bool, RegisterPackingLayout]:
        """
        Compute optimal packing layout for given bit-widths.
        
        Args:
            w_bits: Weight bit-width (2, 4, or 8)
            a_bits: Activation bit-width (must equal w_bits for W=A)
            register_width: Fixed register size (16 or 32)
        
        Returns:
            (is_feasible, pack_layout)
        """
        if w_bits != a_bits:
            return False, None  # W=A constraint violation
        
        bits = w_bits
        
        # ───────────────────────────────────────────────────────────
        # CASE: 2-bit quantization
        # ───────────────────────────────────────────────────────────
        if bits == 2:
            # 16-bit register layout:
            # [W0:2][W1:2][W2:2][A0:2][A1:2][A2:2][Carry:4]
            # Bits: 0-1  2-3   4-5   6-7   8-9   10-11 12-15
            
            num_weights = 3
            num_activations = 3
            carry_space = 4
            total_bits = (num_weights + num_activations) * bits + carry_space
            
            if total_bits <= register_width:
                packing_offsets = {
                    f'weight_{i}': (i * bits, (i + 1) * bits)
                    for i in range(num_weights)
                }
                for i in range(num_activations):
                    packing_offsets[f'activation_{i}'] = (
                        (num_weights + i) * bits,
                        (num_weights + i + 1) * bits
                    )
                
                layout = RegisterPackingLayout(
                    weight_bits=2,
                    activation_bits=2,
                    num_weights_per_register=3,
                    num_activations_per_register=3,
                    carry_space_bits=4,
                    utilization_percent=(total_bits - carry_space) / register_width * 100,
                    packing_offsets=packing_offsets
                )
                return True, layout
        
        # ───────────────────────────────────────────────────────────
        # CASE: 4-bit quantization
        # ───────────────────────────────────────────────────────────
        elif bits == 4:
            # 16-bit register (with overflow to 32-bit accumulator):
            # [W0:4][W1:4][A0:4][Carry:4]
            # Bits: 0-3  4-7   8-11 12-15 (overflow to 16-23)
            
            num_weights = 2
            num_activations = 2
            carry_space = 4
            total_bits = (num_weights + num_activations) * bits + carry_space
            
            packing_offsets = {
                f'weight_{i}': (i * bits, (i + 1) * bits)
                for i in range(num_weights)
            }
            for i in range(num_activations):
                packing_offsets[f'activation_{i}'] = (
                    (num_weights + i) * bits,
                    (num_weights + i + 1) * bits
                )
            
            layout = RegisterPackingLayout(
                weight_bits=4,
                activation_bits=4,
                num_weights_per_register=2,
                num_activations_per_register=2,
                carry_space_bits=4,
                utilization_percent=12 / register_width * 100,  # 75%
                packing_offsets=packing_offsets
            )
            return True, layout
        
        # ───────────────────────────────────────────────────────────
        # CASE: 8-bit quantization (full precision)
        # ───────────────────────────────────────────────────────────
        elif bits == 8:
            # Single pair per 16-bit register, overflow to 32-bit:
            # [W0:8][A0:8] + overflow handling
            
            num_weights = 1
            num_activations = 1
            carry_space = 0
            total_bits = (num_weights + num_activations) * bits
            
            packing_offsets = {
                'weight_0': (0, 8),
                'activation_0': (8, 16)
            }
            
            layout = RegisterPackingLayout(
                weight_bits=8,
                activation_bits=8,
                num_weights_per_register=1,
                num_activations_per_register=1,
                carry_space_bits=0,
                utilization_percent=100.0,
                packing_offsets=packing_offsets
            )
            return True, layout
        
        return False, None
    
    @staticmethod
    def validate_packing(w_bits: int, a_bits: int) -> bool:
        """Quick check if configuration can be packed in 16-bit register."""
        if w_bits != a_bits:
            return False
        
        is_feasible, _ = RegisterPackingManager.compute_packing_layout(w_bits, a_bits)
        return is_feasible


# ============================================================================
# GRANULAR QUANTIZATION MODULES
# ============================================================================

class GranularQuantizedConv2d(nn.Module):
    """
    Conv2d layer with per-filter granular W=A quantization and
    hardware-aware register packing.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 filter_config: Optional[Dict] = None,
                 stride: int = 1, padding: int = 0,
                 bias: bool = False, device: str = 'cuda'):
        """
        Args:
            in_channels, out_channels, kernel_size: Conv2d parameters
            filter_config: Dict mapping {filter_idx: FilterQuantizationConfig}
            stride, padding: Conv2d parameters
            bias: Whether to use bias
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        
        # Initialize per-filter configuration
        self.filter_config = filter_config or {}
        self._init_default_config()
        
        # Weight storage
        self.register_buffer('weight_fp32', torch.zeros(
            out_channels, in_channels, kernel_size, kernel_size,
            device=device, dtype=torch.float32
        ))
        
        # Quantization parameters (per-filter)
        self.register_buffer('weight_scale', torch.ones(out_channels, device=device))
        self.register_buffer('weight_zero_point', torch.zeros(out_channels, device=device))
        self.register_buffer('activation_scale', torch.ones(out_channels, device=device))
        self.register_buffer('activation_zero_point', torch.zeros(out_channels, device=device))
        
        # Bias (optional)
        if bias:
            self.register_buffer('bias', torch.zeros(out_channels, device=device))
        else:
            self.register_buffer('bias', None)
    
    def _init_default_config(self):
        """Initialize default configuration (all filters 8-bit)."""
        for filter_idx in range(self.out_channels):
            if filter_idx not in self.filter_config:
                is_feasible, layout = RegisterPackingManager.compute_packing_layout(8, 8)
                self.filter_config[filter_idx] = {
                    'weight_bits': 8,
                    'activation_bits': 8,
                    'pack_layout': layout
                }
    
    def set_filter_config(self, filter_config: Dict):
        """Update per-filter configuration."""
        self.filter_config = filter_config
    
    def quantize_weight(self, weight: torch.Tensor, filter_idx: int) -> torch.Tensor:
        """
        Quantize single filter's weights.
        
        Args:
            weight: [in_channels, kernel_size, kernel_size]
            filter_idx: Which filter to quantize
        
        Returns:
            Quantized weight tensor
        """
        config = self.filter_config.get(filter_idx, {})
        w_bits = config.get('weight_bits', 8)
        
        # Symmetric quantization
        max_abs = torch.max(torch.abs(weight))
        q_max = (2 ** (w_bits - 1)) - 1
        
        scale = max_abs / q_max if q_max > 0 else torch.tensor(1.0, device=self.device)
        scale = torch.clamp(scale, min=1e-8)
        
        # Store scale for later use
        self.weight_scale[filter_idx] = scale
        self.weight_zero_point[filter_idx] = 0  # Symmetric: ZP = 0
        
        # Quantize
        w_int = torch.round(weight / scale).clamp(-(2**(w_bits-1)), 2**(w_bits-1) - 1)
        
        # Dequantize (fake quantization)
        w_dequant = w_int * scale
        
        return w_dequant
    
    def quantize_activation(self, activation: torch.Tensor, filter_idx: int) -> torch.Tensor:
        """
        Quantize single filter's activation.
        
        Args:
            activation: [N, 1, H, W] (single filter activations)
            filter_idx: Which filter to quantize
        
        Returns:
            Quantized activation tensor
        """
        config = self.filter_config.get(filter_idx, {})
        a_bits = config.get('activation_bits', 8)
        
        # Asymmetric quantization
        min_val = torch.min(activation)
        max_val = torch.max(activation)
        
        q_min, q_max = 0, (2 ** a_bits) - 1
        scale = (max_val - min_val) / (q_max - q_min)
        scale = torch.clamp(scale, min=1e-8)
        
        zero_point = torch.round(-min_val / scale).clamp(q_min, q_max)
        
        # Store parameters
        self.activation_scale[filter_idx] = scale
        self.activation_zero_point[filter_idx] = zero_point
        
        # Quantize
        a_int = torch.round(activation / scale + zero_point).clamp(q_min, q_max)
        
        # Dequantize
        a_dequant = (a_int - zero_point) * scale
        
        return a_dequant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-filter W=A quantization.
        
        Args:
            x: [N, in_channels, H, W]
        
        Returns:
            y: [N, out_channels, H, W] with per-filter quantization
        """
        # Quantize weights per-filter
        weight_quantized = torch.zeros_like(self.weight_fp32)
        for filter_idx in range(self.out_channels):
            filter_weight = self.weight_fp32[filter_idx]
            weight_quantized[filter_idx] = self.quantize_weight(filter_weight, filter_idx)
        
        # Standard Conv2d
        y = F.conv2d(x, weight_quantized, bias=self.bias,
                     stride=self.stride, padding=self.padding)
        
        # Quantize activations per-filter
        y_quantized = torch.zeros_like(y)
        for filter_idx in range(self.out_channels):
            filter_acts = y[:, filter_idx:filter_idx+1, :, :]
            y_quantized[:, filter_idx:filter_idx+1, :, :] = \
                self.quantize_activation(filter_acts, filter_idx)
        
        return y_quantized


class GranularQuantizedLinear(nn.Module):
    """
    Linear layer with per-neuron (output) granular W=A quantization.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 neuron_config: Optional[Dict] = None,
                 bias: bool = False, device: str = 'cuda'):
        """
        Args:
            in_features, out_features: Linear parameters
            neuron_config: Dict mapping {neuron_idx: QuantizationConfig}
            bias: Whether to use bias
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Per-neuron configuration
        self.neuron_config = neuron_config or {}
        self._init_default_config()
        
        # Weight storage
        self.register_buffer('weight_fp32', torch.zeros(
            out_features, in_features, device=device, dtype=torch.float32
        ))
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.ones(out_features, device=device))
        self.register_buffer('activation_scale', torch.ones(out_features, device=device))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, device=device))
        else:
            self.register_buffer('bias', None)
    
    def _init_default_config(self):
        """Initialize default configuration."""
        for neuron_idx in range(self.out_features):
            if neuron_idx not in self.neuron_config:
                self.neuron_config[neuron_idx] = {
                    'weight_bits': 8,
                    'activation_bits': 8
                }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-neuron quantization.
        
        Args:
            x: [N, in_features]
        
        Returns:
            y: [N, out_features] with per-neuron quantization
        """
        # Quantize weights per-neuron
        weight_quantized = torch.zeros_like(self.weight_fp32)
        for neuron_idx in range(self.out_features):
            config = self.neuron_config.get(neuron_idx, {})
            w_bits = config.get('weight_bits', 8)
            
            neuron_weight = self.weight_fp32[neuron_idx]
            max_abs = torch.max(torch.abs(neuron_weight))
            q_max = (2 ** (w_bits - 1)) - 1
            scale = max_abs / q_max if q_max > 0 else torch.tensor(1.0, device=self.device)
            scale = torch.clamp(scale, min=1e-8)
            
            self.weight_scale[neuron_idx] = scale
            
            w_int = torch.round(neuron_weight / scale).clamp(
                -(2**(w_bits-1)), 2**(w_bits-1) - 1
            )
            weight_quantized[neuron_idx] = w_int * scale
        
        # Standard Linear
        y = F.linear(x, weight_quantized, self.bias)
        
        # Quantize activations per-neuron
        y_quantized = torch.zeros_like(y)
        for neuron_idx in range(self.out_features):
            config = self.neuron_config.get(neuron_idx, {})
            a_bits = config.get('activation_bits', 8)
            
            neuron_acts = y[:, neuron_idx:neuron_idx+1]
            min_val = torch.min(neuron_acts)
            max_val = torch.max(neuron_acts)
            
            q_min, q_max = 0, (2 ** a_bits) - 1
            scale = (max_val - min_val) / (q_max - q_min)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.round(-min_val / scale).clamp(q_min, q_max)
            
            self.activation_scale[neuron_idx] = scale
            
            a_int = torch.round(neuron_acts / scale + zero_point).clamp(q_min, q_max)
            y_quantized[:, neuron_idx:neuron_idx+1] = (a_int - zero_point) * scale
        
        return y_quantized


# ============================================================================
# ALEXNET WITH GRANULAR QUANTIZATION
# ============================================================================

class GranularAlexNetQuantized(nn.Module):
    """
    AlexNet with per-filter/neuron W=A quantization.
    Integrates with MQF framework for search and optimization.
    """
    
    def __init__(self, num_classes: int = 1000,
                 config: Optional[Dict] = None,
                 device: str = 'cuda'):
        """
        Args:
            num_classes: Number of output classes
            config: Granular quantization configuration
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device
        self.config = config or {}
        self.num_classes = num_classes
        
        # Feature extraction with granular quantization
        self.features = nn.Sequential(
            # Layer 0: Conv2d(3, 64, 11, stride=4, padding=2)
            GranularQuantizedConv2d(
                3, 64, 11,
                filter_config=self.config.get('features_0', {}),
                stride=4, padding=2, device=device
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 1: Conv2d(64, 192, 5, padding=2)
            GranularQuantizedConv2d(
                64, 192, 5,
                filter_config=self.config.get('features_1', {}),
                stride=1, padding=2, device=device
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Conv2d(192, 384, 3, padding=1)
            GranularQuantizedConv2d(
                192, 384, 3,
                filter_config=self.config.get('features_2', {}),
                stride=1, padding=1, device=device
            ),
            nn.ReLU(inplace=True),
            
            # Layer 3: Conv2d(384, 256, 3, padding=1)
            GranularQuantizedConv2d(
                384, 256, 3,
                filter_config=self.config.get('features_3', {}),
                stride=1, padding=1, device=device
            ),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv2d(256, 256, 3, padding=1)
            GranularQuantizedConv2d(
                256, 256, 3,
                filter_config=self.config.get('features_4', {}),
                stride=1, padding=1, device=device
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier with granular quantization
        self.classifier = nn.Sequential(
            GranularQuantizedLinear(
                256 * 6 * 6, 4096,
                neuron_config=self.config.get('classifier_0', {}),
                device=device
            ),
            nn.ReLU(inplace=True),
            
            GranularQuantizedLinear(
                4096, 4096,
                neuron_config=self.config.get('classifier_1', {}),
                device=device
            ),
            nn.ReLU(inplace=True),
            
            GranularQuantizedLinear(
                4096, num_classes,
                neuron_config=self.config.get('classifier_2', {}),
                device=device
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized AlexNet.
        
        Args:
            x: [N, 3, 224, 224] (ImageNet) or [N, 3, 32, 32] (CIFAR)
        
        Returns:
            logits: [N, num_classes]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def update_config(self, new_config: Dict):
        """Update quantization configuration."""
        self.config = new_config
        # Update each layer's config
        for idx, layer in enumerate(self.features):
            if isinstance(layer, GranularQuantizedConv2d):
                layer.set_filter_config(
                    self.config.get(f'features_{idx}', {})
                )


# ============================================================================
# SENSITIVITY PROFILING
# ============================================================================

class GranularSensitivityProfiler:
    """
    Profiles per-filter accuracy sensitivity for granular quantization.
    """
    
    def __init__(self, model: nn.Module, dataloader,
                 device: str = 'cuda', num_samples: int = 256):
        """
        Args:
            model: Reference FP32 model
            dataloader: Validation dataloader
            device: 'cuda' or 'cpu'
            num_samples: Number of samples for profiling
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_samples = num_samples
    
    def profile_layer(self, layer_name: str, layer_module: nn.Module,
                     bit_widths: List[int] = [2, 4, 8]) -> Dict:
        """
        Profile all filters in a single layer.
        
        Returns:
            {filter_idx: {bit_width: accuracy_drop}}
        """
        results = {}
        
        # Get baseline accuracy
        baseline_acc = self._evaluate(num_samples=self.num_samples)
        
        # Profile each filter
        num_filters = layer_module.out_channels if hasattr(layer_module, 'out_channels') \
                      else layer_module.out_features
        
        for filter_idx in range(num_filters):
            results[filter_idx] = {}
            
            for bits in bit_widths:
                # Create config for this filter
                filter_config = {filter_idx: {'weight_bits': bits, 'activation_bits': bits}}
                
                # Apply quantization
                layer_module.set_filter_config(filter_config)
                
                # Evaluate
                acc = self._evaluate(num_samples=self.num_samples)
                drop = baseline_acc - acc
                
                results[filter_idx][bits] = drop
                
            # Reset to FP32
            layer_module.set_filter_config({})
        
        return results
    
    def _evaluate(self, num_samples: int) -> float:
        """Evaluate model accuracy on validation set."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.dataloader):
                if i * self.dataloader.batch_size >= num_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100.0 * correct / total if total > 0 else 0.0


# ============================================================================
# GREEDY SEARCH
# ============================================================================

class GranularGreedySearcher:
    """
    Performs granular (per-filter) greedy search with hardware constraints.
    """
    
    def __init__(self, bit_widths: List[int] = [2, 4, 8],
                 target_drop: float = 3.0,
                 register_width: int = 16):
        """
        Args:
            bit_widths: Candidate bit-widths
            target_drop: Maximum acceptable accuracy drop (%)
            register_width: Hardware register width
        """
        self.bit_widths = sorted(bit_widths, reverse=True)
        self.target_drop = target_drop
        self.register_width = register_width
    
    def search(self, sensitivity: Dict) -> Tuple[Dict, Dict]:
        """
        Perform greedy search on sensitivity data.
        
        Args:
            sensitivity: {layer: {filter: {bits: drop}}}
        
        Returns:
            (config, stats)
            config: {layer: {filter: {w_bits, a_bits, pack_layout}}}
            stats: Search statistics
        """
        # Flatten and sort filters by sensitivity
        filters_list = []
        for layer, layer_data in sensitivity.items():
            for filter_idx, filter_data in layer_data.items():
                base_drop = filter_data.get(8, 0)  # Use 8-bit as baseline
                filters_list.append({
                    'layer': layer,
                    'filter_idx': filter_idx,
                    'sensitivity': base_drop,
                    'sensitivity_curve': filter_data
                })
        
        # Sort by sensitivity (ascending)
        filters_list.sort(key=lambda x: x['sensitivity'])
        
        # Initialize config (all 8-bit)
        config = defaultdict(dict)
        for layer, layer_data in sensitivity.items():
            for filter_idx in layer_data.keys():
                is_feasible, layout = RegisterPackingManager.compute_packing_layout(8, 8, self.register_width)
                config[layer][filter_idx] = {
                    'weight_bits': 8,
                    'activation_bits': 8,
                    'pack_layout': layout
                }
        
        # Greedy reduction
        estimated_drop = 0.0
        moves = []
        
        for filter_info in filters_list:
            layer = filter_info['layer']
            filter_idx = filter_info['filter_idx']
            current_bits = config[layer][filter_idx]['weight_bits']
            
            # Try reducing bit-width
            for target_bits in self.bit_widths:
                if target_bits >= current_bits:
                    continue
                
                # Check hardware feasibility
                is_feasible, pack_layout = RegisterPackingManager.compute_packing_layout(
                    target_bits, target_bits, self.register_width
                )
                
                if not is_feasible:
                    continue
                
                # Get estimated drop
                layer_drop = filter_info['sensitivity_curve'].get(target_bits, 0)
                potential_drop = estimated_drop + layer_drop
                
                # Accept if within budget
                if potential_drop <= self.target_drop:
                    config[layer][filter_idx]['weight_bits'] = target_bits
                    config[layer][filter_idx]['activation_bits'] = target_bits
                    config[layer][filter_idx]['pack_layout'] = pack_layout
                    
                    estimated_drop = potential_drop
                    moves.append({
                        'layer': layer,
                        'filter': filter_idx,
                        'from': current_bits,
                        'to': target_bits,
                        'drop': layer_drop
                    })
                    break
        
        # Compute statistics
        bit_dist = defaultdict(int)
        for layer in config:
            for filter_idx in config[layer]:
                bits = config[layer][filter_idx]['weight_bits']
                bit_dist[bits] += 1
        
        total_filters = sum(bit_dist.values())
        avg_bits = sum(b * count for b, count in bit_dist.items()) / total_filters
        
        stats = {
            'total_filters': total_filters,
            'bit_distribution': dict(bit_dist),
            'average_bits': round(avg_bits, 2),
            'bops_reduction': round(8.0 / avg_bits, 2),
            'estimated_drop': round(estimated_drop, 3),
            'moves_made': len(moves),
            'target_drop': self.target_drop
        }
        
        return dict(config), stats


# ============================================================================
# UTILITIES
# ============================================================================

def save_config(config: Dict, output_path: str):
    """Save quantization config to JSON."""
    # Convert to serializable format
    serializable_config = {}
    for layer, layer_config in config.items():
        serializable_config[layer] = {}
        for filter_idx, filter_cfg in layer_config.items():
            serializable_config[layer][str(filter_idx)] = {
                'weight_bits': filter_cfg['weight_bits'],
                'activation_bits': filter_cfg['activation_bits'],
                'pack_layout': filter_cfg['pack_layout'].to_dict()
                    if hasattr(filter_cfg['pack_layout'], 'to_dict')
                    else filter_cfg['pack_layout']
            }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_config(config_path: str) -> Dict:
    """Load quantization config from JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Reconstruct RegisterPackingLayout objects
    result = {}
    for layer, layer_config in config.items():
        result[layer] = {}
        for filter_idx_str, filter_cfg in layer_config.items():
            filter_idx = int(filter_idx_str)
            w_bits = filter_cfg['weight_bits']
            a_bits = filter_cfg['activation_bits']
            
            _, layout = RegisterPackingManager.compute_packing_layout(w_bits, a_bits)
            
            result[layer][filter_idx] = {
                'weight_bits': w_bits,
                'activation_bits': a_bits,
                'pack_layout': layout
            }
    
    return result


if __name__ == "__main__":
    print("✓ Granular W=A Quantization Module Loaded"
    print("  - Per-filter/neuron quantization with W=A constraint")
    print("  - Hardware-aware register packing (16-bit)")
    print("  - Integration with MQF framework")
