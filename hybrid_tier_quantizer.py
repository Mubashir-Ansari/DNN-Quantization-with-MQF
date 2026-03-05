"""
HYBRID TIER QUANTIZATION: Production Implementation
Mixed-Precision Quantization with Tier-Based Selective Granular Optimization

Author: Senior AI Engineer
Date: March 5, 2026
Target: AlexNet POC with Hardware-Aware Register Packing
"""

import torch
import torch.nn as nn
import numpy as np
import json
import csv
import os
import sys
import math
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import logging
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PART 1: DATA STRUCTURES & TYPE HINTS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class TierAssignment:
    """Classification of layers into optimization tiers"""
    layer_name: str
    tier: int  # 1=sensitive, 2=medium, 3=insensitive
    layer_drop_4bit: float  # Accuracy drop at 4-bit (%)
    layer_drop_2bit: float  # Accuracy drop at 2-bit (%)
    rationale: str

@dataclass
class GranularFilterConfig:
    """Per-filter quantization configuration"""
    layer_name: str
    filter_idx: int
    weight_bits: int
    activation_bits: int  # Must equal weight_bits (W=A constraint)
    magnitude: float  # Filter weight magnitude (used for heuristic)
    drop_estimate: float  # Estimated drop if set to weight_bits

@dataclass
class RegisterPackingInfo:
    """Hardware register packing metadata"""
    layer_name: str
    filter_idx: Optional[int]
    bit_width: int
    registers_needed: int
    register_utilization: float  # Percentage
    carry_bits_allocated: int

class HybridQuantizer:
    """Main optimization framework"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', 
                 register_width: int = 16,
                 tier1_threshold: float = 5.0,
                 tier2_threshold: float = 2.0):
        self.model = model
        self.device = device
        self.register_width = register_width
        
        # Results storage
        self.layer_sensitivity: Dict = {}
        self.tier_assignments: List[TierAssignment] = []
        self.final_config: Dict = {}
        self.packing_info: Dict = {}
        self.metrics: Dict = {}
        
        # Robust State Backup (Deep Copy)
        # We need this because Profiling modifies weight.data in-place
        self.original_state_dict = {
            n: p.data.clone().cpu() for n, p in model.named_parameters()
        }
        
        # Thresholds (Tuned by user config)
        self.thresholds = {
            'tier1_threshold': tier1_threshold,    # Drop_4bit > threshold1 → Tier 1 (8-bit)
            'tier2_threshold': tier2_threshold,    # Drop_4bit > threshold2 → Tier 2 (4-bit)
            'filter_robust_percentile': 0.70,      # Bottom 70% = candidate for 2-bit
            'filter_2bit_threshold': tier2_threshold / 2, # If filter-drop <= threshold/2, use 2-bit
            'filter_4bit_threshold': tier2_threshold * 1.5, # If filter-drop <= threshold*1.5, use 4-bit
            'carry_bits_2bit': 3,
            'carry_bits_4bit': 4,
        }
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: FAST LAYER-WISE PROFILING
    # ══════════════════════════════════════════════════════════════════════════
    
    def stage1_fast_layer_profiling(self, dataloader, samples=256):
        """
        STAGE 1: Fast layer-wise sensitivity profiling
        
        Reuses existing MQF infrastructure.
        Time: ~1 hour on GPU (256 samples per layer, 3 bit-widths)
        """
        
        print("\n" + "="*80)
        print("[STAGE 1] FAST LAYER-WISE PROFILING (Tier Detection)")
        print("="*80)
        
        self.model.eval()
        bit_widths = [2, 4, 8]
        sensitivity = {}
        
        # Get all quantizable layers
        layers = self._get_quantizable_layers()
        
        # Measure baseline accuracy
        baseline_acc = self._evaluate_accuracy(dataloader, samples=samples)
        self.metrics['baseline_accuracy'] = baseline_acc
        print(f"\nBaseline Accuracy (8-bit Baseline): {baseline_acc:.2f}%")
        
        # Profile each layer
        for layer_name in tqdm(layers, desc="Layer Profiling"):
            layer_sensitivity = {}
            
            for bits in bit_widths:
                # Apply layer-wise quantization
                self._apply_layer_quantization(layer_name, bits)
                
                # Measure accuracy drop
                acc = self._evaluate_accuracy(dataloader, samples=samples)
                drop = baseline_acc - acc
                layer_sensitivity[bits] = drop
                
                # Restore layer
                self._restore_layer(layer_name)
            
            sensitivity[layer_name] = layer_sensitivity
        
        self.layer_sensitivity = sensitivity
        
        # Summary
        print(f"\n✓ Profiled {len(layers)} layers")
        print(f"  Sample size: {samples} images per test")
        print(f"  Bit-widths tested: {bit_widths}")
        
        return sensitivity
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: TIER CATEGORIZATION & CONFIG GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def stage2_categorize_and_config(self):
        """
        STAGE 2: Categorize layers into tiers
        
        Tier 1 (Sensitive, ~30%): Keep 8-bit uniform
        Tier 2 (Medium, ~45%): Keep 4-bit uniform
        Tier 3 (Insensitive, ~25%): Candidate for granular
        """
        
        print("\n" + "="*80)
        print("[STAGE 2] TIER CATEGORIZATION & CONFIG GENERATION")
        print("="*80)
        
        if not self.layer_sensitivity:
            raise ValueError("Run stage1_fast_layer_profiling first")
        
        # Sort layers by 4-bit drop (sensitivity metric)
        layers_sorted = sorted(
            self.layer_sensitivity.items(),
            key=lambda x: x[1][4],  # Sort by 4-bit drop
            reverse=True  # Highest first = most sensitive
        )
        
        # Categorize by Threshold instead of fixed percentages
        tier1_threshold = self.thresholds['tier1_threshold']
        tier2_threshold = self.thresholds['tier2_threshold']
        
        tiers = {
            'tier1_sensitive': [],
            'tier2_medium': [],
            'tier3_insensitive': []
        }
        
        for layer_name, layer_drop in layers_sorted:
            drop_4 = layer_drop[4]
            
            if drop_4 > tier1_threshold:
                tiers['tier1_sensitive'].append(layer_name)
            elif drop_4 > tier2_threshold:
                tiers['tier2_medium'].append(layer_name)
            else:
                tiers['tier3_insensitive'].append(layer_name)

        config = {}
        total_layers = len(layers_sorted)
        
        # TIER 1: Sensitive (Keep 8-bit)
        for layer_name in tiers['tier1_sensitive']:
            layer_drop = self.layer_sensitivity[layer_name]
            config[layer_name] = 8
            assignment = TierAssignment(
                layer_name=layer_name, tier=1,
                layer_drop_4bit=layer_drop[4], layer_drop_2bit=layer_drop[2],
                rationale=f"Sensitive (drop {layer_drop[4]:.2f}% > {tier1_threshold}%)"
            )
            self.tier_assignments.append(assignment)
            
        # TIER 2: Medium (4-bit default)
        for layer_name in tiers['tier2_medium']:
            layer_drop = self.layer_sensitivity[layer_name]
            config[layer_name] = 4
            assignment = TierAssignment(
                layer_name=layer_name, tier=2,
                layer_drop_4bit=layer_drop[4], layer_drop_2bit=layer_drop[2],
                rationale=f"Medium (drop {layer_drop[4]:.2f}% > {tier2_threshold}%)"
            )
            self.tier_assignments.append(assignment)

        # TIER 3: Insensitive (Granular candidate)
        for layer_name in tiers['tier3_insensitive']:
            layer_drop = self.layer_sensitivity[layer_name]
            config[layer_name] = 4 # Placeholder for Refinement
            assignment = TierAssignment(
                layer_name=layer_name, tier=3,
                layer_drop_4bit=layer_drop[4], layer_drop_2bit=layer_drop[2],
                rationale=f"Insensitive (drop {layer_drop[4]:.2f}% <= {tier2_threshold}%)"
            )
            self.tier_assignments.append(assignment)
        
        # Print summary
        print(f"\n[TIER BREAKDOWN]")
        print(f"Tier 1 (Sensitive):     {len(tiers['tier1_sensitive']):2d} layers → 8-bit uniform")
        for layer in tiers['tier1_sensitive']:
            drop = self.layer_sensitivity[layer][4]
            print(f"  ├─ {layer:<20} (drop@4b: {drop:.3f}%)")
        
        print(f"\nTier 2 (Medium):        {len(tiers['tier2_medium']):2d} layers → layer-wise 4-bit")
        for layer in tiers['tier2_medium'][:3]:
            drop = self.layer_sensitivity[layer][4]
            print(f"  ├─ {layer:<20} (drop@4b: {drop:.3f}%)")
        if len(tiers['tier2_medium']) > 3:
            print(f"  └─ ... +{len(tiers['tier2_medium'])-3} more")
        
        print(f"\nTier 3 (Insensitive):   {len(tiers['tier3_insensitive']):2d} layers → GRANULAR refinement")
        for layer in tiers['tier3_insensitive']:
            drop = self.layer_sensitivity[layer][4]
            print(f"  ├─ {layer:<20} (drop@4b: {drop:.3f}%)")
        
        self.final_config = config
        self.tiers = tiers
        
        return config, tiers
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: SELECTIVE GRANULAR REFINEMENT (TIER 3 ONLY)
    # ══════════════════════════════════════════════════════════════════════════
    
    def stage3_selective_granular_refinement(self, dataloader, samples=256):
        """
        STAGE 3: Granular refinement for Tier 3 (insensitive) layers only
        
        Strategy:
        - Profile top-10 filters per layer (fast test)
        - Use heuristic: filter magnitude → robustness
        - Assign 2-bit to robust, keep 4-bit for borderline
        
        Time: ~2 hours (vs 8+ for all layers)
        """
        
        print("\n" + "="*80)
        print("[STAGE 3] SELECTIVE GRANULAR REFINEMENT (Tier 3 Only)")
        print("="*80)
        
        if not self.final_config or not hasattr(self, 'tiers'):
            raise ValueError("Run stage2_categorize_and_config first")
        
        self.model.eval()
        granular_configs = {}
        
        # Baseline for comparison
        baseline_acc = self._evaluate_accuracy(dataloader, samples=samples)
        
        tier3_layers = self.tiers['tier3_insensitive']
        
        for layer_name in tqdm(tier3_layers, desc="Granular Refinement"):
            module = self._get_module(layer_name)
            num_filters = self._get_num_filters(module)
            
            # Strategy: Heuristic based on magnitude
            filter_magnitudes = []
            for f_idx in range(num_filters):
                w = module.weight.data[f_idx]
                magnitude = torch.norm(w).item()
                filter_magnitudes.append((f_idx, magnitude))
            
            # Sort by magnitude (small = more robust, typically easier to quantize)
            filter_magnitudes.sort(key=lambda x: x[1])
            
            # Assign bits based on magnitude percentile
            robust_threshold = int(num_filters * self.thresholds['filter_robust_percentile'])
            filter_bits = [8] * num_filters  # Default
            
            # Stage 3 MQF Logic: 2-bit, 4-bit, or 8-bit
            for rank, (f_idx, mag) in enumerate(filter_magnitudes):
                if rank < robust_threshold:
                    # Robust filter → start with 2-bit
                    filter_bits[f_idx] = 2
                else:
                    # Medium sensitivity → start with 4-bit
                    filter_bits[f_idx] = 4
            
            # Optional: Verify top-5 robust ones with quick profiling
            top_robust_indices = [f[0] for f in filter_magnitudes[:5]]
            
            for f_idx in top_robust_indices:
                # Profile filter to see if 2-bit or 4-bit is actually safe
                self._apply_filter_quantization(layer_name, f_idx, bits=2)
                acc_2b = self._evaluate_accuracy(dataloader, samples=64)
                drop_2b = baseline_acc - acc_2b
                
                if drop_2b > self.thresholds['filter_2bit_threshold']:
                    # 2-bit failed → try 4-bit
                    self._apply_filter_quantization(layer_name, f_idx, bits=4)
                    acc_4b = self._evaluate_accuracy(dataloader, samples=64)
                    drop_4b = baseline_acc - acc_4b
                    
                    if drop_4b > self.thresholds['filter_4bit_threshold']:
                        # 4-bit also failed! → Keep 8-bit for this specific filter
                        filter_bits[f_idx] = 8
                    else:
                        filter_bits[f_idx] = 4
                else:
                    filter_bits[f_idx] = 2
                
                self._restore_filter(layer_name, f_idx)
            
            granular_configs[layer_name] = filter_bits
            
            # Statistics
            count_2b = sum(1 for b in filter_bits if b == 2)
            count_4b = sum(1 for b in filter_bits if b == 4)
            count_8b = sum(1 for b in filter_bits if b == 8)
            
            print(f"INFO: {layer_name}: 2-bit={count_2b}, 4-bit={count_4b}, 8-bit={count_8b}")
        
        # Merge granular config into main config
        for layer_name, filter_bits in granular_configs.items():
            self.final_config[layer_name] = filter_bits
        
        print(f"\n✓ Granular refinement complete for {len(tier3_layers)} Tier 3 layers")
        
        return granular_configs
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: HARDWARE-AWARE REGISTER PACKING
    # ══════════════════════════════════════════════════════════════════════════
    
    def stage4_record_packing(self):
        """
        STAGE 4: Analyze register packing efficiency
        
        Calculates:
        - Registers needed per layer
        - Packing efficiency (utilization %)
        - Carry space allocated
        """
        
        print("\n" + "="*80)
        print("[STAGE 4] HARDWARE REGISTER PACKING ANALYSIS")
        print("="*80)
        
        packing_info = {}
        total_registers_mqf = 0
        total_registers_8bit_baseline = 0
        total_params = 0
        total_bits_mqf = 0
        total_bits_8bit = 0
        total_efficiency = 0
        layer_count = 0
        
        for layer_name, layer_config in self.final_config.items():
            module = self._get_module(layer_name)
            w = module.weight.data
            num_filters = w.shape[0]
            params_per_filter = w[0].numel()
            
            # Calculate Baseline (8-bit uniform) for comparison
            # In a 16-bit register, 8-bit takes 2 slots
            baseline_regs_layer = num_filters * math.ceil(params_per_filter / 2)
            total_registers_8bit_baseline += baseline_regs_layer

            if isinstance(layer_config, int):
                # Uniform bit-width (Tier 1 or 2)
                bit_width = layer_config
                
                # Calculate registers for current config
                registers_needed = num_filters * math.ceil(
                    params_per_filter / (self.register_width / bit_width)
                )
                
                # Calculate efficiency
                actual_bits_used = params_per_filter * bit_width
                utilization = (actual_bits_used / 
                              (registers_needed / num_filters * self.register_width)) * 100
                
                # Allocate carry bits
                if bit_width == 2:
                    carry_bits = self.thresholds['carry_bits_2bit']
                elif bit_width == 4:
                    carry_bits = self.thresholds['carry_bits_4bit']
                else:
                    carry_bits = 0
                
                total_registers_mqf += registers_needed
                total_params += w.numel()
                total_bits_mqf += (w.numel() * bit_width)
                total_bits_8bit += (w.numel() * 8)
                total_efficiency += utilization
                layer_count += 1
                
                packing = {
                    layer_name: {
                        'type': 'uniform',
                        'bit_width': bit_width,
                        'registers': registers_needed,
                        'utilization_percent': utilization,
                        'carry_bits': carry_bits
                    }
                }
                
            else:
                # Mixed bit-width (Tier 3, granular)
                bit_widths = layer_config
                layer_registers = 0
                layer_utilization = 0
                
                for f_idx, bits in enumerate(bit_widths):
                    regs = math.ceil(params_per_filter / (self.register_width / bits))
                    layer_registers += regs
                    
                    actual_bits = params_per_filter * bits
                    util = (actual_bits / (regs * self.register_width)) * 100
                    layer_utilization += util
                
                avg_utilization = layer_utilization / len(bit_widths)
                total_registers_mqf += layer_registers
                total_params += w.numel()
                total_bits_mqf += sum(b * (w.numel() // len(bit_widths)) for b in bit_widths)
                total_bits_8bit += (w.numel() * 8)
                total_efficiency += avg_utilization
                layer_count += 1
                
                bit_dist = {2: 0, 4: 0, 8: 0}
                for b in bit_widths:
                    bit_dist[b] += 1
                
                packing[layer_name] = {
                    'type': 'granular',
                    'bit_distribution': bit_dist,
                    'registers': layer_registers,
                    'utilization_percent': avg_utilization,
                    'carry_bits': 4  # Conservative for mixed
                }
            
            packing_info.update(packing)
        
        self.packing_info = packing_info
        self.metrics['registers_baseline'] = total_registers_8bit_baseline
        self.metrics['registers_mqf'] = total_registers_mqf
        self.metrics['register_savings_percent'] = (1 - total_registers_mqf/total_registers_8bit_baseline) * 100
        self.metrics['average_bits'] = total_bits_mqf / total_params if total_params > 0 else 8
        
        # Print summary
        avg_efficiency = total_efficiency / layer_count
        
        print(f"\n[PACKING SUMMARY]")
        print(f"Total Parameters:          {total_params:,}")
        print(f"Total 16-bit Registers:    {total_registers_mqf:,}")
        print(f"Avg Registers/Param:       {total_registers_mqf / total_params:.3f}")
        print(f"Overall Register Utilization: {avg_efficiency:.1f}%")
        
        return packing_info
    
    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5: VALIDATION & QAT
    # ══════════════════════════════════════════════════════════════════════════
    
    def stage5_validation_and_qat(self, model_baseline, dataloader, 
                                  device, samples=1000,
                                  qat_threshold=2.0, target_drop=3.0):
        """
        STAGE 5: PTQ validation and optional QAT recovery
        """
        
        print("\n" + "="*80)
        print("[STAGE 5] VALIDATION & QAT RECOVERY")
        print("="*80)
        
        # Use previously saved baseline instead of re-evaluating potentially modified model
        baseline_acc = self.metrics.get('baseline_accuracy', 0.0)
        if baseline_acc == 0:
             baseline_acc = self._evaluate_accuracy_model(model_baseline, dataloader, 
                                                         device=device, samples=samples)
        print(f"\nBaseline Accuracy (8-bit Baseline): {baseline_acc:.2f}%")
        
        # Apply PTQ with config
        model_quantized = self._apply_tier_quantization(self.model, self.final_config)
        model_quantized.to(device)
        model_quantized.eval()
        
        # Measure PTQ accuracy
        ptq_acc = self._evaluate_accuracy_model(model_quantized, dataloader,
                                               device=device, samples=samples)
        ptq_drop = baseline_acc - ptq_acc
        
        print(f"PTQ Accuracy:             {ptq_acc:.2f}%")
        print(f"Accuracy Drop (PTQ):      {ptq_drop:.3f}%")
        
        # Decision
        final_acc = ptq_acc
        used_qat = False
        
        if ptq_drop <= qat_threshold:
            print(f"\n✓ PTQ PASSED (drop {ptq_drop:.3f}% ≤ {qat_threshold}%)")
        else:
            print(f"\n⚠ PTQ MARGINAL (drop {ptq_drop:.3f}% > {qat_threshold}%)")
            print("→ QAT recovery recommended but skipped for research")
            # In production, would run QAT here
        
        metrics = {
            'baseline_accuracy': baseline_acc,
            'ptq_accuracy': ptq_acc,
            'final_accuracy': final_acc,
            'accuracy_drop_percent': baseline_acc - final_acc,
            'ptq_drop_percent': ptq_drop,
            'used_qat': used_qat,
            'target_drop_percent': target_drop
        }
        self.metrics.update(metrics)
        
        return metrics
    
    # ══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _get_quantizable_layers(self) -> List[str]:
        """Get list of Conv2d and Linear layers"""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append(name)
        return layers
    
    def _get_module(self, layer_name: str) -> nn.Module:
        """Get module by name"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found")
    
    def _get_num_filters(self, module: nn.Module) -> int:
        """Get output channels/features"""
        if isinstance(module, nn.Conv2d):
            return module.out_channels
        elif isinstance(module, nn.Linear):
            return module.out_features
        return 1
    
    def _evaluate_accuracy(self, dataloader, samples=256, device=None) -> float:
        """Evaluate current model accuracy"""
        if device is None:
            device = self.device
        
        self.model.to(device)
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                if i * dataloader.batch_size >= samples:
                    break
                
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def _evaluate_accuracy_model(self, model, dataloader, device='cuda', 
                                samples=1000) -> float:
        """Evaluate specific model accuracy"""
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                if i * dataloader.batch_size >= samples:
                    break
                
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def _apply_layer_quantization(self, layer_name: str, bits: int):
        """Quantize a layer uniformly"""
        module = self._get_module(layer_name)
        w = module.weight.data
        
        # Symmetric quantization
        qmax = (1 << (bits - 1)) - 1
        w_abs_max = torch.max(torch.abs(w))
        scale = w_abs_max / qmax if w_abs_max > 0 else 1.0
        
        w_int = torch.round(w / scale).clamp(-qmax - 1, qmax)
        w_dequant = w_int * scale
        
        module.weight.data = w_dequant
    
    def _apply_filter_quantization(self, layer_name: str, filter_idx: int, 
                                  bits: int):
        """Quantize a specific filter"""
        module = self._get_module(layer_name)
        w = module.weight.data
        
        qmax = (1 << (bits - 1)) - 1
        w_f = w[filter_idx]
        w_f_max = torch.max(torch.abs(w_f))
        scale = w_f_max / qmax if w_f_max > 0 else 1.0
        
        w_f_int = torch.round(w_f / scale).clamp(-qmax - 1, qmax)
        w_f_dequant = w_f_int * scale
        
        w[filter_idx] = w_f_dequant
    
    def _restore_layer(self, layer_name: str):
        """Reload original weights from deep copy"""
        module = self._get_module(layer_name)
        orig_data = self.original_state_dict[f"{layer_name}.weight"].to(self.device)
        module.weight.data = orig_data.clone()
    
    def _restore_filter(self, layer_name: str, filter_idx: int):
        """Restore specific filter"""
        module = self._get_module(layer_name)
        original_w = self.model.state_dict()[f"{layer_name}.weight"]
        module.weight.data[filter_idx] = original_w[filter_idx].clone()
    
    def _apply_tier_quantization(self, model, config):
        """Apply quantization according to config"""
        for layer_name, layer_config in config.items():
            module = self._get_module(layer_name)
            w = module.weight.data
            
            if isinstance(layer_config, int):
                # Uniform quantization
                bits = layer_config
                qmax = (1 << (bits - 1)) - 1
                w_abs_max = torch.max(torch.abs(w))
                scale = w_abs_max / qmax if w_abs_max > 0 else 1.0
                w_int = torch.round(w / scale).clamp(-qmax - 1, qmax)
                module.weight.data = w_int * scale
            else:
                # Per-filter quantization
                for f_idx, bits in enumerate(layer_config):
                    w_f = w[f_idx]
                    qmax = (1 << (bits - 1)) - 1
                    w_f_max = torch.max(torch.abs(w_f))
                    scale = w_f_max / qmax if w_f_max > 0 else 1.0
                    w_f_int = torch.round(w_f / scale).clamp(-qmax - 1, qmax)
                    w[f_idx] = w_f_int * scale
        
        return model
    
    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS & REPORTING
    # ══════════════════════════════════════════════════════════════════════════
    
    def save_results(self, output_dir: str = "results"):
        """Save all results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(output_dir, "hybrid_config.json")
        with open(config_path, 'w') as f:
            # Convert config for JSON serialization
            json_config = {}
            for layer, cfg in self.final_config.items():
                if isinstance(cfg, int):
                    json_config[layer] = cfg
                else:
                    json_config[layer] = [int(b) for b in cfg]
            json.dump(json_config, f, indent=2)
        
        # Save tier assignments
        tiers_path = os.path.join(output_dir, "tier_assignments.csv")
        with open(tiers_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'layer_name', 'tier', 'layer_drop_4bit', 'layer_drop_2bit', 'rationale'
            ])
            writer.writeheader()
            for assignment in self.tier_assignments:
                writer.writerow(asdict(assignment))
        
        # Save detailed granular configs (for Tier 3 filter-level detail)
        granular_path = os.path.join(output_dir, "granular_filter_configs.json")
        granular_detail = {}
        for assignment in self.tier_assignments:
            if assignment.tier == 3:  # Only Tier 3 has granular
                layer_name = assignment.layer_name
                if isinstance(self.final_config.get(layer_name), list):
                    cfg = self.final_config[layer_name]
                    granular_detail[layer_name] = {
                        'num_filters': len(cfg),
                        'filter_bits': [int(b) for b in cfg],
                        'bit_distribution': {
                            '2bit': sum(1 for b in cfg if b == 2),
                            '4bit': sum(1 for b in cfg if b == 4),
                            '8bit': sum(1 for b in cfg if b == 8),
                        },
                        'average_bits': round(sum(cfg) / len(cfg), 2),
                        'rationale': f"Tier 3 (Insensitive): Granular optimized. Drop@4b: {assignment.layer_drop_4bit:.3f}%"
                    }
        if granular_detail:
            with open(granular_path, 'w') as f:
                json.dump(granular_detail, f, indent=2)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\n✓ Results saved to {output_dir}/")
        print(f"  ├─ hybrid_config.json (Layer-wise quantization spec)")
        print(f"  ├─ granular_filter_configs.json (Filter-level detail for Tier 3)")
        print(f"  ├─ tier_assignments.csv (Layer categorization)")
        print(f"  └─ metrics.json (Accuracy & compression metrics)")

    def get_hardware_stats(self) -> Dict:
        """Calculate per-layer bit distribution and packing stats"""
        stats = {}
        for name, cfg in self.final_config.items():
            module = self._get_module(name)
            out_ch = self._get_num_filters(module)
            
            if isinstance(cfg, int):
                # Uniform
                bit_counts = {2: 0, 4: 0, 8: 0}
                bit_counts[cfg] = 100.0
                avg_pack = self.register_width / cfg
                status = "UNIFORM"
            else:
                # Granular
                bit_counts = {2: 0, 4: 0, 8: 0}
                for b in cfg:
                    bit_counts[b] = bit_counts.get(b, 0) + 1
                
                for b in bit_counts:
                    bit_counts[b] = (bit_counts[b] / out_ch) * 100
                
                # For granular, average packing is tricky but we follow hub logic
                avg_pack = self.packing_info[name]['registers'] / out_ch
                status = "GRANULAR"
                
            stats[name] = {
                '2b': bit_counts[2],
                '4b': bit_counts[4],
                '8b': bit_counts[8],
                'pack': avg_pack,
                'status': status
            }
        return stats
    def print_summary(self):
        """Print detailed summary in the user's requested format"""
        print("\n" + "="*80)
        print("HYBRID TIER QUANTIZATION - EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\n[RESULTS]")
        baseline_acc = self.metrics.get('baseline_accuracy', 0)
        final_acc = self.metrics.get('final_accuracy', 0)
        drop = self.metrics.get('accuracy_drop_percent', 0)
        print(f"Baseline Accuracy:        {baseline_acc:.2f}%")
        print(f"Final Accuracy:           {final_acc:.2f}%")
        print(f"Accuracy Drop:            {drop:.3f}%")
        
        print(f"\n[ALGO REPORT: REGISTER-MISMATCH ANALYSIS]")
        print(f"{'Layer':<15} | {'2b %':<6} | {'4b %':<6} | {'8b %':<6} | {'Pack (A)':<8} | {'Status'}")
        print("-" * 80)
        
        stats = self.get_hardware_stats()
        for name, s in stats.items():
            print(f"{name:<15} | {s['2b']:>5.1f}% | {s['4b']:>5.1f}% | {s['8b']:>5.1f}% | {s['pack']:<8.1f} | {s['status']}")
            
        print("-" * 80)
        
        tier1_count = len(self.tiers.get('tier1_sensitive', []))
        tier2_count = len(self.tiers.get('tier2_medium', []))
        tier3_count = len(self.tiers.get('tier3_insensitive', []))
        
        print(f"Tier 1 Layers:            {tier1_count} (8-bit uniform)")
        print(f"Tier 2 Layers:            {tier2_count} (4-bit layer-wise)")
        print(f"Tier 3 Layers:            {tier3_count} (granular 2-4-8bit)")
        
        print(f"\n[HARDWARE: 16-bit REGISTER PACKING]")
        reg_baseline = self.metrics.get('registers_baseline', 0)
        reg_mqf = self.metrics.get('registers_mqf', 0)
        savings = self.metrics.get('register_savings_percent', 0)
        
        print(f"Baseline Registers (8-bit): {reg_baseline:,}")
        print(f"MQF Registers (Mixed):      {reg_mqf:,}")
        print(f"Space Acquired (Savings):   {savings:.2f}%")
        
        avg_util = sum(p['utilization_percent'] 
                      for p in self.packing_info.values()) / len(self.packing_info) if self.packing_info else 0
        print(f"Register Utilization:       {avg_util:.1f}%")
        print(f"Strategy:                   MQF Carry-Aware Packing (Accordion Flow)")
        print("="*80 + "\n")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ END-TO-END EXECUTION FUNCTION                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_hybrid_tier_quantization(model_baseline, dataloader, device='cuda',
                                output_dir='results',
                                tier1_threshold=5.0,
                                tier2_threshold=2.0,
                                register_width=16):
    """
    Complete hybrid tier quantization pipeline
    
    Time estimate: ~4-5 hours end-to-end
    """
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" +  "  HYBRID TIER QUANTIZATION: FULL PIPELINE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # Initialize
    quantizer = HybridQuantizer(model_baseline, device=device, 
                               register_width=register_width,
                               tier1_threshold=tier1_threshold,
                               tier2_threshold=tier2_threshold)
    
    # Timing
    start_time = time.time()
    
    # Stage 1
    print(f"\n[{time.time()-start_time:.1f}s] Starting Stage 1...")
    quantizer.stage1_fast_layer_profiling(dataloader, samples=256)
    
    # Stage 2
    print(f"[{time.time()-start_time:.1f}s] Starting Stage 2...")
    config, tiers = quantizer.stage2_categorize_and_config()
    
    # Stage 3
    print(f"[{time.time()-start_time:.1f}s] Starting Stage 3...")
    quantizer.stage3_selective_granular_refinement(dataloader, samples=256)
    
    # Stage 4
    print(f"[{time.time()-start_time:.1f}s] Starting Stage 4...")
    quantizer.stage4_record_packing()
    
    # Stage 5
    print(f"[{time.time()-start_time:.1f}s] Starting Stage 5...")
    metrics = quantizer.stage5_validation_and_qat(model_baseline, dataloader, device)
    
    total_time = time.time() - start_time
    
    # Results
    quantizer.save_results(output_dir)
    quantizer.print_summary()
    
    print(f"\n✓ TOTAL EXECUTION TIME: {total_time/60:.1f} minutes")
    print(f"  (Expected: 4-5 hours on GPU)")
    
    return quantizer

if __name__ == "__main__":
    print("Hybrid Tier Quantization Module")
    print("Ready for integration with AlexNet POC")
