"""Fault injection utilities for reliability testing."""

import struct
import random
import copy
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from multiprocessing import Pool
import numpy as np


class FaultInjector:
    """Handles fault injection into neural network models."""
    
    def __init__(self, fault_type: str = "bit_flip"):
        """
        Initialize fault injector.
        
        Args:
            fault_type: Type of fault to inject (only "bit_flip" supported for optimal configuration)
        """
        if fault_type != "bit_flip":
            raise ValueError("Only 'bit_flip' fault type is supported for optimal configuration")
        self.fault_type = fault_type
        
    def inject_single_bit_flip(self, model: nn.Module, layer_name: str) -> bool:
        """
        Inject a single, safe bit-flip fault into a specific layer, retrying if
        the fault results in a non-finite value (NaN or Inf).
        
        Args:
            model: The neural network model
            layer_name: Name of the layer to inject fault into
            
        Returns:
            True if fault was successfully injected, False otherwise
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                target_weight_tensor = model.state_dict()[layer_name]
                if target_weight_tensor.numel() == 0:
                    return False
                    
                # Select random weight and bit position
                weight_idx = random.randint(0, target_weight_tensor.numel() - 1)
                # Exclude problematic exponent bits to prevent NaN/Inf
                # IEEE 754: bit 31=sign, bits 30-23=exponent, bits 22-0=mantissa
                # Only flip mantissa bits (0-22) and sign bit (31) to avoid extreme values
                safe_bit_positions = list(range(0, 23)) + [31]  # Mantissa + sign bit
                bit_position = random.choice(safe_bit_positions)
                
                with torch.no_grad():
                    weight_1d = target_weight_tensor.view(-1)
                    value_to_flip = weight_1d[weight_idx].item()
                    
                    # Convert float to binary representation
                    binary_val = struct.pack('!f', value_to_flip)
                    int_val = struct.unpack('!I', binary_val)[0]
                    
                    # Flip the bit
                    flipped_int_val = int_val ^ (1 << bit_position)
                    
                    # Convert back to float
                    flipped_binary = struct.pack('!I', flipped_int_val)
                    flipped_float_val = struct.unpack('!f', flipped_binary)[0]

                    # --- SAFETY CHECK ---
                    # If the flipped value is NaN or Inf, this attempt is invalid.
                    if not np.isfinite(flipped_float_val):
                        # print(f"  (Fault injection attempt {attempt+1} for {layer_name} resulted in non-finite value, retrying...)")
                        continue # Retry with a new random weight/bit

                    # Update the weight
                    weight_1d[weight_idx] = flipped_float_val
                    
                return True # Successful, finite fault injected
                
            except Exception as e:
                print(f"Error injecting fault into {layer_name} on attempt {attempt+1}: {e}")
                # Fall through to retry
        
        print(f"Warning: Failed to inject a valid, finite fault into {layer_name} after {max_attempts} attempts.")
        return False
    
    def inject_stuck_at_fault(self, model: nn.Module, layer_name: str, stuck_value: float = 0.0) -> bool:
        """
        Inject a stuck-at fault (set weight to specific value).
        
        Args:
            model: The neural network model
            layer_name: Name of the layer to inject fault into
            stuck_value: Value to set the weight to (0.0 for stuck-at-0)
            
        Returns:
            True if fault was successfully injected, False otherwise
        """
        try:
            target_weight_tensor = model.state_dict()[layer_name]
            if target_weight_tensor.numel() == 0:
                return False
                
            weight_idx = random.randint(0, target_weight_tensor.numel() - 1)
            
            with torch.no_grad():
                weight_1d = target_weight_tensor.view(-1)
                weight_1d[weight_idx] = stuck_value
                
            return True
            
        except Exception as e:
            print(f"Error injecting stuck-at fault into {layer_name}: {e}")
            return False
    
    def inject_faults_inplace(self, model: nn.Module, num_faults: int, 
                             target_layers: List[str]) -> int:
        """
        Inject multiple faults into a model in-place.
        
        Args:
            model: The neural network model to inject faults into
            num_faults: Number of faults to inject
            target_layers: List of layer names to target for fault injection
            
        Returns:
            Number of faults successfully injected
        """
        if not target_layers:
            return 0
            
        successful_injections = 0
        
        for _ in range(num_faults):
            chosen_layer = random.choice(target_layers)
            
            # Only bit-flip faults supported for optimal configuration
            success = self.inject_single_bit_flip(model, chosen_layer)
                
            if success:
                successful_injections += 1
                
        return successful_injections
    
    def create_faulty_model(self, original_model: nn.Module, num_faults: int, 
                           target_layers: List[str]) -> nn.Module:
        """
        Create a new model with faults injected.
        
        Args:
            original_model: The original model to copy and inject faults into
            num_faults: Number of faults to inject
            target_layers: List of layer names to target
            
        Returns:
            New model with faults injected
        """
        faulty_model = copy.deepcopy(original_model)
        self.inject_faults_inplace(faulty_model, num_faults, target_layers)
        return faulty_model


def get_weight_layer_names(model: nn.Module) -> List[str]:
    """Get list of weight layer names suitable for fault injection."""
    return [
        name for name, param in model.named_parameters() 
        if param.requires_grad and 'weight' in name
    ]


def inject_faults_parallel_worker(args):
    """
    Worker function for parallel fault injection.
    
    Args:
        args: Tuple containing (model_state_dict, num_faults, target_layers, fault_type, rep_id)
        
    Returns:
        Dictionary with results of the fault injection
    """
    model_state_dict, num_faults, target_layers, fault_type, rep_id = args
    
    # This function would be used in a multiprocessing context
    # Note: Due to the complexity of serializing models across processes,
    # parallel fault injection is typically done at a higher level
    return {
        'rep_id': rep_id,
        'num_faults': num_faults,
        'target_layers': len(target_layers)
    }


class FaultInjectionCampaign:
    """Manages a campaign of fault injection experiments."""
    
    def __init__(self, fault_types: List[str] = None, enable_parallel: bool = False):
        """
        Initialize fault injection campaign.
        
        Args:
            fault_types: List of fault types to test
            enable_parallel: Whether to enable parallel processing
        """
        # Validate fault types - only bit_flip supported
        if fault_types:
            for fault_type in fault_types:
                if fault_type != "bit_flip":
                    raise ValueError(f"Unsupported fault type: {fault_type}. Only 'bit_flip' is supported.")
        self.fault_types = ["bit_flip"]  # Force only bit_flip for optimal configuration
        self.enable_parallel = enable_parallel
        self.results = []
    
    def run_campaign(self, model: nn.Module, fault_levels: List[int], 
                    repetitions: int, target_layers: List[str], 
                    evaluation_func) -> Dict:
        """
        Run a complete fault injection campaign.
        
        Args:
            model: The model to test
            fault_levels: List of fault counts to test
            repetitions: Number of repetitions per fault level
            target_layers: Layers to target for fault injection
            evaluation_func: Function to evaluate model performance
            
        Returns:
            Dictionary containing campaign results
        """
        campaign_results = {}
        
        for fault_type in self.fault_types:
            fault_injector = FaultInjector(fault_type)
            fault_type_results = {}
            
            for num_faults in fault_levels:
                fault_level_results = []
                
                print(f"Testing {fault_type} with {num_faults} faults...")
                
                for rep in range(repetitions):
                    # Create faulty model
                    faulty_model = fault_injector.create_faulty_model(
                        model, num_faults, target_layers
                    )
                    
                    # Evaluate performance
                    performance = evaluation_func(faulty_model)
                    
                    fault_level_results.append({
                        'repetition': rep,
                        'num_faults': num_faults,
                        'performance': performance,
                        'fault_type': fault_type
                    })
                    
                    # Clean up
                    del faulty_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate statistics
                performances = [r['performance'] for r in fault_level_results]
                fault_type_results[num_faults] = {
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'min': np.min(performances),
                    'max': np.max(performances),
                    'raw_results': fault_level_results
                }
                
                print(f"  {num_faults} faults: {np.mean(performances):.2f}% ± {np.std(performances):.2f}%")
            
            campaign_results[fault_type] = fault_type_results
        
        return campaign_results
    
    def get_summary_statistics(self, results: Dict) -> Dict:
        """Get summary statistics from campaign results."""
        summary = {}
        
        for fault_type, fault_type_results in results.items():
            summary[fault_type] = {}
            for num_faults, stats in fault_type_results.items():
                summary[fault_type][num_faults] = {
                    'mean_performance': stats['mean'],
                    'std_performance': stats['std'],
                    'performance_drop': 100.0 - stats['mean']  # Assuming baseline is 100%
                }
        
        return summary