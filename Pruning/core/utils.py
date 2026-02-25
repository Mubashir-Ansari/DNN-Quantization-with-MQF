"""Common utility functions."""

import os
import glob
import timeit
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_nonzero_parameters(model: nn.Module) -> int:
    """Count the number of non-zero parameters in a model."""
    return sum(p.data.count_nonzero().item() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_sparsity(model: nn.Module) -> float:
    """Calculate the sparsity percentage of a model."""
    total_params = count_total_parameters(model)
    nonzero_params = count_nonzero_parameters(model)
    if total_params == 0:
        return 0.0
    return (total_params - nonzero_params) / total_params * 100.0


def measure_latency(model: nn.Module, dummy_input: torch.Tensor, 
                   num_warmup: int = 10, num_runs: int = 100) -> float:
    """Measure the inference latency of a model."""
    model.eval()
    model_device = next(model.parameters()).device
    dummy_input = dummy_input.to(model_device)
    
    with torch.no_grad():
        # Warmup runs
        for _ in range(num_warmup):
            _ = model(dummy_input)
        
        # Measurement runs
        start_time = timeit.default_timer()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = timeit.default_timer()
    
    return (end_time - start_time) / num_runs * 1000  # Return in milliseconds


def test_accuracy(model: nn.Module, device: torch.device, dataloader: DataLoader) -> float:
    """Test the accuracy of a model on a given dataset."""
    model.eval()
    model.to(device)
    correct, total, nan_batches = 0, 0, 0
    
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Check for corrupted outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                nan_batches += 1
                total += labels.size(0)
                continue  # Skip this batch

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if nan_batches > 0:
        print(f"  Warning: {nan_batches}/{len(dataloader)} batches had corrupted outputs (NaN/Inf)")
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    # Validate accuracy range - should never be negative or >100
    if accuracy < 0 or accuracy > 100:
        print(f"ERROR: Invalid accuracy computed: {accuracy}% (correct={correct}, total={total})")
        print(f"  This indicates a serious bug in the evaluation or data processing")
        # Clamp to valid range as emergency fallback
        accuracy = max(0, min(100, accuracy))
        
    return accuracy


def get_layer_score_files_map(score_dir_path: str, model_state_dict: Dict) -> Dict[str, str]:
    """Create a mapping from layer names to their sensitivity score files."""
    score_files_map = {}
    pattern = os.path.join(score_dir_path, "weight_sensitivity_scores_*.weight.csv")
    available_model_layers = {name: None for name in model_state_dict.keys() if 'weight' in name}
    
    for file_path in glob.glob(pattern):
        file_name = os.path.basename(file_path)
        try:
            # Extract layer name from files like: weight_sensitivity_scores_features.0.weight.csv
            layer_name_candidate = file_name.split("scores_")[1].replace(".csv", "")
            if layer_name_candidate in available_model_layers:
                score_files_map[layer_name_candidate] = file_path
        except IndexError:
            print(f"Warning: Could not parse layer name from score file: {file_name}")
    
    if not score_files_map:
        print(f"Warning: No score files found in {score_dir_path} matching model layers.")
    
    return score_files_map


def load_sensitivity_scores(score_file_path: str) -> Optional[np.ndarray]:
    """Load sensitivity scores from a CSV file."""
    try:
        scores_df = pd.read_csv(score_file_path)
        return scores_df['sensitivity_score'].astype(float).values
    except Exception as e:
        print(f"Error loading scores from {score_file_path}: {e}")
        return None


def get_prunable_layers(model: nn.Module, score_files_map: Dict[str, str]) -> List[str]:
    """Get list of prunable layer names that have sensitivity scores."""
    return sorted([
        name for name in model.state_dict().keys()
        if ('features' in name or 'classifier' in name) and 'weight' in name and name in score_files_map
    ])


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results_to_csv(results: List[Dict], filename: str) -> None:
    """Save results list to CSV file."""
    if results:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


def load_results_from_csv(filename: str) -> List[Dict]:
    """Load results from CSV file."""
    try:
        df = pd.read_csv(filename)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading results from {filename}: {e}")
        return []


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print a summary of the model."""
    total_params = count_total_parameters(model)
    nonzero_params = count_nonzero_parameters(model)
    sparsity = calculate_sparsity(model)
    
    print(f"\n{model_name} Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero parameters: {nonzero_params:,}")
    print(f"  Sparsity: {sparsity:.2f}%")


def validate_device(device_str: str) -> torch.device:
    """Validate and return torch device."""
    if device_str.lower() == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return device


def cleanup_memory() -> None:
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = timeit.default_timer()
    
    def update(self, step: int = 1) -> None:
        """Update progress by given steps."""
        self.current_step += step
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print current progress."""
        if self.total_steps > 0:
            percentage = (self.current_step / self.total_steps) * 100
            elapsed = timeit.default_timer() - self.start_time
            
            if self.current_step > 0:
                eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                print(f"{self.description}: {self.current_step}/{self.total_steps} "
                      f"({percentage:.1f}%) - ETA: {format_time(eta)}")
            else:
                print(f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = timeit.default_timer() - self.start_time
        print(f"{self.description} completed in {format_time(elapsed)}")


def get_available_methods() -> Dict[str, List[str]]:
    """Get dictionary of available pruning methods by category."""
    return {
        'classical': ['magnitude', 'random', 'layer_wise_magnitude'],
        'sota': ['wanda', 'snip', 'grasp', 'sparsegpt'],
        'structured': ['channel_pruning', 'filter_pruning'],
        'reliability_aware': ['ga_nsga2']
    }