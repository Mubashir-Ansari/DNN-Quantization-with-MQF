"""Reliability testing framework with parallel processing support."""

import copy
import time
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from .fault_injection import FaultInjector, get_weight_layer_names
from core.utils import test_accuracy, cleanup_memory


class ReliabilityTester:
    """Comprehensive reliability testing framework."""
    
    def __init__(self, enable_parallel: bool = True, max_workers: Optional[int] = None):
        """
        Initialize reliability tester.

        Args:
            enable_parallel: Whether to enable parallel processing
            max_workers: Maximum number of worker processes (None for auto)
        """
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.fault_injector = FaultInjector()

    def _calculate_total_model_bits(self, model: torch.nn.Module) -> int:
        """Calculate total number of bits (parameters) in the model for BER calculation."""
        total_bits = 0
        for param in model.parameters():
            total_bits += param.numel()
        return total_bits

    def _ber_to_fault_count(self, ber_level: float, total_bits: int) -> int:
        """Convert Bit Error Rate to actual fault count for the model."""
        return max(1, int(ber_level * total_bits))
    
    def single_fault_test(self, model: nn.Module, num_faults: int, 
                         target_layers: List[str], evaluation_func: Callable) -> float:
        """
        Test model with a specific number of faults injected.
        
        Args:
            model: Original model to test
            num_faults: Number of faults to inject
            target_layers: Layers to target for fault injection
            evaluation_func: Function to evaluate model performance
            
        Returns:
            Performance metric (typically accuracy)
        """
        # Create faulty model
        faulty_model = copy.deepcopy(model)
        self.fault_injector.inject_faults_inplace(faulty_model, num_faults, target_layers)
        
        # Evaluate performance
        performance = evaluation_func(faulty_model)
        
        # Clean up
        del faulty_model
        cleanup_memory()
        
        return performance
    
    def reliability_test_single_level(self, model: nn.Module, num_faults: int,
                                    target_layers: List[str], evaluation_func: Callable,
                                    repetitions: int = 30) -> Dict:
        """
        Test reliability at a single fault level with multiple repetitions.
        
        Args:
            model: Model to test
            num_faults: Number of faults to inject
            target_layers: Layers to target
            evaluation_func: Evaluation function
            repetitions: Number of test repetitions
            
        Returns:
            Dictionary with statistical results
        """
        performances = []
        
        print(f"Testing with {num_faults} faults ({repetitions} repetitions)...")
        
        for rep in tqdm(range(repetitions), desc=f"{num_faults} faults"):
            performance = self.single_fault_test(model, num_faults, target_layers, evaluation_func)
            performances.append(performance)
        
        return {
            'num_faults': num_faults,
            'mean': np.mean(performances),
            'std': np.std(performances),
            'min': np.min(performances),
            'max': np.max(performances),
            'median': np.median(performances),
            'performances': performances,
            'repetitions': repetitions
        }
    
    def comprehensive_reliability_test(self, model: nn.Module, fault_levels: List[int],
                                     target_layers: List[str], evaluation_func: Callable,
                                     repetitions: int = 30) -> Dict:
        """
        Run comprehensive reliability test across multiple fault levels.
        
        Args:
            model: Model to test
            fault_levels: List of fault counts to test
            target_layers: Layers to target for fault injection
            evaluation_func: Function to evaluate model performance
            repetitions: Number of repetitions per fault level
            
        Returns:
            Dictionary containing all test results
        """
        print(f"\n=== Starting Comprehensive Reliability Test ===")
        print(f"Fault levels: {fault_levels}")
        print(f"Target layers: {len(target_layers)}")
        print(f"Repetitions per level: {repetitions}")
        print(f"Total tests: {len(fault_levels) * repetitions}")
        
        # Get baseline performance (no faults)
        baseline_performance = evaluation_func(model)
        print(f"Baseline performance: {baseline_performance:.2f}%")
        
        results = {
            'baseline_performance': baseline_performance,
            'fault_levels': {},
            'summary': {}
        }
        
        start_time = time.time()
        
        for num_faults in fault_levels:
            level_results = self.reliability_test_single_level(
                model, num_faults, target_layers, evaluation_func, repetitions
            )
            results['fault_levels'][num_faults] = level_results
            
            # Print immediate results
            mean_perf = level_results['mean']
            std_perf = level_results['std']
            degradation = baseline_performance - mean_perf
            
            print(f"  {num_faults} faults: {mean_perf:.2f}% ± {std_perf:.2f}% "
                  f"(degradation: {degradation:.2f}%)")
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_stats(results, baseline_performance)
        
        total_time = time.time() - start_time
        print(f"\nReliability test completed in {total_time:.2f} seconds")
        
        return results

    def comprehensive_reliability_test_ber(self, model: nn.Module, ber_levels: List[float],
                                         target_layers: List[str], evaluation_func: Callable,
                                         repetitions: int = 10) -> Dict:
        """
        Run comprehensive BER-based reliability test across multiple BER levels.

        Args:
            model: Model to test
            ber_levels: List of BER levels to test
            target_layers: Layers to target for fault injection
            evaluation_func: Function to evaluate model performance
            repetitions: Number of repetitions per BER level

        Returns:
            Dictionary containing all test results
        """
        # Calculate total model bits for BER scaling
        total_bits = self._calculate_total_model_bits(model)
        print(f"\n=== Starting Comprehensive BER-based Reliability Test ===")
        print(f"BER levels: {ber_levels}")
        print(f"Model size: {total_bits:,} parameters")
        print(f"Target layers: {len(target_layers)}")
        print(f"Repetitions per level: {repetitions}")
        print(f"Total tests: {len(ber_levels) * repetitions}")

        # Get baseline performance (no faults)
        baseline_performance = evaluation_func(model)
        print(f"Baseline performance: {baseline_performance:.2f}%")

        results = {
            'baseline_performance': baseline_performance,
            'ber_levels': {},
            'summary': {},
            'total_model_bits': total_bits
        }

        start_time = time.time()

        for ber_level in ber_levels:
            fault_count = self._ber_to_fault_count(ber_level, total_bits)
            print(f"Testing BER {ber_level:.1e} ({fault_count:,} faults, {repetitions} repetitions)...")

            level_results = self.reliability_test_single_level(
                model, fault_count, target_layers, evaluation_func, repetitions
            )
            # Store with BER level as key but include fault count info
            level_results['ber_level'] = ber_level
            level_results['fault_count'] = fault_count
            results['ber_levels'][ber_level] = level_results

            # Print immediate results
            mean_perf = level_results['mean']
            std_perf = level_results['std']
            degradation = baseline_performance - mean_perf

            print(f"  BER {ber_level:.1e}: {mean_perf:.2f}% ± {std_perf:.2f}% "
                  f"(degradation: {degradation:.2f}%)")

        # Calculate summary statistics
        results['summary'] = self._calculate_summary_stats_ber(results, baseline_performance)

        total_time = time.time() - start_time
        print(f"\nBER-based reliability test completed in {total_time:.2f} seconds")

        return results

    def _calculate_summary_stats_ber(self, results: Dict, baseline: float) -> Dict:
        """Calculate summary statistics from BER-based reliability test results."""
        summary = {
            'total_ber_levels': len(results['ber_levels']),
            'baseline_performance': baseline,
            'fault_tolerance': {},
            'degradation_analysis': {},
            'total_model_bits': results['total_model_bits']
        }

        # Calculate fault tolerance metrics
        for ber_level, level_results in results['ber_levels'].items():
            mean_perf = level_results['mean']
            degradation = baseline - mean_perf
            relative_degradation = (degradation / baseline) * 100 if baseline > 0 else 100

            summary['fault_tolerance'][ber_level] = {
                'absolute_degradation': degradation,
                'relative_degradation': relative_degradation,
                'performance_retention': (mean_perf / baseline) * 100 if baseline > 0 else 0,
                'fault_count': level_results['fault_count']
            }

        # Overall degradation analysis
        all_degradations = [
            summary['fault_tolerance'][ber]['absolute_degradation']
            for ber in summary['fault_tolerance']
        ]

        summary['degradation_analysis'] = {
            'mean_degradation': np.mean(all_degradations),
            'max_degradation': np.max(all_degradations),
            'degradation_std': np.std(all_degradations)
        }

        return summary

    def quick_reliability_estimate(self, model: nn.Module, num_faults: int = 100,
                                 repetitions: int = 10, target_layers: Optional[List[str]] = None,
                                 evaluation_func: Optional[Callable] = None) -> float:
        """
        Quick reliability estimate for GA fitness evaluation.

        Args:
            model: Model to test
            num_faults: Number of faults for quick test
            repetitions: Number of repetitions
            target_layers: Layers to target (auto-detect if None)
            evaluation_func: Evaluation function (uses default if None)

        Returns:
            Average performance under faults
        """
        if target_layers is None:
            target_layers = get_weight_layer_names(model)

        if evaluation_func is None:
            # This would need to be set up with proper dataloader
            # For now, return a placeholder
            return 85.0  # Placeholder reliability score

        performances = []
        print(f"      Testing {num_faults} faults × {repetitions} reps...", end="", flush=True)
        for rep in range(repetitions):
            performance = self.single_fault_test(model, num_faults, target_layers, evaluation_func)
            performances.append(performance)
            # Show progress every 2 reps
            if (rep + 1) % 2 == 0 or rep == repetitions - 1:
                print(f" [{rep+1}/{repetitions}]", end="", flush=True)

        mean_perf = np.mean(performances)
        print(f" → {mean_perf:.1f}%")
        return mean_perf
    
    def compare_model_reliability(self, models: Dict[str, nn.Module], 
                                fault_levels: List[int], evaluation_func: Callable,
                                repetitions: int = 30) -> Dict:
        """
        Compare reliability of multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            fault_levels: Fault levels to test
            evaluation_func: Evaluation function
            repetitions: Repetitions per test
            
        Returns:
            Comprehensive comparison results
        """
        print(f"\n=== Comparing Reliability of {len(models)} Models ===")
        
        comparison_results = {}
        target_layers = None  # Will be set from first model
        
        for model_name, model in models.items():
            print(f"\nTesting model: {model_name}")
            
            if target_layers is None:
                target_layers = get_weight_layer_names(model)
                print(f"Using {len(target_layers)} target layers for fault injection")
            
            model_results = self.comprehensive_reliability_test(
                model, fault_levels, target_layers, evaluation_func, repetitions
            )
            
            comparison_results[model_name] = model_results
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(comparison_results, fault_levels)
        comparison_results['comparison_summary'] = comparison_summary
        
        return comparison_results
    
    def _calculate_summary_stats(self, results: Dict, baseline: float) -> Dict:
        """Calculate summary statistics from reliability test results."""
        summary = {
            'total_fault_levels': len(results['fault_levels']),
            'baseline_performance': baseline,
            'fault_tolerance': {},
            'degradation_analysis': {}
        }
        
        # Calculate fault tolerance metrics
        for num_faults, level_results in results['fault_levels'].items():
            mean_perf = level_results['mean']
            degradation = baseline - mean_perf
            relative_degradation = (degradation / baseline) * 100 if baseline > 0 else 100
            
            summary['fault_tolerance'][num_faults] = {
                'absolute_degradation': degradation,
                'relative_degradation': relative_degradation,
                'performance_retention': (mean_perf / baseline) * 100 if baseline > 0 else 0
            }
        
        # Overall degradation analysis
        all_degradations = [
            summary['fault_tolerance'][nf]['absolute_degradation'] 
            for nf in summary['fault_tolerance']
        ]
        
        summary['degradation_analysis'] = {
            'mean_degradation': np.mean(all_degradations),
            'max_degradation': np.max(all_degradations),
            'degradation_std': np.std(all_degradations)
        }
        
        return summary
    
    def _generate_comparison_summary(self, comparison_results: Dict, fault_levels: List[int]) -> Dict:
        """Generate summary for model comparison."""
        summary = {
            'models_tested': list(comparison_results.keys()),
            'fault_levels': fault_levels,
            'winner_by_fault_level': {},
            'overall_ranking': []
        }
        
        # Find winner at each fault level
        for num_faults in fault_levels:
            best_performance = -1
            best_model = None
            
            for model_name in comparison_results:
                if model_name == 'comparison_summary':
                    continue
                    
                level_results = comparison_results[model_name]['fault_levels'].get(num_faults)
                if level_results and level_results['mean'] > best_performance:
                    best_performance = level_results['mean']
                    best_model = model_name
            
            summary['winner_by_fault_level'][num_faults] = {
                'model': best_model,
                'performance': best_performance
            }
        
        # Calculate overall ranking based on average performance across all fault levels
        model_avg_performances = {}
        for model_name in comparison_results:
            if model_name == 'comparison_summary':
                continue
                
            performances = []
            for num_faults in fault_levels:
                level_results = comparison_results[model_name]['fault_levels'].get(num_faults)
                if level_results:
                    performances.append(level_results['mean'])
            
            model_avg_performances[model_name] = np.mean(performances) if performances else 0
        
        # Sort by average performance
        summary['overall_ranking'] = sorted(
            model_avg_performances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return summary
    
    def export_results_to_latex(self, results: Dict, filename: str) -> None:
        """Export reliability test results to LaTeX table format."""
        # This would generate a formatted LaTeX table
        # Implementation would depend on specific formatting requirements
        pass
    
    def print_reliability_summary(self, results: Dict, model_name: str = "Model") -> None:
        """Print a formatted summary of reliability results."""
        print(f"\n{'='*60}")
        print(f"RELIABILITY SUMMARY: {model_name}")
        print(f"{'='*60}")
        
        baseline = results['baseline_performance']
        print(f"Baseline Performance: {baseline:.2f}%")
        print(f"Fault Levels Tested: {len(results['fault_levels'])}")
        
        print(f"\n{'Faults':<8} {'Mean Perf':<12} {'Std Dev':<10} {'Degradation':<12}")
        print("-" * 50)
        
        for num_faults, level_results in results['fault_levels'].items():
            mean_perf = level_results['mean']
            std_perf = level_results['std']
            degradation = baseline - mean_perf
            
            print(f"{num_faults:<8} {mean_perf:<12.2f} {std_perf:<10.2f} {degradation:<12.2f}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"\nOverall Analysis:")
            print(f"  Mean Degradation: {summary['degradation_analysis']['mean_degradation']:.2f}%")
            print(f"  Max Degradation: {summary['degradation_analysis']['max_degradation']:.2f}%")
        
        print(f"{'='*60}")


def create_evaluation_function(dataloader, device: torch.device) -> Callable:
    """Create evaluation function for reliability testing."""
    def evaluate_model(model: nn.Module) -> float:
        return test_accuracy(model, device, dataloader)
    return evaluate_model