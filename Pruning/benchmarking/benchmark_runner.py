"""Main benchmarking runner for comparing pruning methods."""

import os
import time
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Callable
import torch
import torch.nn as nn
import copy

from config.benchmark_config import BenchmarkConfig
from config.model_config import ModelConfig
from core.models import create_model_from_config
from core.data import get_data_loaders, get_single_dataloader
from core.utils import cleanup_memory, create_directories, format_time
from benchmarking.reliability.reliability_test import ReliabilityTester, create_evaluation_function
from benchmarking.unstructured.classical.magnitude import MagnitudePruning, LayerWiseMagnitudePruning
from benchmarking.unstructured.classical.random import RandomPruning
from benchmarking.unstructured.y2023.wanda import WANDAPruning
from benchmarking.unstructured.y2019.snip import SNIPPruning, GraSPPruning
from benchmarking.unstructured.y2023.pdp import PDPPruning
from benchmarking.unstructured.y2024.evop import EVOPPruning
from benchmarking.custom_strategy import CustomStrategyLoader
from benchmarking.reliability.fault_analysis import ComprehensiveFaultAnalyzer
from genetic_algorithm.agents import PruningStrategyAgent, ModelPruningAgent, FineTuningAgent, EvaluationAgent, ConstrainedLayerwisePruningAgent, SensitivityAwarePruningStrategyAgent
from genetic_algorithm.sensitivity_driven_agents import SensitivityAwarePruningAgent

class BenchmarkRunner:
    """Main benchmarking framework for comparing pruning methods."""
    
    def __init__(self, benchmark_config: BenchmarkConfig, model_config: ModelConfig):
        self.benchmark_config = benchmark_config
        self.model_config = model_config
        self.base_model = None
        self.train_loader = None
        self.val_loader = None
        self.reliability_tester = None
        self.benchmark_results = {}
        self.finetuner = None
        self._initialize_components()
    
    def _initialize_components(self):
        print("=== Initializing Benchmark Components ===")
        self.base_model = create_model_from_config(self.model_config)

        # DIAGNOSTIC: Check base model accuracy before pruning
        self.base_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in get_data_loaders(self.model_config)[1]:  # Use val loader
                images, labels = images.to(self.model_config.device), labels.to(self.model_config.device)
                outputs = self.base_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total >= 1000:  # Quick check with 1000 samples
                    break
        base_accuracy = (correct / total) * 100.0
        print(f"✓ Base model accuracy (before pruning): {base_accuracy:.2f}% (should be ~90%)")
        if base_accuracy < 80:
            print(f"⚠️  WARNING: Base model accuracy is too low! Model may not be loading pretrained weights correctly!")

        self.train_loader, self.val_loader = get_data_loaders(self.model_config)
        self.reliability_tester = ReliabilityTester(
            enable_parallel=self.benchmark_config.enable_parallel_reliability,
            max_workers=self.benchmark_config.max_workers_reliability
        )
        self.finetuner = FineTuningAgent(
            self.train_loader, 
            self.val_loader, 
            self.benchmark_config.device
        )
        create_directories([self.benchmark_config.results_dir])
        print("✓ All benchmark components initialized")
    
    def _finetune_model(self, pruned_model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        print("  Skipping fine-tuning (disabled for benchmark)")
        return pruned_model
    
    def _finetune_model_ga(self, pruned_model: nn.Module, pruning_masks: Dict[str, torch.Tensor]) -> nn.Module:
        """Research-grade fine-tuning implementation for GA strategies (adapted from hypothesis test)."""
        # ENABLED: Fine-tuning is critical for vulnerability-based pruning (mentor's approach)
        # The vulnerability approach removes weak points, but remaining weights need adaptation

        epochs = self.benchmark_config.finetune_epochs
        lr = self.benchmark_config.finetune_lr
        patience = 3
        
        try:
            target_model = pruned_model._orig_mod if hasattr(pruned_model, '_orig_mod') else pruned_model
            target_model.train()
            target_model.to(self.benchmark_config.device)
            
            # CRITICAL FIX: Match GA fine-tuning config exactly
            # NO weight_decay (hurts vulnerability-based pruning that relies on magnitude)
            # NO LR scheduler (constant LR like GA)
            optimizer = torch.optim.Adam(target_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            loss_fn = nn.CrossEntropyLoss()
            
            best_accuracy = 0.0
            patience_counter = 0
            best_model_state = None
            
            print(f"  Fine-tuning for {epochs} epochs...")
            
            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_loss = 0.0
                target_model.train()
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.benchmark_config.device), target.to(self.benchmark_config.device)
                    
                    optimizer.zero_grad()
                    output = target_model(data)
                    loss = loss_fn(output, target)
                    loss.backward()

                    # Apply masks to gradients to maintain sparsity
                    for name, param in target_model.named_parameters():
                        if name in pruning_masks and param.grad is not None:
                            mask = pruning_masks[name]
                            if param.grad.numel() == mask.numel():
                                param.grad.data.view(-1).mul_(mask.to(param.device))

                    # CRITICAL FIX: Add gradient clipping like GA (prevents exploding gradients)
                    torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)

                    optimizer.step()
                    
                    # Zero out pruned weights to maintain exact sparsity (DOUBLE ENFORCEMENT)
                    with torch.no_grad():
                        for name, param in target_model.named_parameters():
                            if name in pruning_masks:
                                mask = pruning_masks[name]
                                if param.numel() == mask.numel():
                                    # Use exact zero assignment for perfect sparsity preservation
                                    mask_reshaped = mask.view(param.shape).to(param.device)
                                    param.data[mask_reshaped == 0] = 0.0
                    
                    epoch_loss += loss.item()

                # CRITICAL FIX: Validate EVERY epoch like GA (better early stopping)
                epoch_train_time = time.time() - epoch_start
                target_model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.benchmark_config.device), labels.to(self.benchmark_config.device)
                        outputs = target_model(images)

                        # Check for corrupted outputs
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            total += labels.size(0)
                            continue

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                current_accuracy = (correct / total) * 100.0 if total > 0 else 0.0
                print(f"    Epoch {epoch+1}: Loss={epoch_loss/(batch_idx+1):.4f}, Acc={current_accuracy:.1f}%, Time={epoch_train_time:.1f}s")

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    patience_counter = 0
                    best_model_state = copy.deepcopy(target_model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            # Restore best model
            if best_model_state is not None:
                target_model.load_state_dict(best_model_state)
            
            target_model.eval()
            cleanup_memory()
            print(f"  ✓ Fine-tuning completed ({epochs} epochs)")
            return target_model
            
        except Exception as e:
            print(f"  WARNING: Fine-tuning failed: {e}. Returning model without fine-tuning.")
            return pruned_model
        
    def _benchmark_custom_strategies(self, custom_strategies_csv_path: str) -> Dict[str, Any]:
        results = {}
        if not custom_strategies_csv_path:
            return results
    
        print(f"\n{'-'*60}\nBENCHMARKING CUSTOM STRATEGIES from {custom_strategies_csv_path}\n{'-'*60}")
    
        try:
            strategy_loader = CustomStrategyLoader(
                model_state_dict=self.base_model.state_dict(),
                score_dir_path=self.model_config.score_dir_path
            )
            custom_strategies = strategy_loader.load_strategies_from_csv(
                custom_strategies_csv_path, 
                best_only=self.benchmark_config.use_best_reliable_strategy_only
            )
            print(f"Loaded {len(custom_strategies)} custom strategies.")
        except Exception as e:
            print(f"Error loading custom strategies: {e}")
            return results
        
        model_pruner = ModelPruningAgent()
        for idx, strategy_data in enumerate(custom_strategies):
            strategy_name = strategy_data.get('name', f'custom_strategy_{idx}')

            # CRITICAL FIX: Convert percentile dictionary to list in ALPHABETICAL order
            # The GA CSV was generated with alphabetical ordering (sorted)
            # So we must use the same ordering for the agent to get correct results
            percentile_dict = strategy_data['percentile_dict']

            # Use ALPHABETICAL ordering (same as GA used when generating this CSV)
            from core.utils import get_layer_score_files_map

            score_files_map = get_layer_score_files_map(
                self.model_config.score_dir_path,
                self.base_model.state_dict()
            )
            available_layers = sorted([  # ALPHABETICAL, not sensitivity-based!
                name for name in self.base_model.state_dict().keys()
                if ('features' in name or 'classifier' in name) and 'weight' in name
                and name in score_files_map
            ])

            # Convert dict to list in alphabetical order
            individual_percentiles = [percentile_dict[layer] for layer in available_layers]

            print(f"Evaluating custom strategy: {strategy_name}")
            print(f"DEBUG: Converted {len(percentile_dict)} layer percentiles to agent order: {[f'{p:.2f}' for p in individual_percentiles[:5]]}...")

            try:
                # Use SensitivityAwarePruningAgent to match GA behavior
                actual_sparsity = strategy_data.get('actual_sparsity')
                print(f"  Using SensitivityAwarePruningAgent with target sparsity: {actual_sparsity:.2f}%" if actual_sparsity else "  Using SensitivityAwarePruningAgent")
                from genetic_algorithm.sensitivity_driven_agents import SensitivityAwarePruningAgent
                strategy_agent = SensitivityAwarePruningAgent(
                    strategy_params=individual_percentiles,
                    model_state_dict=self.base_model.state_dict(),
                    score_dir_path=self.model_config.score_dir_path,
                    use_sensitivity_constraints=True
                )
                pruning_masks = strategy_agent.generate_pruning_mask(device=self.benchmark_config.device)
                current_model = copy.deepcopy(self.base_model)
                pruned_model = model_pruner.prune_model(current_model, pruning_masks)
                finetuned_model = self._finetune_model_ga(pruned_model, pruning_masks)
                
                results[strategy_name] = self._evaluate_pruned_model(finetuned_model, 'custom_ga_strategy')
                
                del current_model, pruned_model, finetuned_model
                cleanup_memory()
    
            except Exception as e:
                print(f"  ERROR evaluating strategy {strategy_name}: {e}")
                results[strategy_name] = {'success': False, 'error': str(e), 'method_type': 'custom_ga_strategy'}
                cleanup_memory()
        return results
    
    def _benchmark_classical_methods(self) -> Dict[str, Any]:
        results = {}
        methods = {
            'magnitude_global': lambda: MagnitudePruning().prune_model(self.base_model, self.benchmark_config.target_sparsity, global_pruning=True),
            'random_global': lambda: RandomPruning().prune_model(self.base_model, self.benchmark_config.target_sparsity, global_pruning=True)
        }
        for name, func in methods.items():
            if name in self.benchmark_config.classical_methods:
                print(f"\nTesting {name}...")
                results[name] = self._evaluate_single_method(name, func, 'classical')
        return results
    
    def _benchmark_sota_methods(self) -> Dict[str, Any]:
        results = {}
        calibration_loader = get_single_dataloader(self.model_config, train=True, batch_size=32, num_workers=0)
        
        methods = {
            'wanda': lambda: WANDAPruning().prune_model(self.base_model, self.benchmark_config.target_sparsity, calibration_loader, self.benchmark_config.device),
            'snip': lambda: SNIPPruning().prune_model(self.base_model, self.benchmark_config.target_sparsity, calibration_loader, self.benchmark_config.device),
            'grasp': lambda: GraSPPruning().prune_model(self.base_model, self.benchmark_config.target_sparsity, calibration_loader, self.benchmark_config.device),
            'pdp': lambda: PDPPruning().prune_model(self.base_model, self.benchmark_config.target_sparsity, self.train_loader, self.benchmark_config.device),
            'evop_ga_accuracy': lambda: EVOPPruning().prune_model(self.base_model, self.train_loader, self.benchmark_config.target_sparsity / 100.0)
        }
        for name, func in methods.items():
            if name in self.benchmark_config.sota_methods:
                print(f"\nTesting {name}...")
                results[name] = self._evaluate_single_method(name, func, 'sota')
        return results

    def _evaluate_single_method(self, method_name: str, method_func: Callable, method_type: str) -> Dict[str, Any]:
        try:
            pruned_model, masks = method_func()
            finetuned_model = self._finetune_model(pruned_model, masks)
            results = self._evaluate_pruned_model(finetuned_model, method_type)
            del pruned_model, finetuned_model
            cleanup_memory()
            return results
        except Exception as e:
            print(f"  ERROR testing {method_name}: {e}")
            cleanup_memory()
            return {'accuracy': 0.0, 'sparsity': 0.0, 'reliability_results': None, 'method_type': method_type, 'success': False, 'error': str(e)}

    def _evaluate_pruned_model(self, model: nn.Module, method_type: str) -> Dict[str, Any]:
        eval_agent = EvaluationAgent(self.val_loader, self.benchmark_config.device)
        dummy_input = self.model_config.get_dummy_input()
        metrics = eval_agent.evaluate(model, dummy_input)
        
        target_layers = [name for name, _ in model.named_parameters() if 'weight' in name]
        reliability_results = self.reliability_tester.comprehensive_reliability_test_ber(
            model=model,
            ber_levels=self.benchmark_config.ber_levels,
            target_layers=target_layers,
            evaluation_func=create_evaluation_function(self.val_loader, self.benchmark_config.device),
            repetitions=self.benchmark_config.reliability_repetitions
        )
        return {**metrics, 'reliability_results': reliability_results, 'method_type': method_type, 'success': True}

    def run_comprehensive_benchmark_with_analysis(self) -> Dict[str, Any]:
        print("Starting Comprehensive Benchmark with Enhanced Analysis")
        all_results = {}
        actual_target_sparsity = self.benchmark_config.target_sparsity

        if self.benchmark_config.custom_strategies_csv:
            print(f"\n{'🧬 STARTING WITH GA RELIABILITY-AWARE METHODS':=^80}")
            ga_results = self._benchmark_custom_strategies(self.benchmark_config.custom_strategies_csv)
            all_results.update(ga_results)
            
            if ga_results:
                ga_method_name = list(ga_results.keys())[0]
                if ga_results[ga_method_name].get('success', False):
                    actual_target_sparsity = ga_results[ga_method_name]['sparsity']
                    print(f"GA achieved {actual_target_sparsity:.2f}% sparsity. Updating ALL methods to this target.")
                    self.benchmark_config.target_sparsity = actual_target_sparsity
        
        if self.benchmark_config.enable_classical_methods:
            all_results.update(self._benchmark_classical_methods())
        
        if self.benchmark_config.enable_sota_methods:
            all_results.update(self._benchmark_sota_methods())
        
        self.benchmark_results = all_results
        self._save_and_analyze_results()
        return all_results
    
    def print_benchmark_summary(self):
        """Prints a formatted summary of the benchmark results to the console."""
        print("\n" + "="*80)
        print(f"{'BENCHMARK SUMMARY':^80}")
        print("="*80)

        if not self.benchmark_results:
            print("No benchmark results to display.")
            print("="*80)
            return

        summary_data = []
        for method_name, results in self.benchmark_results.items():
            if results.get('success', True):
                row = {
                    'Method': method_name,
                    'Type': results.get('method_type', 'unknown'),
                    'Accuracy': f"{results.get('accuracy', 0):.2f}%",
                    'Sparsity': f"{results.get('sparsity', 0):.2f}%"
                }
                
                if results.get('reliability_results'):
                    # Handle both old fault_levels and new ber_levels format
                    if 'ber_levels' in results['reliability_results']:
                        for ber_level, stats in results['reliability_results']['ber_levels'].items():
                            row[f'Reliability @ BER {ber_level:.0e}'] = f"{stats['mean']:.2f}%"
                    elif 'fault_levels' in results['reliability_results']:
                        for fault_level, stats in results['reliability_results']['fault_levels'].items():
                            row[f'Reliability @ {fault_level} faults'] = f"{stats['mean']:.2f}%"
                summary_data.append(row)
        
        if not summary_data:
            print("No successful benchmark runs to display.")
            print("="*80)
            return
            
        summary_df = pd.DataFrame(summary_data)
        
       
        cols = ['Method', 'Type', 'Accuracy', 'Sparsity']
        reliability_cols = sorted([col for col in summary_df.columns if 'Reliability' in col])
        
        
        existing_cols = [col for col in cols if col in summary_df.columns]
        final_cols = existing_cols + reliability_cols
        
        
        if not reliability_cols and 'Method' in summary_df.columns:
             final_cols = [col for col in ['Method', 'Type', 'Accuracy', 'Sparsity'] if col in summary_df.columns]
        
        summary_df = summary_df[final_cols]
        
        print(summary_df.to_string(index=False))
        print("="*80)
    
    def _save_and_analyze_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_benchmark_summary(timestamp)
        # Placeholder for future additions like plotting or full pickle save
        print("Benchmark analysis and saving complete.")

    def _save_benchmark_summary(self, timestamp: str):
        summary_data = []
        for method_name, results in self.benchmark_results.items():
            if results.get('success', True):
                row = {'method': method_name, 'method_type': results.get('method_type', 'unknown'),
                       'accuracy': results.get('accuracy', 0), 'sparsity': results.get('sparsity', 0)}
                
                if results.get('reliability_results'):
                    # Handle both old fault_levels and new ber_levels format
                    if 'ber_levels' in results['reliability_results']:
                        for ber_level, stats in results['reliability_results']['ber_levels'].items():
                            row[f'reliability_ber_{ber_level:.0e}'] = stats['mean']
                    elif 'fault_levels' in results['reliability_results']:
                        for fault_level, stats in results['reliability_results']['fault_levels'].items():
                            row[f'reliability_{fault_level}_faults'] = stats['mean']
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.benchmark_config.results_dir, f"benchmark_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"✓ Summary saved to {summary_file}")

def create_benchmark_runner(benchmark_config: BenchmarkConfig, model_config: ModelConfig) -> BenchmarkRunner:
    return BenchmarkRunner(benchmark_config, model_config)