#Benchmarking Configuration.

import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from typing import List, Dict, Any, Optional # Add Optional

@dataclass
class BenchmarkConfig:
    
    
    # General Benchmarking Parameters
    target_sparsity: float = 65.0
    finetune_epochs: int = 20  # FIXED: Match GA config (was 25, causing mismatch)
    finetune_lr: float = 1e-4
    
    # BER-based Reliability Testing Parameters
    ber_levels: List[float] = None
    reliability_repetitions: int = 10
    fault_type: str = "bit_flip"  # Only bit-flip faults for realistic hardware simulation
    enable_parallel_reliability: bool = True
    max_workers_reliability: int = 4
    
    # Methods to benchmark
    enable_classical_methods: bool = True
    enable_sota_methods: bool = True
    enable_structured_methods: bool = True
    enable_ga_comparison: bool = True
    
    # Classical Methods Configuration
    classical_methods: Dict[str, Dict[str, Any]] = None
    
    # SOTA Methods Configuration
    sota_methods: Dict[str, Dict[str, Any]] = None
    
    # Structured Methods Configuration
    structured_methods: Dict[str, Dict[str, Any]] = None
    
    # Output and Results
    results_dir: str = "benchmark_results"
    save_latex_tables: bool = True
    save_plots: bool = True
    generate_statistical_analysis: bool = True
    
    # Device and Paths
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_strategies_csv: Optional[str] = None
    use_best_reliable_strategy_only: bool = True
    
    # Quantization Options
    enable_quantization: bool = False
    quantization_strategies: List[str] = None
    quantization_calibration_samples: int = 100
    def __post_init__(self):
        """Initialize default configurations."""
        if self.ber_levels is None:
            self.ber_levels = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # Research-focused BER levels
        
        if self.classical_methods is None:
            self.classical_methods = {
                "magnitude_global": {"global_pruning": True},
                "random_global": {"seed": 42}
            }
        
        if self.sota_methods is None:
            self.sota_methods = {
                "wanda": {},
                "snip": {},
                "grasp": {},
                "pdp": {},
                "evop_ga_accuracy": {}
            }
        
        if self.structured_methods is None:
            self.structured_methods = {
                "channel_pruning": {"importance_metric": "l1_norm"},
                "filter_pruning": {"importance_metric": "l2_norm"}
            }
        
        if self.quantization_strategies is None:
            self.quantization_strategies = ["int8_static", "mixed_sensitive"] if self.enable_quantization else []
    
    @classmethod
    def create_quick_benchmark_config(cls):
        """Create configuration for quick benchmarking (fewer methods, faster)."""
        return cls(
            target_sparsity=50.0,
            ber_levels=[5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            reliability_repetitions=10,
            classical_methods={
                "magnitude_global": {"global_pruning": True},
                "random_global": {"seed": 42}
            },
            sota_methods={
                "wanda": {"activation_aware": True},
                "snip": {"single_shot": True},
                "torque": {"structured": True, "global_ranking": True}
            },
            enable_structured_methods=False
        )
    
    @classmethod
    def create_comprehensive_config(cls):
        """Create configuration for comprehensive benchmarking."""
        return cls(
            target_sparsity=65.0,
            ber_levels=[5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            reliability_repetitions=10,  # Reduced for quick testing
            enable_classical_methods=True,
            enable_sota_methods=True,
            enable_structured_methods=True,
            enable_ga_comparison=True
        )
    
    @classmethod
    def create_ablation_study_config(cls):
        """Create configuration for ablation studies."""
        return cls(
            target_sparsity=65.0,
            ber_levels=[5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            reliability_repetitions=20,
            classical_methods={
                "magnitude_global": {"global_pruning": True},
                "magnitude_layerwise": {"global_pruning": False}
            },
            sota_methods={
                "wanda_baseline": {"activation_aware": False},
                "wanda_activated": {"activation_aware": True},
                "torque": {"structured": True, "global_ranking": True}
            },
            enable_structured_methods=False,
            generate_statistical_analysis=True
        )
    
    @classmethod
    def create_quantization_config(cls):
        """Create configuration for quantization benchmarking."""
        return cls(
            target_sparsity=65.0,
            ber_levels=[5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            reliability_repetitions=20,
            enable_classical_methods=True,
            enable_sota_methods=True,
            enable_structured_methods=False,
            enable_ga_comparison=True,
            enable_quantization=True,
            quantization_strategies=["int8_static", "int8_qat", "mixed_sensitive", "adaptive_8x"],
            quantization_calibration_samples=200
        )