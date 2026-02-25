"""Global Genetic Algorithm Configuration."""

import torch
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class GlobalGAConfig:
    """Configuration for Global Genetic Algorithm with sensitivity-based global pruning."""
    
    # GA Parameters
    population_size: int = 50
    num_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    mutation_sigma: float = 5.0  # Smaller sigma for global sparsity mutation
    mutation_indpb: float = 0.3
    
    # Multi-objective weights (Accuracy, Latency, Reliability, Sparsity)
    fitness_weights: tuple = (3.0, -1.0, 3.5, 1.0)
    
    # Global Pruning Strategy Parameters
    global_sparsity_min: float = 10.0     # Minimum global sparsity percentage
    global_sparsity_max: float = 80.0     # Maximum global sparsity percentage 
    min_overall_sparsity_threshold: float = 15.0
    sparsity_weight_factor: float = 100.0
    
    # Layer type weights for hybrid global approach (disabled by default for pure global)
    enable_layer_type_weighting: bool = False
    conv_layer_weight: float = 1.0        # Equal treatment (pure global)
    fc_layer_weight: float = 1.0          # Equal treatment (pure global)
    
    # Sensitivity score normalization
    normalize_sensitivity_scores: bool = True
    sensitivity_normalization_method: str = "z_score"  # "z_score", "min_max", "layer_wise"
    
    # Advanced global features
    enable_progressive_pruning: bool = False  # Gradually increase sparsity over generations
    progressive_sparsity_increase: float = 2.0  # % increase per generation
    
    # Layer protection (prevent complete layer destruction)
    min_layer_survival_rate: float = 0.05  # At least 5% weights per layer must survive
    enable_layer_protection: bool = True
    
    # Reliability Assessment for GA (optimized for evolution speed)
    reliability_estimation_faults: int = 10
    reliability_estimation_reps: int = 5
    fault_type: str = "bit_flip"
    
    # Fine-tuning Parameters
    finetune_epochs: int = 20
    finetune_lr: float = 1e-4
    finetune_patience: int = 3
    
    # Checkpointing
    checkpoint_frequency: int = 5
    checkpoint_path: str = "global_ga_checkpoint.pkl"
    
    # Device and Paths
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_dir_path: str = "VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"
    
    # Results
    results_dir: str = "global_ga_results"
    save_pareto_front: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.num_generations <= 0:
            raise ValueError("Number of generations must be positive")
        if not (0 <= self.crossover_prob <= 1):
            raise ValueError("Crossover probability must be between 0 and 1")
        if not (0 <= self.mutation_prob <= 1):
            raise ValueError("Mutation probability must be between 0 and 1")
        if not (0 <= self.global_sparsity_min <= self.global_sparsity_max <= 100):
            raise ValueError("Global sparsity bounds must be between 0-100 and min <= max")
        if self.enable_layer_protection and not (0 < self.min_layer_survival_rate < 1):
            raise ValueError("Layer survival rate must be between 0 and 1")
    
    @classmethod
    def create_local_config(cls, score_dir_path: str = "VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for local development."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=20,  # Smaller for local testing
            num_generations=20,
            global_sparsity_min=15.0,
            global_sparsity_max=70.0,
            reliability_estimation_faults=10,
            reliability_estimation_reps=5
        )
    
    @classmethod
    def create_server_config(cls, score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for server deployment."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=50,
            num_generations=50,
            global_sparsity_min=10.0,
            global_sparsity_max=80.0,
            reliability_estimation_faults=10,
            reliability_estimation_reps=5
        )
    
    @classmethod
    def create_aggressive_config(cls, **kwargs):
        """Create configuration for aggressive global pruning research."""
        return cls(
            population_size=30,
            num_generations=40,
            global_sparsity_min=20.0,
            global_sparsity_max=90.0,
            min_overall_sparsity_threshold=25.0,
            conv_layer_weight=0.5,     # Very conservative conv pruning
            fc_layer_weight=1.8,       # Very aggressive FC pruning
            enable_progressive_pruning=True,
            progressive_sparsity_increase=1.5,
            **kwargs
        )
    
    @classmethod 
    def create_hybrid_config(cls, **kwargs):
        """Create configuration for hybrid global-layerwise approach."""
        return cls(
            population_size=40,
            num_generations=50,
            enable_layer_type_weighting=True,
            conv_layer_weight=0.8,
            fc_layer_weight=1.2,
            normalize_sensitivity_scores=True,
            sensitivity_normalization_method="layer_wise",
            enable_layer_protection=True,
            min_layer_survival_rate=0.1,
            **kwargs
        )