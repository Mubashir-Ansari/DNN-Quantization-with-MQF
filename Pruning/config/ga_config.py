import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm."""
    
    # GA Parameters - RELIABILITY-DOMINANT EXPLORATION
    population_size: int = 50
    num_generations: int = 50
    crossover_prob: float = 0.7  # Balanced for island hopping
    mutation_prob: float = 0.6   #  for exploration
    mutation_sigma: float = 20.0 #  for larger exploration jumps
    mutation_indpb: float = 0.4  #  for more gene-wise exploration
    
    # NSGA-II Multi-objective setup 
    # NSGA-II handles objectives through dominance relationships
    fitness_weights: tuple = (1.0, 1.0)  # Both positive for maximization in NSGA-II
    
    # RELIABILITY DOMINANCE SETTINGS
    reliability_dominance_factor: float = 3.0   # Reliability 3x more important than accuracy
    min_reliability_threshold: float = 25.0     # Below this, severely penalize accuracy fitness
    elite_reliability_threshold: float = 60.0   # Lower threshold for accuracy boost to encourage reliability
    
    # Pruning Strategy Parameters - RELAXED SPARSITY LIMITS FOR BASELINE ACCURACY
    # NOTE: These values are OVERRIDDEN by hardcoded bounds in genetic_algorithm/operators.py
    # ACTUAL LAYER-WISE BOUNDS USED: [30.0, 85.0]% (see operators.py:176,185)
    initial_percentile_min: float = 30.0  # Allow conservative pruning for baseline accuracy (UNUSED - see operators.py)
    initial_percentile_max: float = 70.0  # Reduced max to prevent model destruction (OVERRIDDEN - actual max: 85.0%)
    min_overall_sparsity_threshold: float = 25.0  # Allow meaningful but conservative pruning
    max_overall_sparsity_threshold: float = 80.0  # Upper bound for exploration (UNUSED for layer-wise bounds)
    sparsity_weight_factor: float = 100.0
    
    # Multi-Level BER-based Reliability Assessment for GA
    # UPDATED: Match benchmark's 6 BER levels for consistent comparison
    reliability_ber_levels: tuple = (5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3)  # 6 BER levels (same as benchmark)
    reliability_aggregation: str = "auc_penalty"  # "auc_penalty"
    reliability_level_weights: tuple = (0.5, 1, 1.5, 2, 2.5, 3)  # Weights for BER levels (higher stress = higher weight)
    reliability_estimation_reps: int = 10   # Reduced reps for faster BER-based testing
    fault_type: str = "bit_flip"  # Only bit-flip faults for realistic hardware simulation

    # Legacy single-level support (derived from multi-level)
    @property
    def reliability_estimation_faults(self) -> int:
        """For backward compatibility - returns highest BER level converted to fault count."""
        # This will need model size to convert BER to fault count
        # Return a reasonable default for now
        return 100000  # Placeholder - actual implementation will use model-specific calculation
    
    # Fine-tuning Parameters
    finetune_epochs: int = 20
    finetune_lr: float = 1e-4
    finetune_patience: int = 3

    # Sequential Layerwise Mode - Fine-tuning Configuration
    # IMPORTANT: Fine-tuning is CRITICAL for vulnerability-based pruning to recover accuracy
    # The vulnerability approach removes weak points, but fine-tuning helps remaining weights adapt
    skip_finetuning_during_initialization: bool = False  # Enable fine-tuning for vulnerability-based pruning
    
    # Checkpointing
    checkpoint_frequency: int = 5
    checkpoint_path: str = "ga_checkpoint.pkl"
    
    # Device and Paths
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"

    # Fault-Aware Pruning (optional)
    fault_impact_filepath: str = None  # Path to layer_fault_impacts.csv (set to enable fault-aware mode)
    
    # Results
    results_dir: str = "ga_results"
    save_pareto_front: bool = True
    
    # Global vs Layer-wise mode
    global_mode: bool = False  # If True, use single global sparsity parameter
    balanced_global: bool = True  # If True, use balanced global agent (prevents extreme imbalances)
    global_balance_factor: float = 20.0  # Allowed variation around target sparsity (±20%)
    
    # Constrained Layer-wise mode - NEW APPROACH
    constrained_layerwise: bool = False  # If True, use layer-wise with global sparsity constraints
    target_global_sparsity: float = 65.0  # Target global sparsity for constrained layer-wise mode
    
    # Sensitivity-Aware Agent Configuration
    use_sensitivity_aware_agents: bool = True  # Use new sensitivity-driven pruning agents
    use_pattern_based_agents: bool = False     # Use 4D architectural pattern agents instead of 19D layerwise

    # MENTOR'S VULNERABILITY-BASED PRUNING (Novel Approach)
    # Prune weights with HIGH sensitivity but LOW magnitude (reliability weak points)
    # This removes vulnerable bottlenecks while preserving strong critical weights
    use_vulnerability_pruning: bool = True  # Enable mentor's vulnerability-based approach (sensitivity/magnitude ratio)
    
    # RL Enhancement Setting
    use_rl_enhancement: bool = False  # Enable RL-guided operator selection and parameter adaptation
    # DEAD CODE - These RL parameters are never accessed in the codebase
    # rl_learning_rate: float = 0.001   # Learning rate for RL components
    # rl_exploration_rate: float = 0.3  # Initial exploration rate for RL agents
    # rl_training_interval: int = 5     # Train RL components every N generations
    
    # Exploration vs Sensitivity Balance
    # DEAD CODE - These ratios are not used in the actual initialization algorithm
    # The real implementation uses create_sensitivity_informed_population() with 4-strategy approach:
    # 20% Conservative, 30% Moderate, 30% Aggressive, 20% Ultra-Conservative (all sensitivity-driven)
    # pure_exploration_ratio: float = 0.30   # 30% pure random exploration
    # sensitivity_guided_ratio: float = 0.10  # Only 10% sensitivity-guided
    # trust_reliability_fitness: bool = True  # Trust fault injection over sensitivity heuristics
    
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
    
    @classmethod
    def create_local_config(cls, score_dir_path: str = "VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for local development."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=10,  # Very small for memory-constrained testing
            num_generations=10,
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels (match benchmark)
            reliability_aggregation="auc_penalty",  # Use AUC-penalty for better stability assessment
            reliability_estimation_reps=8  # Minimal reps for speed
        )
    
    @classmethod
    def create_server_config(cls, score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for server deployment with sequential layerwise approach."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=50,  # Increased for sequential diversity
            num_generations=50,  # Reduced since no finetuning = faster
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels (match benchmark)
            reliability_aggregation="auc_penalty",  # Use AUC-penalty for better stability assessment
            reliability_estimation_reps=10,    # 10 reps per BER level (6 levels now vs 3 before)
            initial_percentile_min=30.0,       # RELAXED: Allow conservative pruning for baseline accuracy
            initial_percentile_max=70.0,       # REDUCED: Prevent model destruction while allowing exploration
            min_overall_sparsity_threshold=25.0,  # RELAXED: Allow meaningful but conservative pruning
            max_overall_sparsity_threshold=80.0,  # INCREASED: Upper bound for exploration
            # LAYERWISE MODE for sequential approach (like successful layerwise method)
            global_mode=False,  # Use layer-wise parameters for sequential exhaustion
            balanced_global=False,  # Not needed in layer-wise mode
            # Sequential Layerwise Mode - No Finetuning During Population Initialization
            skip_finetuning_during_initialization=True,  # Match layerwise method
            # RL ENHANCEMENTS DISABLED for sequential approach
            use_rl_enhancement=False,
            # DEAD CODE - These RL parameters are never accessed in the codebase
            # rl_learning_rate=0.001,
            # rl_exploration_rate=0.3,
            # rl_training_interval=3,
            # DEAD CODE - These exploration ratios are not used in actual initialization
            # pure_exploration_ratio=0.40,   # 40% pure exploration
            # sensitivity_guided_ratio=0.05,  # Only 5% sensitivity guidance
        )
    
    @classmethod
    def create_constrained_layerwise_config(cls, target_sparsity: float = 65.0, score_dir_path: str = "VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for constrained layer-wise approach (solves accuracy problem)."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=50,
            num_generations=50,
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels (match benchmark)
            reliability_aggregation="auc_penalty",
            reliability_estimation_reps=10,  # 10 reps per BER level (6 levels now vs 3 before)
            # CONSTRAINED LAYER-WISE MODE - BEST OF BOTH WORLDS
            global_mode=False,  # Layer-wise for rich search space
            constrained_layerwise=True,  # But with global constraints
            target_global_sparsity=target_sparsity,  # Match benchmarking sparsity
            # Relaxed bounds for layer-wise evolution to achieve baseline accuracy
            initial_percentile_min=25.0,
            initial_percentile_max=75.0, 
            min_overall_sparsity_threshold=25.0,  # Allow conservative pruning regardless of target
            max_overall_sparsity_threshold=max(80.0, target_sparsity * 1.2),  # Upper bound for exploration
        )
    
    @classmethod
    def create_rl_enhanced_config(cls, score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration with RL enhancements enabled."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=50,
            num_generations=75,  # Longer runs for RL learning
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels (match benchmark)
            reliability_aggregation="auc_penalty",
            reliability_estimation_reps=10,  # 10 reps per BER level (6 levels now vs 3 before)    # 15 reps per BER level for faster testing
            initial_percentile_min=50.0,
            initial_percentile_max=75.0,
            min_overall_sparsity_threshold=50.0,
            max_overall_sparsity_threshold=75.0,
            # RL enhancements
            use_rl_enhancement=True,
            # DEAD CODE - These RL parameters are never accessed in the codebase
            # rl_learning_rate=0.001,
            # rl_exploration_rate=0.3,
            # rl_training_interval=3,
            # DEAD CODE - These exploration ratios are not used in actual initialization
            # pure_exploration_ratio=0.40,   # INCREASED pure exploration
            # sensitivity_guided_ratio=0.05,  # MINIMAL sensitivity guidance
            # trust_reliability_fitness=True,  # DEAD CODE - never accessed
        )
    
    @classmethod 
    def create_pure_exploration_config(cls, score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration with MAXIMUM exploration, minimal sensitivity constraints."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=50,
            num_generations=60,
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels (match benchmark)
            reliability_aggregation="auc_penalty",
            reliability_estimation_reps=10,  # 10 reps per BER level (6 levels now vs 3 before)
            initial_percentile_min=10.0,  # FULL range
            initial_percentile_max=95.0,  # FULL range
            min_overall_sparsity_threshold=25.0,  # FIXED: Allow exploration with lower sparsity
            # DEAD CODE - These exploration ratios are not used in actual initialization
            # pure_exploration_ratio=0.60,   # 60% pure exploration
            # sensitivity_guided_ratio=0.05, # Only 5% sensitivity
            # trust_reliability_fitness=True,  # DEAD CODE - never accessed
            # Aggressive mutation for discovery
            mutation_prob=0.8,
            mutation_sigma=25.0,
        )
    
    @classmethod
    def create_pattern_based_config(cls, score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for pattern-based 4D architecture-aware pruning (MUCH faster)."""
        return cls(
            score_dir_path=score_dir_path,
            population_size=30,  # Single value - reduced from 50 (4D space is much smaller)
            num_generations=40,  # Reduced from 50 (faster convergence expected)
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels (match benchmark)
            reliability_aggregation="auc_penalty",
            reliability_estimation_reps=10,  # 10 reps per BER level (6 levels now vs 3 before)
            # PATTERN-BASED MODE: 4D search space instead of 19D
            global_mode=False,  # Still layerwise, but pattern-driven
            use_pattern_based_agents=True,  # Enable 4D architectural patterns
            use_sensitivity_aware_agents=False,  # Disable 19D sensitivity agents
            # Pattern bounds (for 4D individuals: [base_sparsity, conv_bias, fc_bias, protection])
            initial_percentile_min=25.0,  # base_sparsity range
            initial_percentile_max=75.0,  # base_sparsity range
        )

    @classmethod
    def create_fault_aware_config(cls, fault_impact_filepath: str = "layer_fault_impacts.csv",
                                  score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"):
        """Create configuration for fault-aware reliability-optimized pruning."""
        return cls(
            score_dir_path=score_dir_path,
            fault_impact_filepath=fault_impact_filepath,  # Enable fault-aware mode
            population_size=50,
            num_generations=50,
            reliability_ber_levels=(5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3),  # 6 BER levels
            reliability_aggregation="auc_penalty",
            reliability_estimation_reps=10,
            # Layer-wise mode for fault-aware initialization
            global_mode=False,
            use_sensitivity_aware_agents=True,
            # Relaxed bounds to allow fault-aware initialization to guide pruning
            initial_percentile_min=25.0,
            initial_percentile_max=75.0,
            min_overall_sparsity_threshold=25.0,
        )