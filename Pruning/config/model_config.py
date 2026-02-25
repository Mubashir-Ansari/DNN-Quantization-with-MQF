"""Model and Dataset Configuration."""

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for models and datasets."""
    
    # Model Parameters
    model_name: str = "vgg11_bn"
    num_classes: int = 10
    pretrained: bool = True
    model_path: str = "VGG11/vgg11_bn.pt"
    score_dir_path: str = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg_weight_grades/vgg_weight_sensitivity_score/"
    
    # Dataset Parameters
    dataset_name: str = "CIFAR10"
    data_root: str = "datasets/cifar10_data"
    
    # Data Augmentation
    use_data_augmentation: bool = True
    normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    normalize_std: Tuple[float, float, float] = (0.2471, 0.2435, 0.2616)
    
    # DataLoader Parameters
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    
    # Training Parameters
    optimizer: str = "SGD"  # or "Adam"
    learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Evaluation Parameters
    dummy_input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)
    latency_warmup_runs: int = 10
    latency_measurement_runs: int = 100
    
    # Device Configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
    
    @classmethod
    def create_vgg11_cifar10_config(cls):
        """Create standard VGG-11 CIFAR-10 configuration."""
        return cls(
            model_name="vgg11_bn",
            dataset_name="CIFAR10",
            num_classes=10,
            pretrained=True
        )
    
    @classmethod
    def create_local_config(cls):
        """Create configuration optimized for local development."""
        return cls(
            batch_size=16,  # Very small batch size for limited memory
            num_workers=0,  # Avoid multiprocessing issues on Windows
            pin_memory=False,
            use_mixed_precision=False
        )
    
    @classmethod
    def create_server_config(cls):
        """Create configuration optimized for server deployment."""
        return cls(
            batch_size=128,
            num_workers=4,  # More workers for faster data loading
            pin_memory=True,
            use_mixed_precision=True  # Enable for faster training
        )
    
    def get_dummy_input(self) -> torch.Tensor:
        """Get dummy input tensor for latency measurement."""
        return torch.randn(*self.dummy_input_shape, device=self.device)