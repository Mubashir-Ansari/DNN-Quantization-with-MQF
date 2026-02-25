"""Data loading and preprocessing utilities."""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from typing import Tuple, Optional
from config.model_config import ModelConfig


def get_cifar10_transforms(config: ModelConfig, train: bool = True) -> T.Compose:
    """Get CIFAR-10 transforms based on configuration."""
    if train and config.use_data_augmentation:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(config.normalize_mean, config.normalize_std),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(config.normalize_mean, config.normalize_std),
        ])
    return transform


def get_data_loaders(config: ModelConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from configuration."""
    
    # Training data loader
    train_transform = get_cifar10_transforms(config, train=True)
    train_dataset = CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0 and torch.cuda.is_available()
    )
    
    # Validation data loader
    val_transform = get_cifar10_transforms(config, train=False)
    val_dataset = CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0 and torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def get_single_dataloader(config: ModelConfig, train: bool = True, 
                         batch_size: Optional[int] = None, 
                         num_workers: Optional[int] = None) -> DataLoader:
    """Create a single data loader with optional parameter overrides."""
    
    transform = get_cifar10_transforms(config, train=train)
    dataset = CIFAR10(
        root=config.data_root,
        train=train,
        download=True,
        transform=transform
    )
    
    # Use provided parameters or fall back to config
    batch_size = batch_size or config.batch_size
    num_workers = num_workers if num_workers is not None else config.num_workers
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0 and torch.cuda.is_available()
    )


def get_evaluation_dataloader(config: ModelConfig, num_workers: int = 0) -> DataLoader:
    """Create a data loader optimized for evaluation (no multiprocessing issues)."""
    
    transform = get_cifar10_transforms(config, train=False)
    dataset = CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,  # Often 0 to avoid multiprocessing issues
        drop_last=config.drop_last,
        pin_memory=False  # Disabled for evaluation
    )


class DatasetInfo:
    """Information about the dataset."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._train_size = None
        self._val_size = None
    
    @property
    def train_size(self) -> int:
        """Get training dataset size."""
        if self._train_size is None:
            train_dataset = CIFAR10(root=self.config.data_root, train=True, download=True)
            self._train_size = len(train_dataset)
        return self._train_size
    
    @property
    def val_size(self) -> int:
        """Get validation dataset size."""
        if self._val_size is None:
            val_dataset = CIFAR10(root=self.config.data_root, train=False, download=True)
            self._val_size = len(val_dataset)
        return self._val_size
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self.config.num_classes
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Get input shape (C, H, W)."""
        return (3, 32, 32)
    
    def get_class_names(self) -> list:
        """Get CIFAR-10 class names."""
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]