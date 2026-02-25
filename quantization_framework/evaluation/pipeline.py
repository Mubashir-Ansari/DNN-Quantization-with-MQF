"""
Data Loading and Evaluation Pipeline
Features:
1. Robust CIFAR-10/100 loaders (backward compatible with hardware_aware_search).
2. GTSRB Loader with Label Mapping Fix (handles ImageFolder alphabetical sorting vs Test.csv).
3. GTSRB Internal Train/Val Split Support (for models trained on Train folder split).
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from PIL import Image
import random
import numpy as np
import os


# ============================================================================
# CIFAR-10 Data Loading
# ============================================================================

def get_cifar10_dataloader(train=False, batch_size=128, input_size=32, num_workers=0, data_path='./data', split=None):
    """
    Load CIFAR-10 dataset. Supports 'train' arg for backward compatibility.
    """
    # handle split vs train arg
    if split is not None:
        train = (split == 'train')
        
    transforms_list = []
    if input_size and input_size != 32:
        transforms_list.append(transforms.Resize((input_size, input_size)))
        
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])
    transform = transforms.Compose(transforms_list)
    
    dataset = datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


# ============================================================================
# CIFAR-100 Data Loading
# ============================================================================

def get_cifar100_dataloader(train=False, batch_size=128, input_size=32, num_workers=0, data_path='./data', split=None):
    """
    Load CIFAR-100 dataset. Supports 'train' arg for backward compatibility.
    """
    if split is not None:
        train = (split == 'train')

    transforms_list = []
    if input_size and input_size != 32:
        transforms_list.append(transforms.Resize((input_size, input_size)))

    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform = transforms.Compose(transforms_list)
    
    dataset = datasets.CIFAR100(root=data_path, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


# ============================================================================
# Fashion-MNIST Data Loading
# ============================================================================

class FashionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        label, image = [], []
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype("float32")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform is not None:
            pil_image = Image.fromarray(np.uint8(image))
            # MQF FIX: Removed hardcoded RGB conversion. 
            # AlexNet/FashionMNIST models expect 1-channel grayscale.
            image = self.transform(pil_image)
        return image, label

def get_fashionmnist_dataloader(train=False, batch_size=128, input_size=227, num_workers=0, data_path='./data', split=None):
    """
    Load Fashion-MNIST dataset. Checks for CSV format first to match user's baseline.
    """
    if split is not None:
        train = (split == 'train')

    # Note: We use 3-channel normalization if we know the model is RGB
    # But for simplicity, we'll keep the single channel norm and the dataset will convert
    transforms_list = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    transform = transforms.Compose(transforms_list)
    
    # Try to find user's CSV first
    csv_file = "data/fashionmnist/fashion-mnist_test.csv" if not train else "data/fashionmnist/fashion-mnist_train.csv"
    if os.path.exists(csv_file):
        print(f"[DATA] Loading Fashion-MNIST from CSV: {csv_file}")
        dataset = FashionDataset(csv_file, transform=transform)
    else:
        print(f"[DATA] CSV not found at {csv_file}, falling back to torchvision.datasets.FashionMNIST")
        dataset = datasets.FashionMNIST(root=data_path, train=train, download=True, transform=transform)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


# ============================================================================
# GTSRB Data Loading (With Label Mapping Fix + Train/Val Split Support)
# ============================================================================

class GTSRBMappingDataset(Dataset):
    """
    GTSRB Dataset that maps Test.csv ClassIds to model indices.
    This fixes the issue where ImageFolder sorts folders alphabetically (0, 1, 10...)
    mismatching the sequential Test.csv IDs (0, 1, 2...).
    """
    def __init__(self, csv_file, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.class_to_idx = class_to_idx # Map "ClassStr" -> ModelIndex

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Image Path
        img_path = os.path.join(self.root_dir, row['Path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
             # Try replacing '/' with os.sep just in case
             img_path = os.path.join(self.root_dir, row['Path'].replace('/', os.sep))
             try:
                 image = Image.open(img_path).convert('RGB')
             except FileNotFoundError:
                 # Last resort check absolute/relative issues
                 raise FileNotFoundError(f"Image not found: {img_path}")

        if self.transform:
            image = self.transform(image)

        # Label Mapping
        true_class_id_str = str(int(row['ClassId']))
        
        # Look up what index ImageFolder assigned to folder '10'
        if true_class_id_str in self.class_to_idx:
            label = self.class_to_idx[true_class_id_str]
        else:
            # Fallback (should not happen if Train/ contains all classes)
            label = int(row['ClassId'])

        return image, label


class GTSRBTrainValSplit:
    """
    Create reproducible train/val split from GTSRB Train folder.
    Use this when model was trained on internal split, not Test.csv.
    """
    @staticmethod
    def get_split(dataset, val_ratio=0.2, seed=42):
        """
        Split dataset into train and validation sets.
        
        Args:
            dataset: Full training dataset
            val_ratio: Proportion for validation (default: 0.2 = 20%)
            seed: Random seed for reproducibility (must match training!)
        
        Returns:
            train_subset, val_subset
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_ratio * dataset_size))
        
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        print(f"[GTSRB Split] Total: {dataset_size}, Train: {len(train_indices)}, Val: {len(val_indices)}")
        
        return train_subset, val_subset


def get_gtsrb_dataloader(train=False, batch_size=128, input_size=224, num_workers=0, 
                         data_path='./data', split=None, use_train_val_split=False, 
                         val_ratio=0.2, seed=42):
    """
    Load GTSRB dataset with flexible split options.
    
    Args:
        train: If True, return training data
        batch_size: Batch size for DataLoader
        input_size: Image input size (default: 224)
        num_workers: Number of workers for DataLoader
        data_path: Path to GTSRB data root
        split: Alternative way to specify train/test ('train' or None)
        use_train_val_split: If True, create validation split from Train folder 
                             instead of using Test.csv (use when model was trained 
                             on internal split). DEFAULT: False for backward compatibility.
        val_ratio: Validation split ratio when use_train_val_split=True (default: 0.2)
        seed: Random seed for reproducible splits when use_train_val_split=True (default: 42)
    
    Returns:
        DataLoader for the requested split
    
    Usage Examples:
        # Standard Test.csv evaluation (default, backward compatible)
        loader = get_gtsrb_dataloader(train=False)
        
        # Internal validation split (for ResNet trained on Train folder split)
        loader = get_gtsrb_dataloader(train=False, use_train_val_split=True, val_ratio=0.2, seed=42)
    """
    if split is not None:
        train = (split == 'train')
        
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 1. Determine Train Path (needed for mapping in both train and test cases)
    train_dir = os.path.join(data_path, 'Train')
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_path, 'train')
        if not os.path.exists(train_dir):
             raise FileNotFoundError(f"GTSRB Train folder not found at {train_dir}")

    # 2. Extract Label Mapping from ImageFolder structure
    dummy_dataset = datasets.ImageFolder(train_dir)
    class_to_idx = dummy_dataset.class_to_idx
    
    # 3. MODE 1: Internal Train/Val Split (NEW - for models trained on Train folder split)
    if use_train_val_split:
        print(f"[GTSRB] Using internal train/val split from Train folder")
        print(f"        val_ratio={val_ratio}, seed={seed}")
        print(f"        NOTE: This matches training split, not Test.csv!")
        
        full_dataset = datasets.ImageFolder(train_dir, transform=transform)
        train_subset, val_subset = GTSRBTrainValSplit.get_split(full_dataset, val_ratio, seed)
        
        if train:
            return DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            return DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 4. MODE 2: Standard Train/Test Split (ORIGINAL - backward compatible)
    if train:
        # Training: use full ImageFolder
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    else:
        # Testing: use Test.csv with proper label mapping
        csv_path = os.path.join(data_path, 'Test.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(data_path, 'test.csv')
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"GTSRB Test.csv not found at {csv_path}.\n"
                f"If your model was trained on Train folder split (not Test.csv), "
                f"use: get_gtsrb_dataloader(train=False, use_train_val_split=True)"
            )
             
        dataset = GTSRBMappingDataset(
            csv_file=csv_path,
            root_dir=data_path,
            class_to_idx=class_to_idx,
            transform=transform
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ============================================================================
# Metrics & Helpers
# ============================================================================

def evaluate_accuracy(model, dataloader, device='cpu', max_samples=None):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on ('cpu' or 'cuda')
        max_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        Accuracy as percentage (0-100)
    """
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if max_samples and total >= max_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    if total == 0: return 0.0
    return 100.0 * correct / total

def compute_model_size(model):
    """Compute model size in MB"""
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_bytes = param_size + buffer_size
    return size_bytes / (1024 ** 2)

def count_parameters(model):
    """Count total number of parameters"""
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, dataloader, device='cpu', num_batches=10):
    """Measure average inference time per batch"""
    import time
    model.eval()
    model = model.to(device)
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches: break
            images = images.to(device)
            if i == 0:
                _ = model(images)
                continue
            start = time.time()
            _ = model(images)
            if device == 'cuda': torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)
    avg_time = sum(times) / len(times) if times else 0
    return avg_time
