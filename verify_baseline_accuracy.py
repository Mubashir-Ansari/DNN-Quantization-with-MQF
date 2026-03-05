"""
VERIFY BASELINE ACCURACY
Standalone script to identify the real baseline accuracy of 
the 8-bit quantized AlexNet model from the checkpoint.
"""

import os
import sys
import torch
from pathlib import Path

# Add paths for framework access
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'quantization_framework'))

from quantization_framework.models.model_loaders import load_model
from quantization_framework.evaluation.pipeline import (
    get_fashionmnist_dataloader,
    evaluate_accuracy
)

def verify_baseline():
    # 1. Configuration (Matching the POC)
    CONFIG = {
        'model_name': 'alexnet',
        'checkpoint_path': 'models/qalex-0-7.pth',
        'num_classes': 10,
        'input_size': 227,
        'dataset': 'fashionmnist',
        'data_path': 'data/fashionmnist',
        'batch_size': 128,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n" + "="*80)
    print(f"VERIFYING BASELINE: {CONFIG['model_name']} on {CONFIG['dataset']}")
    print("="*80)
    print(f"Device: {CONFIG['device']}")
    
    # 2. Load the 8-bit Model (Before any MQF optimization)
    # This uses the shim in model_loaders to restore the 8-bit weights
    model = load_model(
        CONFIG['model_name'],
        checkpoint_path=CONFIG['checkpoint_path'],
        num_classes=CONFIG['num_classes']
    )
    model.to(CONFIG['device'])
    model.eval()
    
    print(f"✓ Loaded weights from {CONFIG['checkpoint_path']}")
    
    # 3. Load Validation Data
    dataloader = get_fashionmnist_dataloader(
        train=False,
        input_size=CONFIG['input_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        data_path=CONFIG['data_path']
    )
    print(f"✓ Data loaded from {CONFIG['data_path']}")
    
    # 4. Run Accuracy Test
    print("\n[EVALUATION] Computing Baseline Accuracy...")
    # Using 1000 samples for a stable result, or change to len(dataloader.dataset) for full test
    accuracy = evaluate_accuracy(model, dataloader, device=CONFIG['device'], max_samples=1000)
    
    print("\n" + "*"*30)
    print(f"REAL BASELINE ACCURACY: {accuracy:.2f}%")
    print("*"*30 + "\n")

if __name__ == "__main__":
    verify_baseline()
