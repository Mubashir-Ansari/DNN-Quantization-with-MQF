import torch
import os
import sys

# Add path to model loaders
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, 'quantization_framework'))

from quantization_framework.models.model_loaders import load_model

def sanitize_model(checkpoint_path, output_path):
    """
    Loads a quanto-quantized model and saves it as a standard PyTorch state_dict.
    This version can be loaded on ANY machine without needing the correct quanto version.
    """
    print(f"--- Model Sanitizer ---")
    print(f"Loading: {checkpoint_path}")
    
    # This will use your local 'working' quanto to load and dequantize
    model = load_model('alexnet', checkpoint_path=checkpoint_path)
    
    # Extract the plain state_dict
    plain_state_dict = model.state_dict()
    
    # Save it as a standard .pth
    print(f"Saving plain state_dict to: {output_path}")
    torch.save(plain_state_dict, output_path)
    print("Success! Copy this new file to your VM.")

if __name__ == "__main__":
    sanitize_model('models/qalex-0-7.pth', 'models/qalex-0-7-clean.pth')
