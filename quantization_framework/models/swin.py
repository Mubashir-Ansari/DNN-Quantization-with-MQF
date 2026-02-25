import torch
import torch.nn as nn
import timm

def swin_tiny_patch4_window7_224(num_classes=10):
    """
    Creates a Swin Transformer using timm.
    Matches user's 'swin_tiny_patch4_window7_224'.
    """
    try:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        
        # Proactive fix based on checkpoint keys 'head.0' and 'head.1'
        # This implies a Sequential head (Linear -> Linear or Linear -> Act -> Linear)
        # We'll replace the head with a Sequential to match the keys.
        # Dimensions are tricky without inspecting, but we assume preservation or standard projection.
        # Let's initialize with dummy layers; load_state_dict will fill weights.
        if hasattr(model, 'head'):
             # Create a 2-layer head matching the keys 0 and 1
             # We use the current in_features
             in_features = model.head.in_features if isinstance(model.head, nn.Linear) else 768
             
             # Checkpoint analysis:
             # head.0.weight shape is [768] -> This is LayerNorm, not Linear!
             # head.1.weight shape is [100, 768] -> This is Linear(768, 100). 
             # It seems this model was trained on CIFAR-100 (100 classes).
             
             print("DEBUG: Patching Swin head to Sequential(LayerNorm, Linear(100)) to match checkpoint.")
             model.head = nn.Sequential(
                nn.LayerNorm(in_features), # head.0
                nn.Linear(in_features, 100)  # head.1 (CIFAR-100 dimensions)
             )
        

        
        # Monkey-patch forward to ensure pooling happens (just like LeViT)
        def new_forward(self, x):
            x = self.forward_features(x)
            
            # Adaptive Pooling Logic
            if x.ndim == 4: 
                # Check for Channels Last (B, H, W, C) - Common in Swin
                if x.shape[-1] == in_features: 
                    # Pool spatial dims (H, W) which are 1 and 2
                    x = x.mean([1, 2])
                else: 
                    # Channels First (B, C, H, W)
                    x = x.mean([-2, -1])
            elif x.ndim == 3: # (B, N, C)
                x = x.mean(1)
            
            x = self.head(x)
            return x

        import types
        model.forward = types.MethodType(new_forward, model)
        
        print(f"DEBUG: Swin initialized. num_classes={num_classes} (patched for C100 if needed)")
        return model
    except Exception as e:
        print(f"Error creating Swin model via timm: {e}")
        raise e
