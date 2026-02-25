import torch
import torch.nn as nn
import timm

def levit_cifar(num_classes=10, distillation=False):
    """
    Creates a LeViT model using timm.
    We use 'levit_384' as the base as seen in user's scripts.
    """
    try:
        # Create LeViT-384
        model = timm.create_model('levit_384', pretrained=False, num_classes=num_classes)
        
        # Patch head to match user's checkpoint (Simple Linear vs Timm's BN+Linear)
        # ckpt has: head.weight, head.bias
        # timm has: head.bn.*, head.linear.*
        # Patch head to match user's checkpoint
        if hasattr(model, 'num_features'):
            in_features = model.num_features
        else:
            in_features = model.head.linear.in_features
            
        model.head = nn.Linear(in_features, num_classes)
        model.head_dist = nn.Identity()
        
        print(f"DEBUG: LeViT initialized. num_classes={num_classes}, in_features={in_features}")
        
        # KEY FIX: Override forward to skip distillation
        def new_forward(self, x):
            # Run Backbone
            x = self.forward_features(x)
            
            # Adaptive Pooling Logic
            if x.ndim == 4:
                x = x.mean([-2, -1])
            elif x.ndim == 3:
                x = x.mean(1)
                
            # Run Head
            x = self.head(x)
            return x
            
        import types
        model.forward = types.MethodType(new_forward, model)
        
        return model
    except Exception as e:
        print(f"Error creating LeViT model via timm: {e}")
        # Fallback or re-raise depending on strictness. 
        # For now, let's create a generic LeViT if timm fails or specific variant missing?
        # Actually, if timm is missing, we are in trouble anyway.
        raise e
