"""VGG Model definitions and utilities."""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


__all__ = [
    "VGG",
    "vgg11_bn",
    "vgg13_bn", 
    "vgg16_bn",
    "vgg19_bn",
]


class VGG(nn.Module):
    """VGG network architecture."""
    
    def __init__(self, features: nn.Module, num_classes: int = 10, init_weights: bool = True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """Initialize model weights using standard initialization schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_prunable_layers(self) -> Dict[str, torch.Tensor]:
        """Get dictionary of prunable layer parameters."""
        prunable_params = {}
        for name, param in self.named_parameters():
            if ('features' in name or 'classifier' in name) and 'weight' in name:
                prunable_params[name] = param
        return prunable_params
    
    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about each layer."""
        layer_info = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_info[name] = {
                    'type': type(module).__name__,
                    'in_features': getattr(module, 'in_features', getattr(module, 'in_channels', None)),
                    'out_features': getattr(module, 'out_features', getattr(module, 'out_channels', None)),
                    'kernel_size': getattr(module, 'kernel_size', None),
                    'parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
                }
        return layer_info


def make_layers(cfg: list, batch_norm: bool = False) -> nn.Sequential:
    """Create VGG layers from configuration."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# VGG configurations
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, 
         device: torch.device, **kwargs) -> VGG:
    """Create VGG model with optional pretrained weights."""
    if pretrained:
        kwargs["init_weights"] = False
    
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    
    if pretrained:
        # Try multiple possible paths for the pretrained model
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "VGG11", f"{arch}.pt"),
            os.path.join("VGG11", f"{arch}.pt"),
            f"{arch}.pt"
        ]
        
        model_loaded = False
        for pt_file_path in possible_paths:
            if os.path.exists(pt_file_path):
                try:
                    state_dict = torch.load(pt_file_path, map_location=device)
                    model.load_state_dict(state_dict)
                    model_loaded = True
                    print(f"Loaded pretrained model from: {pt_file_path}")
                    break
                except Exception as e:
                    print(f"Failed to load from {pt_file_path}: {e}")
                    continue
        
        if not model_loaded:
            raise FileNotFoundError(
                f"Pretrained model file not found. Tried paths: {possible_paths}"
            )
    
    return model.to(device)


def vgg11_bn(pretrained: bool = False, progress: bool = True, 
             device: torch.device = torch.device('cuda'), **kwargs) -> VGG:
    """VGG 11-layer model with batch normalization."""
    return _vgg("vgg11_bn", "A", True, pretrained, progress, device, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, 
             device: torch.device = torch.device('cuda'), **kwargs) -> VGG:
    """VGG 13-layer model with batch normalization.""" 
    return _vgg("vgg13_bn", "B", True, pretrained, progress, device, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True,
             device: torch.device = torch.device('cuda'), **kwargs) -> VGG:
    """VGG 16-layer model with batch normalization."""
    return _vgg("vgg16_bn", "D", True, pretrained, progress, device, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True,
             device: torch.device = torch.device('cuda'), **kwargs) -> VGG:
    """VGG 19-layer model with batch normalization."""
    return _vgg("vgg19_bn", "E", True, pretrained, progress, device, **kwargs)


def create_model_from_config(model_config) -> VGG:
    """Create VGG model from ModelConfig."""
    if model_config.model_name == "vgg11_bn":
        return vgg11_bn(
            pretrained=model_config.pretrained,
            device=model_config.device,
            num_classes=model_config.num_classes
        )
    elif model_config.model_name == "vgg13_bn":
        return vgg13_bn(
            pretrained=model_config.pretrained,
            device=model_config.device,
            num_classes=model_config.num_classes
        )
    elif model_config.model_name == "vgg16_bn":
        return vgg16_bn(
            pretrained=model_config.pretrained,
            device=model_config.device,
            num_classes=model_config.num_classes
        )
    elif model_config.model_name == "vgg19_bn":
        return vgg19_bn(
            pretrained=model_config.pretrained,
            device=model_config.device,
            num_classes=model_config.num_classes
        )
    else:
        raise ValueError(f"Unsupported model: {model_config.model_name}")