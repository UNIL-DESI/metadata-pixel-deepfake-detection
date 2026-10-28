"""ImageNet CNN backbones used as frozen feature extractors (PyTorch).

The cluster image ships PyTorch with CUDA, so the CNN features use
torchvision instead of Keras. Each model returns one globally average-pooled
vector per image (1280 values for both MobileNetV2 and EfficientNet-B0).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import models

# torchvision's ImageNet weights expect RGB in [0, 1] normalised like this.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_BACKBONES = {
    "mobilenetv2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "efficientnetb0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


class PooledBackbone(nn.Module):
    """Convolutional trunk followed by global average pooling."""

    def __init__(self, trunk: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.trunk(x)), 1)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cnn_model(model_name: str, weights: str | None = "imagenet",
                  device: torch.device | None = None) -> nn.Module:
    """Return a frozen, eval-mode feature extractor on `device`."""
    key = model_name.lower()
    if key not in _BACKBONES:
        raise ValueError(f"Unsupported CNN model: {model_name}")
    builder, imagenet_weights = _BACKBONES[key]
    if weights not in (None, "imagenet"):
        raise ValueError(f"Unsupported CNN weights: {weights}")
    network = builder(weights=imagenet_weights if weights == "imagenet" else None)
    model = PooledBackbone(network.features)
    model.eval().requires_grad_(False)
    return model.to(device or get_device())


def preprocess_batch(images: list[np.ndarray] | np.ndarray,
                     device: torch.device) -> torch.Tensor:
    """RGB uint8 images of shape (N, H, W, 3) -> normalised NCHW tensor."""
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(images)))
    batch = batch.to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


@torch.no_grad()
def embed_images(model: nn.Module, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Pooled features for a batch of RGB uint8 images, as float32 numpy."""
    device = next(model.parameters()).device
    return model(preprocess_batch(images, device)).float().cpu().numpy()
