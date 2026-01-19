# src/infer.py
from __future__ import annotations

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import models


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float]  = (0.229, 0.224, 0.225)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()

    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise ValueError(f"Unknown model_name: {model_name}")


def build_eval_transform(image_size: int) -> transforms.Compose:
    # MATCHES your notebook eval_tf: Resize((224,224)) + Normalize
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)

    model_name: str = ckpt["model_name"]
    num_classes: int = int(ckpt["num_classes"])
    model: nn.Module = build_model(model_name, num_classes)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()
def predict_pil_image(
    model: nn.Module,
    image: Image.Image,
    tfm: transforms.Compose,
    device: torch.device
) -> Tuple[int, torch.Tensor]:
    """
    Returns:
      - predicted class index (int)
      - probabilities tensor shape (num_classes,)
    """
    x = tfm(image).unsqueeze(0).to(device)  # (1,3,H,W)
    logits = model(x)                      # (1,C)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()  # (C,)
    pred_idx = int(torch.argmax(probs).item())
    return pred_idx, probs
