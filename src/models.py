from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

def build_model(model_name: str, num_classes: int) -> nn.Module:
    name: str = model_name.lower()

    if name == "mobilenet_v3_small":
        m: nn.Module = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise ValueError(f"Unknown model_name: {model_name}")

def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = trainable
    else:
        # fallback: train/untrain all
        for p in model.parameters():
            p.requires_grad = trainable

def save_checkpoint(path: str, model: nn.Module, model_name: str, num_classes: int, image_size: int) -> None:
    payload = {
        "model_name": model_name,
        "num_classes": num_classes,
        "image_size": image_size,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)

def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt: dict = torch.load(path, map_location=device)
    model: nn.Module = build_model(ckpt["model_name"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
