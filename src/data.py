from __future__ import annotations
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    # Matches notebook exactly (note: Resize((224,224)) not crop)
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf

def build_dataloaders(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(image_size)

    train_ds = Flowers102(root=root, split="train", download=download, transform=train_tf)
    val_ds   = Flowers102(root=root, split="val",   download=download, transform=eval_tf)
    test_ds  = Flowers102(root=root, split="test",  download=download, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
