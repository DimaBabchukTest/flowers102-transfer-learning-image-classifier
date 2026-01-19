from __future__ import annotations
import os
import random
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig
from data import build_dataloaders
from models import build_model, set_backbone_trainable, save_checkpoint

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs: int = 0

    def step(self, val_loss: float) -> bool:
        if self.best is None:
            self.best = val_loss
            return False
        improved = (self.best - val_loss) > self.min_delta
        if improved:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)

def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)

def main() -> None:
    cfg = TrainConfig()  # edit config.py or replace with argparse later
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")

    train_loader, val_loader, test_loader = build_dataloaders(
        root="./data",
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    model = build_model(cfg.model_name, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # ---- Stage 1: warmup head ----
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_warmup,
        weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.warmup_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[warmup {epoch+1}/{cfg.warmup_epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

    # ---- Stage 2: finetune all ----
    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    best_val_loss: float | None = None

    for epoch in range(cfg.finetune_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Save best by VAL LOSS (matches notebook best-practice choice)
        if best_val_loss is None or va_loss < best_val_loss:
            best_val_loss = va_loss
            save_checkpoint(best_path, model, cfg.model_name, cfg.num_classes, cfg.image_size)

        print(f"[finetune {epoch+1}/{cfg.finetune_epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if stopper.step(va_loss):
            print(f"Early stopping at epoch {epoch+1} (no val_loss improvement).")
            break

    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"[final] test_loss={te_loss:.4f} test_acc={te_acc:.4f}")
    print("Saved best model to:", best_path)
    print("Config:", asdict(cfg))

if __name__ == "__main__":
    main()
