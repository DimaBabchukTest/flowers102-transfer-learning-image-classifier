from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "mobilenet_v3_large"
    num_classes: int = 102

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    warmup_epochs: int = 2
    finetune_epochs: int = 10

    lr_warmup: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-5

    dropout: float = 0.3  # will be used only if model supports it easily
    patience: int = 4
    min_delta: float = 1e-3

    device: str = "cuda"
    seed: int = 42

    save_dir: str = "artifacts"
    run_name: str = "flowers102_run"
