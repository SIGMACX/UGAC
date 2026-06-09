from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, models: dict[str, nn.Module], optimizers: dict[str, torch.optim.Optimizer], device):
    path = Path(path)
    if not path.exists():
        return 0, 0.0, 0

    checkpoint = torch.load(path, map_location=device)
    for name, model in models.items():
        key = f"state_dict_{name}"
        if key in checkpoint:
            unwrap_model(model).load_state_dict(checkpoint[key])
    for name, optimizer in optimizers.items():
        if name in checkpoint:
            optimizer.load_state_dict(checkpoint[name])
    return checkpoint.get("epoch", 0), checkpoint.get("best_dice", 0.0), checkpoint.get("no_improve_epoch", 0)
