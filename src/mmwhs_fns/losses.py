from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-5) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def mse_consistency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target.detach())


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    mask = (confidence > threshold).float()
    return (mask * F.mse_loss(prediction, target.detach(), reduction="none")).mean()


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def discriminator_adversarial_loss(fake_logits: torch.Tensor, real_logits: torch.Tensor, smooth: float = 0.9) -> torch.Tensor:
    fake_targets = torch.zeros_like(fake_logits)
    real_targets = torch.ones_like(real_logits) * smooth
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    return 0.5 * (fake_loss + real_loss)


def sigmoid_rampup(current: int | float, rampup_length: int | float) -> float:
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(float(current), float(rampup_length)))
    phase = 1.0 - current / float(rampup_length)
    return float(math.exp(-5.0 * phase * phase))


def consistency_weight(step: int, consistency: float, rampup: float) -> float:
    return consistency * sigmoid_rampup(step, rampup)


@torch.no_grad()
def class_weights_from_dataset(
    dataset,
    num_classes: int,
    device: torch.device,
    distributed: bool = False,
    max_weight: float = 10.0,
    min_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.utils.data import DataLoader, DistributedSampler

    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=0)
    for _, labels, _, _ in loader:
        labels = labels.to(device).view(-1)
        hist = torch.bincount(labels, minlength=num_classes).double()
        counts += hist[:num_classes]

    if distributed:
        import torch.distributed as dist

        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    freqs = counts / counts.sum().clamp_min(1.0)
    weights = torch.log((1.0 / (freqs + 1e-6)) + 1.0)
    weights = torch.clamp(weights, min=min_weight, max=max_weight).float()
    return counts.float(), weights
