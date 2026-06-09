from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class UncertaintyGuidedAdaptiveFuser(nn.Module):
    def __init__(
        self,
        method: str = "uncertainty_fuse",
        weights: tuple[float, float] | list[float] = (0.5, 0.5),
        patch_size: int = 17,
        entropy_threshold: float = 0.5,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.method = method
        self.weights = tuple(float(w) for w in weights)
        self.entropy_threshold = entropy_threshold
        self.num_classes = num_classes
        self.conv = nn.Conv2d(
            num_classes,
            num_classes,
            kernel_size=patch_size,
            padding=patch_size // 2,
            groups=num_classes,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / (patch_size * patch_size))
        self.conv.weight.requires_grad_(False)

    def fuse(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        logits1 = logits1.detach()
        logits2 = logits2.detach()
        if self.method == "average":
            return 0.5 * (logits1 + logits2)
        if self.method == "weighted_average":
            w1, w2 = self.weights
            return (w1 * logits1 + w2 * logits2) / (w1 + w2)
        if self.method == "confidence":
            return self._confidence(logits1, logits2)
        if self.method == "uncertainty_fuse":
            return self._uncertainty_fuse(logits1, logits2)
        raise ValueError(f"Unknown fusion method: {self.method}")

    def _confidence(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        prob1 = F.softmax(logits1, dim=1)
        prob2 = F.softmax(logits2, dim=1)
        mask = prob1.max(dim=1, keepdim=True).values > prob2.max(dim=1, keepdim=True).values
        return torch.where(mask, logits1, logits2)

    def _uncertainty_fuse(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        prob1 = F.softmax(logits1, dim=1)
        prob2 = F.softmax(logits2, dim=1)
        max_entropy = math.log(self.num_classes)
        entropy1 = -torch.sum(prob1 * torch.log(prob1 + 1e-8), dim=1) / max_entropy
        entropy2 = -torch.sum(prob2 * torch.log(prob2 + 1e-8), dim=1) / max_entropy

        current_entropy = 0.5 * (entropy1.mean() + entropy2.mean())
        corrected1 = logits1
        corrected2 = logits2

        if current_entropy.item() > self.entropy_threshold:
            combined = torch.cat([entropy1.flatten(), entropy2.flatten()])
            uncertainty_threshold = combined.mean() + combined.std()
            high1 = entropy1 > uncertainty_threshold
            high2 = entropy2 > uncertainty_threshold
            both = high1 & high2

            corrected1 = torch.where((high1 & ~high2).unsqueeze(1), logits2, corrected1)
            corrected2 = torch.where((high2 & ~high1).unsqueeze(1), logits1, corrected2)

            if both.any():
                local1 = torch.log(self.conv(prob1) + 1e-8)
                local2 = torch.log(self.conv(prob2) + 1e-8)
                corrected1 = torch.where(both.unsqueeze(1), local2, corrected1)
                corrected2 = torch.where(both.unsqueeze(1), local1, corrected2)

        prob_c1 = F.softmax(corrected1, dim=1)
        prob_c2 = F.softmax(corrected2, dim=1)
        ent_c1 = -torch.sum(prob_c1 * torch.log(prob_c1 + 1e-8), dim=1) / max_entropy
        ent_c2 = -torch.sum(prob_c2 * torch.log(prob_c2 + 1e-8), dim=1) / max_entropy
        conf1 = 1.0 - ent_c1
        conf2 = 1.0 - ent_c2
        denom = conf1 + conf2 + 1e-8
        w1 = (conf1 / denom).unsqueeze(1)
        w2 = (conf2 / denom).unsqueeze(1)
        return self.weights[0] * w1 * corrected1 + self.weights[1] * w2 * corrected2
