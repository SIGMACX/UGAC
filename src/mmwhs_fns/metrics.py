from __future__ import annotations

import numpy as np
import torch


def compute_confusion_counts(pred: np.ndarray, target: np.ndarray, num_classes: int):
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if target.ndim == 4:
        target = np.argmax(target, axis=1)

    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    tn = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        tp[cls] = np.logical_and(pred_cls, target_cls).sum()
        fp[cls] = np.logical_and(pred_cls, ~target_cls).sum()
        fn[cls] = np.logical_and(~pred_cls, target_cls).sum()
        tn[cls] = np.logical_and(~pred_cls, ~target_cls).sum()
    return tp, fp, fn, tn


def dice_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (2 * tp) / (2 * tp + fp + fn + eps)
