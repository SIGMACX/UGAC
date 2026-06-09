from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_MMWHS_LABEL_VALUES = (0, 60, 126, 150, 165, 179, 246, 255)


class MMWHSDataset(Dataset):
    """MMWHS 2D PNG slice dataset.

    Text files contain image filenames relative to ``image_dir``. When labels are
    available, label filenames are expected to match image filenames.
    """

    def __init__(
        self,
        names_txt: str | Path,
        image_dir: str | Path,
        label_dir: str | Path | None = None,
        image_size: int | tuple[int, int] = 512,
        num_classes: int = 8,
        label_values: Iterable[int] = DEFAULT_MMWHS_LABEL_VALUES,
        use_cutout: bool = False,
        use_color_jitter: bool = False,
        use_blur: bool = False,
    ) -> None:
        self.names_txt = Path(names_txt)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.num_classes = num_classes
        self.label_values = tuple(int(v) for v in label_values)
        self.use_cutout = use_cutout
        self.use_color_jitter = use_color_jitter
        self.use_blur = use_blur

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        if not self.names_txt.exists():
            raise FileNotFoundError(f"Split file not found: {self.names_txt}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if self.label_dir is not None and not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.image_names = [line.strip() for line in self.names_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.image_names:
            raise RuntimeError(f"Split file is empty: {self.names_txt}")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = self.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("L")
        original_size = image.size
        image = self._augment_image(image)
        image_tensor = self.image_transform(image)

        if self.label_dir is None:
            label_tensor = torch.zeros(self.image_size, dtype=torch.long)
        else:
            label_path = self.label_dir / image_name
            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")
            label_img = Image.open(label_path).convert("L")
            label_img = label_img.resize((self.image_size[1], self.image_size[0]), resample=Image.NEAREST)
            label_tensor = torch.from_numpy(self._map_label(np.asarray(label_img))).long()

        return image_tensor, label_tensor, image_name, original_size

    def _map_label(self, label: np.ndarray) -> np.ndarray:
        unique = np.unique(label)
        if unique.size and unique.min() >= 0 and unique.max() < self.num_classes:
            return label.astype(np.int64)

        if len(self.label_values) != self.num_classes:
            raise ValueError("label_values length must match num_classes.")

        mapped = np.zeros_like(label, dtype=np.int64)
        known = np.zeros_like(label, dtype=bool)
        for class_idx, pixel_value in enumerate(self.label_values):
            mask = label == pixel_value
            mapped[mask] = class_idx
            known |= mask

        if not known.all():
            unknown = np.unique(label[~known]).tolist()
            raise ValueError(f"Unknown label pixel values {unknown}; configure --label_values.")
        return mapped

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if self.use_cutout:
            image = _cutout_gray(image)
        if self.use_color_jitter:
            image = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)(image)
        if self.use_blur and random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        return image


def _cutout_gray(
    image: Image.Image,
    p: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
    ratio_1: float = 0.3,
    ratio_2: float = 1 / 0.3,
) -> Image.Image:
    if random.random() >= p:
        return image

    arr = np.asarray(image).copy()
    img_h, img_w = arr.shape
    for _ in range(100):
        size = np.random.uniform(size_min, size_max) * img_h * img_w
        ratio = np.random.uniform(ratio_1, ratio_2)
        erase_w = int(np.sqrt(size / ratio))
        erase_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)
        if x + erase_w <= img_w and y + erase_h <= img_h:
            arr[y : y + erase_h, x : x + erase_w] = np.random.randint(0, 2, (erase_h, erase_w))
            break
    return Image.fromarray(arr)
