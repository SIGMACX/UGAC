from __future__ import annotations

from torch import nn

from .unet import UNet


def build_model(
    model_name: str,
    num_classes: int,
    input_channels: int = 1,
    image_size: int = 512,
    patch_size: int = 4,
) -> nn.Module:
    name = model_name.lower()
    if name in {"unet", "u_net"}:
        return UNet(in_channels=input_channels, num_classes=num_classes)
    if name in {"mamba_unet", "vmunet", "vmamba_unet"}:
        from .mamba_unet import VMUNet

        return VMUNet(
            patch_size=patch_size,
            num_classes=num_classes,
            input_channels=input_channels,
            depths=[2, 2, 2, 2],
            depths_decoder=[2, 2, 2, 1],
            drop_path_rate=0.2,
        )

    supported = "unet, mamba_unet"
    raise ValueError(f"Unknown model_name={model_name!r}. Supported models: {supported}.")
