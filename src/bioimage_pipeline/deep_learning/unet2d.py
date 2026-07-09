"""Compact 2D U-Net model."""

import torch
import torch.nn.functional as functional
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution, batch-normalization, and ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Down(nn.Module):
    """Downsample followed by a DoubleConv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Up(nn.Module):
    """Upsample, concatenate a skip connection, and apply DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.reduce(self.upsample(inputs))
        height_difference = skip.size(2) - inputs.size(2)
        width_difference = skip.size(3) - inputs.size(3)
        inputs = functional.pad(
            inputs,
            [
                width_difference // 2,
                width_difference - width_difference // 2,
                height_difference // 2,
                height_difference - height_difference // 2,
            ],
        )
        return self.conv(torch.cat([skip, inputs], dim=1))


class UNet2D(nn.Module):
    """A compact four-level U-Net that returns segmentation logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output = self.up1(x5, x4)
        output = self.up2(output, x3)
        output = self.up3(output, x2)
        output = self.up4(output, x1)
        return self.outc(output)
