"""Lightweight 3D ResNet-10 for the volumetric arm.

Deliberately the direct 3D analogue of the 2D/2.5D ResNet-18: same BasicBlock
structure and same stem, with 3D convolutions and one block per stage instead
of two ([1,1,1,1] vs [2,2,2,2]). ResNet-10 rather than ResNet-18 because a 3D
ResNet-18 would have ~33M parameters against 1876 training volumes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common import config as C


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inp: int, out: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + idt)


class ResNet3D(nn.Module):
    def __init__(self, layers=(1, 1, 1, 1), in_channels: int = 1,
                 widths=(64, 128, 256, 512)):
        super().__init__()
        self.inp = widths[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, widths[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(widths[0]), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = self._stage(widths[0], layers[0], 1)
        self.layer2 = self._stage(widths[1], layers[1], 2)
        self.layer3 = self._stage(widths[2], layers[2], 2)
        self.layer4 = self._stage(widths[3], layers[3], 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = widths[3]

    def _stage(self, out: int, blocks: int, stride: int) -> nn.Sequential:
        down = None
        if stride != 1 or self.inp != out:
            down = nn.Sequential(nn.Conv3d(self.inp, out, 1, stride, bias=False),
                                 nn.BatchNorm3d(out))
        layers = [BasicBlock3d(self.inp, out, stride, down)]
        self.inp = out
        layers += [BasicBlock3d(out, out) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.pool(x), 1)


def build_encoder(in_channels: int = 1) -> nn.Module:
    """3D ResNet-10 trunk, outputs 512-d."""
    return ResNet3D(layers=(1, 1, 1, 1), in_channels=in_channels)


class ResNet10Classifier(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = C.NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = build_encoder(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.encoder(x)))

    def load_encoder(self, state: dict):
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        return list(missing), list(unexpected)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
