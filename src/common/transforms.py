"""HU windowing and training-only augmentation.

The windowing here is the single most important shared decision: all three
approaches (2D / 2.5D / 3D) must apply exactly this function, otherwise the
models see differently-scaled inputs and the comparison is not about spatial
context any more.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.common import config as C


def window_hu(x: np.ndarray | torch.Tensor):
    """Clip raw HU to the lung window and scale to [0, 1]."""
    if isinstance(x, np.ndarray):
        x = np.clip(x, C.HU_MIN, C.HU_MAX)
        return ((x - C.HU_MIN) / (C.HU_MAX - C.HU_MIN)).astype(np.float32)
    x = torch.clamp(x, C.HU_MIN, C.HU_MAX)
    return (x - C.HU_MIN) / (C.HU_MAX - C.HU_MIN)


def standardize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


# --------------------------------------------------------------- augmentation
# Applied to TRAINING data only. Kept geometrically mild: a nodule crop is
# already tightly centred, so large translations would push it out of frame.
def _affine(x: torch.Tensor, angle: float, tx: float, ty: float, scale: float):
    """x: (C, H, W). Rotation in degrees, translation in fraction of extent."""
    th = torch.tensor(angle * np.pi / 180.0, dtype=torch.float32)
    cos, sin = torch.cos(th) / scale, torch.sin(th) / scale
    mat = torch.tensor(
        [[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32
    ).unsqueeze(0)
    grid = F.affine_grid(mat, [1, *x.shape], align_corners=False)
    return F.grid_sample(
        x.unsqueeze(0), grid, mode="bilinear",
        padding_mode="border", align_corners=False,
    ).squeeze(0)


def augment_supervised(x: torch.Tensor, rng: np.random.Generator,
                       strength: str = "mild") -> torch.Tensor:
    """Augmentation for supervised training. x: (C, H, W) in [0, 1].

    `strength="none"` disables augmentation entirely (for the no-augmentation
    baseline). `strength="strong"` widens every range and adds random erasing; with only
    1876 training nodules and an 11.2M-parameter ResNet-18, aggressive
    augmentation is one of the few regularisers the frozen protocol allows.
    """
    if strength == "none":
        return x.clamp(0.0, 1.0)
    p = dict(rot=15.0, tr=0.06, sc=0.1, it=0.1, sh=0.05, nz=0.02, erase=0.0)
    if strength == "strong":
        p = dict(rot=30.0, tr=0.12, sc=0.2, it=0.3, sh=0.15, nz=0.05, erase=0.25)

    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])           # left-right
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-2])           # anterior-posterior
    if rng.random() < 0.8:
        x = _affine(
            x,
            angle=float(rng.uniform(-p["rot"], p["rot"])),
            tx=float(rng.uniform(-p["tr"], p["tr"])),
            ty=float(rng.uniform(-p["tr"], p["tr"])),
            scale=float(rng.uniform(1 - p["sc"], 1 + p["sc"])),
        )
    if rng.random() < 0.5:                      # intensity jitter
        x = x * float(rng.uniform(1 - p["it"], 1 + p["it"])) \
            + float(rng.uniform(-p["sh"], p["sh"]))
    if rng.random() < 0.3:                      # noise
        x = x + torch.randn_like(x) * p["nz"]
    if p["erase"] > 0 and rng.random() < p["erase"]:   # random erasing
        h, w = x.shape[-2:]
        eh, ew = int(h * rng.uniform(.1, .25)), int(w * rng.uniform(.1, .25))
        top, left = int(rng.integers(0, h - eh)), int(rng.integers(0, w - ew))
        x = x.clone()
        x[..., top:top + eh, left:left + ew] = float(x.mean())
    return x.clamp(0.0, 1.0)


def augment_moco(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Stronger augmentation for MoCo v2 views. x: (C, H, W) in [0, 1]."""
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-2])
    x = _affine(                                # always: random resized-crop-ish
        x,
        angle=float(rng.uniform(-30, 30)),
        tx=float(rng.uniform(-0.12, 0.12)),
        ty=float(rng.uniform(-0.12, 0.12)),
        scale=float(rng.uniform(0.7, 1.3)),
    )
    if rng.random() < 0.8:                      # "colour" jitter -> intensity
        x = x * float(rng.uniform(0.6, 1.4)) + float(rng.uniform(-0.2, 0.2))
    if rng.random() < 0.5:                      # gaussian blur (MoCo v2)
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :]) / 16.0
        x = F.conv2d(x.unsqueeze(0), k.view(1, 1, 3, 3).expand(x.shape[0], 1, 3, 3),
                     padding=1, groups=x.shape[0]).squeeze(0)
    if rng.random() < 0.2:
        x = x + torch.randn_like(x) * 0.05
    return x.clamp(0.0, 1.0)
