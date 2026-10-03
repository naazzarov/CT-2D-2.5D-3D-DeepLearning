"""Volume dataset and 3D augmentation for the volumetric arm.

The full crops are 17 GB in total, so unlike 2D/2.5D they are not cached in
RAM -- each sample is memory-mapped from disk on demand.

The augmentation mirrors the 2D "mild" recipe exactly in probabilities and
strengths. The only deliberate difference: no flip or rotation along z, because
CT has a consistent anatomical orientation in z and our z-context is
interpolated from 2.5 mm acquisitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common import config as C
from src.common.transforms import window_hu

# Optional packed volume store: set CT_VOLUME_PACK=/path/to/volumes_int16.npy
# (with a sibling volumes_int16_index.json). Used on Colab so 7385 volumes are
# one sequential file instead of 7385 small reads.
import json as _json
import os as _os

_PACK = _os.environ.get("CT_VOLUME_PACK", "")
_pack_arr = None
_pack_index: dict[str, int] = {}
if _PACK:
    _p = __import__("pathlib").Path(_PACK)
    _pack_arr = np.load(_p, mmap_mode="r")
    _pack_index = _json.loads(
        _p.with_name(_p.stem + "_index.json").read_text())


def _load_volume(path, nodule_id: str) -> np.ndarray:
    """Raw HU volume as float32, from the packed store if available."""
    if _pack_arr is not None and nodule_id in _pack_index:
        return np.asarray(_pack_arr[_pack_index[nodule_id]], dtype=np.float32)
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)


def _affine3d(x: torch.Tensor, angle: float, tz: float, ty: float, tx: float,
              scale: float) -> torch.Tensor:
    """x: (C, D, H, W). In-plane rotation about the z axis only.

    affine_grid orders theta rows and columns as (x, y, z) = (W, H, D), the
    reverse of the tensor's (D, H, W) layout.
    """
    th = torch.tensor(angle * np.pi / 180.0, dtype=torch.float32)
    cos, sin = torch.cos(th) / scale, torch.sin(th) / scale
    mat = torch.tensor([
        [cos, -sin, 0.0, tx],
        [sin, cos, 0.0, ty],
        [0.0, 0.0, 1.0 / scale, tz],
    ], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(mat, [1, *x.shape], align_corners=False)
    return F.grid_sample(x.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="border", align_corners=False).squeeze(0)


def augment3d(x: torch.Tensor, rng: np.random.Generator,
              strength: str = "mild") -> torch.Tensor:
    if strength == "none":
        return x.clamp(0.0, 1.0)
    p = dict(rot=15.0, tr=0.06, sc=0.1, it=0.1, sh=0.05, nz=0.02)
    if strength == "strong":
        p = dict(rot=30.0, tr=0.12, sc=0.2, it=0.3, sh=0.15, nz=0.05)
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])           # left-right
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-2])           # anterior-posterior
    if rng.random() < 0.8:
        x = _affine3d(x, angle=float(rng.uniform(-p["rot"], p["rot"])),
                      tz=float(rng.uniform(-p["tr"], p["tr"])),
                      ty=float(rng.uniform(-p["tr"], p["tr"])),
                      tx=float(rng.uniform(-p["tr"], p["tr"])),
                      scale=float(rng.uniform(1 - p["sc"], 1 + p["sc"])))
    if rng.random() < 0.5:
        x = x * float(rng.uniform(1 - p["it"], 1 + p["it"])) \
            + float(rng.uniform(-p["sh"], p["sh"]))
    if rng.random() < 0.3:
        x = x + torch.randn_like(x) * p["nz"]
    return x.clamp(0.0, 1.0)


class VolumeDataset(Dataset):
    """Full (104, 72, 80) crop -> class label. Returns (1, D, H, W)."""

    def __init__(self, df: pd.DataFrame, mean: float, std: float,
                 train: bool = False, seed: int = 0, aug: str = "mild",
                 task: str = "3class"):
        self.paths = [C.DATA_ROOT / p for p in df[C.PATH_COL]]
        self.ids = df[C.ID_COL].tolist()
        self.labels = df[C.TASKS[task]["label_col"]].to_numpy(dtype=np.int64)
        self.mean, self.std, self.train, self.aug = mean, std, train, aug
        self._seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        vol = _load_volume(self.paths[i], self.ids[i])
        x = torch.from_numpy(window_hu(vol)).unsqueeze(0)   # (1, D, H, W)
        if self.train:
            x = augment3d(x, np.random.default_rng((self._seed, self.epoch, i)),
                          self.aug)
        return (x - self.mean) / self.std, int(self.labels[i])


class VolumeMoCoDataset(Dataset):
    """Two augmented views of the same volume, no labels."""

    def __init__(self, df: pd.DataFrame, mean: float, std: float, seed: int = 0):
        self.paths = [C.DATA_ROOT / p for p in df[C.PATH_COL]]
        self.ids = df[C.ID_COL].tolist()
        self.mean, self.std, self._seed, self.epoch = mean, std, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        vol = _load_volume(self.paths[i], self.ids[i])
        base = torch.from_numpy(window_hu(vol)).unsqueeze(0)
        rng = np.random.default_rng((self._seed, self.epoch, i))
        q = (augment3d(base, rng, "strong") - self.mean) / self.std
        k = (augment3d(base, rng, "strong") - self.mean) / self.std
        return q, k


def volume_train_statistics(train: pd.DataFrame, limit: int = 300) -> tuple[float, float]:
    """Mean/std of windowed intensities over a sample of the TRAIN split."""
    rng = np.random.default_rng(0)
    paths = [C.DATA_ROOT / p for p in train[C.PATH_COL]]
    ids = train[C.ID_COL].tolist()
    idx = rng.choice(len(paths), size=min(limit, len(paths)), replace=False)
    vals, sq = [], []
    for i in idx:
        w = window_hu(_load_volume(paths[i], ids[i]))
        vals.append(w.mean()); sq.append((w ** 2).mean())
    m = float(np.mean(vals))
    return m, float(np.sqrt(max(np.mean(sq) - m * m, 1e-8)))
