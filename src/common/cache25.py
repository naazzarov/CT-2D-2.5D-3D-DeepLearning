"""Cache the 5 neighbouring slices used by the 2.5D representation.

Stores slices 50..54 of every (104, 72, 80) volume as (N, 5, 72, 80) float32
raw HU -- about 850 MB, which still fits in RAM and avoids re-reading 17 GB of
volumes each epoch. Windowing stays in the transform so the cache never needs
rebuilding if the HU window changes.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.common import config as C

CACHE_ARRAY = C.CACHE_DIR / "neighbour_slices.npy"
CACHE_INDEX = C.CACHE_DIR / "neighbour_slices_index.json"


def build(force: bool = False) -> None:
    if CACHE_ARRAY.exists() and CACHE_INDEX.exists() and not force:
        print(f"cache already present: {CACHE_ARRAY}")
        return
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.METADATA / "all_nodules_v2.csv", low_memory=False)
    ids, paths = df[C.ID_COL].tolist(), df[C.PATH_COL].tolist()
    lo, hi = C.SLICE_WINDOW
    n = len(ids)
    print(f"caching slices {lo}..{hi - 1} for {n} nodules")
    out = np.lib.format.open_memmap(
        CACHE_ARRAY, mode="w+", dtype=np.float32,
        shape=(n, hi - lo, *C.SLICE_HW))
    bad = []
    for i, (nid, rel) in enumerate(zip(ids, paths)):
        try:
            vol = np.load(C.DATA_ROOT / rel, mmap_mode="r")
            if tuple(vol.shape) != C.SAMPLE_SHAPE_ZYX:
                bad.append(nid); continue
            out[i] = np.asarray(vol[lo:hi], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{nid}: {exc}")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    out.flush(); del out
    CACHE_INDEX.write_text(json.dumps({nid: i for i, nid in enumerate(ids)}))
    print(f"wrote {CACHE_ARRAY} ({CACHE_ARRAY.stat().st_size / 1e6:.0f} MB)")
    if bad:
        print(f"WARNING: {len(bad)} problem files", file=sys.stderr)


def load() -> tuple[np.ndarray, dict[str, int]]:
    if not CACHE_ARRAY.exists():
        raise FileNotFoundError("run: python scripts/build_cache25.py")
    return np.load(CACHE_ARRAY, mmap_mode="r"), json.loads(CACHE_INDEX.read_text())


if __name__ == "__main__":
    build(force="--force" in sys.argv)
