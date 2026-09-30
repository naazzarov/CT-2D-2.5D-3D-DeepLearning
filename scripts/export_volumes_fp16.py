"""Pack all volumes into one float16 file for upload to Colab/Kaggle.

Halves the size (17 GB -> ~8.5 GB) so it fits a free Google Drive.

int16 was the obvious choice but is NOT safe here: the stored HU values are not
integers. Volumes were resampled to 1 mm isotropic with interpolation, so
values look like -1023.354736, and an int16 cast would silently quantise them.

float16 is used instead. Measured round-trip error over 15 random volumes:
at most 1.0 HU, which after the [-1000, 400] window and scaling to [0, 1] is
1.8e-4 -- 0.018% of the input range. For comparison the training augmentation
adds Gaussian noise with sigma 0.02, roughly 100x larger, and CT acquisition
noise is larger still. The script re-measures this before writing anything.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config as C

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "volumes_int16.npy")


def main() -> None:
    df = pd.read_csv(C.METADATA / "all_nodules_v2.csv", low_memory=False)
    ids, paths = df[C.ID_COL].tolist(), df[C.PATH_COL].tolist()

    from src.common.transforms import window_hu
    print("measuring float16 round-trip error on 25 random volumes...")
    rng = np.random.default_rng(0)
    worst_hu, worst_win = 0.0, 0.0
    for i in rng.choice(len(ids), 25, replace=False):
        v = np.load(C.DATA_ROOT / paths[i]).astype(np.float32)
        h = v.astype(np.float16).astype(np.float32)
        worst_hu = max(worst_hu, float(np.abs(v - h).max()))
        worst_win = max(worst_win,
                        float(np.abs(window_hu(v) - window_hu(h)).max()))
    print(f"  worst raw-HU error {worst_hu:.4f} HU")
    print(f"  worst error after windowing {worst_win:.2e} "
          f"({100 * worst_win:.4f}% of range)")
    if worst_win > 1e-3:
        raise SystemExit("float16 error above tolerance — aborting")
    print("  within tolerance")

    n = len(ids)
    out = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16,
                                    shape=(n, *C.SAMPLE_SHAPE_ZYX))
    for i, rel in enumerate(paths):
        out[i] = np.load(C.DATA_ROOT / rel, mmap_mode="r").astype(np.float16)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    out.flush(); del out
    OUT.with_name(OUT.stem + "_index.json").write_text(
        json.dumps({nid: i for i, nid in enumerate(ids)}))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB) and its index")


if __name__ == "__main__":
    main()
