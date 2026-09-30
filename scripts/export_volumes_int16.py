"""Pack all volumes into one int16 file for upload to Colab/Kaggle.

Stored HU values are float32 but are integral in range [-3024, 3080], so int16
is lossless and halves the size: 17 GB -> ~8.5 GB, which fits a free Google
Drive. Verifies losslessness on a sample before writing everything.
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

    print("verifying int16 is lossless on 25 random volumes...")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ids), 25, replace=False):
        v = np.load(C.DATA_ROOT / paths[i])
        if not np.array_equal(v, v.astype(np.int16).astype(np.float32)):
            raise SystemExit(f"NOT lossless for {ids[i]} "
                             f"(range {v.min()}..{v.max()}) — aborting")
    print("  lossless confirmed")

    n = len(ids)
    out = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.int16,
                                    shape=(n, *C.SAMPLE_SHAPE_ZYX))
    for i, rel in enumerate(paths):
        out[i] = np.load(C.DATA_ROOT / rel, mmap_mode="r").astype(np.int16)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    out.flush(); del out
    OUT.with_name(OUT.stem + "_index.json").write_text(
        json.dumps({nid: i for i, nid in enumerate(ids)}))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB) and its index")


if __name__ == "__main__":
    main()
