"""Pack only the volumes the experiments actually touch, as float16.

The full set is 7385 volumes (8.2 GB), but 1474 of them appear in no split and
no SSL pool. Packing just the 5911 that are used gives ~6.6 GB, which uploads
faster and fits a free Kaggle dataset (20 GB limit) comfortably.

  python scripts/export_volumes_subset.py out.npy            # supervised + SSL
  python scripts/export_volumes_subset.py out.npy --supervised-only
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config as C
from src.common.data import load_splits, load_ssl_pool
from src.common.transforms import window_hu

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "volumes_fp16_subset.npy")
SUP_ONLY = "--supervised-only" in sys.argv


def main() -> None:
    need = set()
    for d in load_splits("3class").values():
        need |= set(d[C.ID_COL])
    for d in load_splits("binary").values():
        need |= set(d[C.ID_COL])
    print(f"supervised volumes: {len(need)}")
    if not SUP_ONLY:
        ssl = set(load_ssl_pool()[C.ID_COL])
        need |= ssl
        print(f"+ SSL pool -> {len(need)} total")

    df = pd.read_csv(C.METADATA / "all_nodules_v2.csv", low_memory=False)
    df = df[df[C.ID_COL].isin(need)].reset_index(drop=True)
    ids, paths = df[C.ID_COL].tolist(), df[C.PATH_COL].tolist()
    missing = need - set(ids)
    if missing:
        raise SystemExit(f"{len(missing)} needed ids not in the manifest")

    print("checking float16 error on 20 random volumes...")
    rng = np.random.default_rng(0); worst = 0.0
    for i in rng.choice(len(ids), 20, replace=False):
        v = np.load(C.DATA_ROOT / paths[i]).astype(np.float32)
        worst = max(worst, float(np.abs(
            window_hu(v) - window_hu(v.astype(np.float16).astype(np.float32))).max()))
    print(f"  worst windowed error {worst:.2e} ({100*worst:.4f}% of range)")
    if worst > 1e-3:
        raise SystemExit("float16 error above tolerance")

    n = len(ids)
    # Resumable: a sidecar records how many rows are written, so an I/O stall
    # (iCloud fetching an offloaded file, for instance) costs one row, not the
    # whole export.
    prog = OUT.with_name(OUT.stem + ".progress")
    start = 0
    if OUT.exists() and prog.exists():
        try:
            saved = json.loads(prog.read_text())
            if saved.get("n") == n:
                start = int(saved.get("done", 0))
                print(f"resuming from volume {start}/{n}")
        except Exception:  # noqa: BLE001
            start = 0

    mode = "r+" if (start and OUT.exists()) else "w+"
    out = (np.lib.format.open_memmap(OUT, mode="r+") if mode == "r+"
           else np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float16,
                                          shape=(n, *C.SAMPLE_SHAPE_ZYX)))
    for i in range(start, n):
        src = C.DATA_ROOT / paths[i]
        for attempt in range(5):
            try:
                out[i] = np.load(src, mmap_mode="r").astype(np.float16)
                break
            except (OSError, TimeoutError) as exc:
                if attempt == 4:
                    raise SystemExit(f"giving up on {ids[i]} after 5 tries: {exc}")
                wait = 2 ** attempt
                print(f"  retry {attempt + 1} for {ids[i]} in {wait}s ({exc})",
                      flush=True)
                time.sleep(wait)
        if (i + 1) % 200 == 0:
            out.flush()
            prog.write_text(json.dumps({"n": n, "done": i + 1}))
            print(f"  {i + 1}/{n}", flush=True)
    out.flush(); del out
    prog.write_text(json.dumps({"n": n, "done": n}))
    OUT.with_name(OUT.stem + "_index.json").write_text(
        json.dumps({nid: i for i, nid in enumerate(ids)}))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB) covering {n} volumes")


if __name__ == "__main__":
    main()
