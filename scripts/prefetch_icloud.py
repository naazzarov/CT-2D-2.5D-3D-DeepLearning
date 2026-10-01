"""Materialise the needed volumes from iCloud before exporting.

The dataset lives under ~/Documents, which is iCloud-synced with "Optimise Mac
Storage" on, so most sample files are *dataless*: the directory entry shows the
full size but no blocks are allocated locally, and any read triggers a network
fetch. Reading them one at a time during the export ran at ~3.6 s/volume and
caused a TimeoutError.

Fetching in parallel is far faster, because the bottleneck is per-file latency
rather than bandwidth. This script lists the files each experiment needs and
reads them concurrently to force materialisation.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config as C
from src.common.data import load_splits, load_ssl_pool

WORKERS = int(os.environ.get("PREFETCH_WORKERS", "16"))


def is_dataless(p: Path) -> bool:
    try:
        st = p.stat()
        return st.st_size > 0 and st.st_blocks == 0
    except OSError:
        return False


def fetch(p: Path) -> bool:
    """Force materialisation by reading the file."""
    try:
        subprocess.run(["brctl", "download", str(p)],
                       capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(p, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except OSError:
        return False


def main() -> None:
    need = set()
    for d in load_splits("3class").values():
        need |= set(d[C.ID_COL])
    for d in load_splits("binary").values():
        need |= set(d[C.ID_COL])
    need |= set(load_ssl_pool()[C.ID_COL])

    df = pd.read_csv(C.METADATA / "all_nodules_v2.csv", low_memory=False)
    df = df[df[C.ID_COL].isin(need)]
    paths = [C.DATA_ROOT / p for p in df[C.PATH_COL]]
    print(f"{len(paths)} volumes needed")

    todo = [p for p in paths if is_dataless(p)]
    print(f"{len(todo)} are dataless (offloaded to iCloud), "
          f"{len(paths) - len(todo)} already local")
    if not todo:
        print("nothing to fetch")
        return
    print(f"fetching with {WORKERS} workers...")

    t0, done, failed = time.time(), 0, 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for ok in ex.map(fetch, todo):
            done += 1
            failed += 0 if ok else 1
            if done % 200 == 0:
                el = time.time() - t0
                rate = done / el
                print(f"  {done}/{len(todo)}  {rate:.1f} files/s  "
                      f"eta {(len(todo) - done) / max(rate, .01) / 60:.0f} min",
                      flush=True)
    print(f"done in {(time.time() - t0)/60:.1f} min, {failed} failures")


if __name__ == "__main__":
    main()
