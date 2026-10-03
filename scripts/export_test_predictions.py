"""Recreate test_probabilities.npy / test_targets.npy from saved checkpoints.

  python scripts/export_test_predictions.py          # 2d and 2p5d runs

For every results/<rep>/<tag>/seed*/ that has best.pt and result.json but no
saved predictions, rebuild the model, evaluate the test split, and save the
predictions -- without retraining. The recomputed test macro-F1 must match the
value recorded in result.json, otherwise the predictions are not written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "2d"))
from src.common import config as C
from src.common.data import CentralSliceDataset, apply_cohort, load_splits
from src.common.seed import device
from model import ResNet18Classifier  # noqa: E402


def export(run_dir: Path, dev) -> str:
    r = json.loads((run_dir / "result.json").read_text())
    if r.get("test") is None:
        return "no test evaluation recorded"
    a = r["args"]
    rep, task = r.get("rep", a.get("rep", "2d")), r["task"]
    spec = C.TASKS[task]
    test = apply_cohort(load_splits(task), a.get("cohort", "all"), task)["test"]
    ds = CentralSliceDataset(test, r["train_mean"], r["train_std"], task=task,
                             rep=rep)
    in_ch = r.get("in_channels", C.N_NEIGHBOUR_SLICES if rep == "2p5d" else 1)
    model = ResNet18Classifier(in_channels=in_ch,
                               num_classes=spec["num_classes"],
                               dropout=a["dropout"]).to(dev)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev))
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=128):
            probs.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
            ys.append(y.numpy())
    p, y = np.concatenate(probs), np.concatenate(ys)
    f1 = f1_score(y, p.argmax(1), average="macro", zero_division=0)
    recorded = r["test"]["macro_f1"]
    if abs(f1 - recorded) > 1e-4:
        return f"MISMATCH: recomputed macro-F1 {f1:.4f} vs recorded {recorded:.4f}"
    np.save(run_dir / "test_probabilities.npy", p)
    np.save(run_dir / "test_targets.npy", y)
    return f"exported ({len(y)} nodules, macro-F1 {f1:.4f} matches)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["2d", "2p5d"])
    args = ap.parse_args()
    dev = device()
    for rep in args.reps:
        for d in sorted((C.RESULTS_DIR / rep).glob("*/seed*")):
            if "sweep" in d.parts or (d / "test_probabilities.npy").exists():
                continue
            if not ((d / "best.pt").exists() and (d / "result.json").exists()):
                continue
            print(f"{d.relative_to(C.RESULTS_DIR)}: {export(d, dev)}")


if __name__ == "__main__":
    main()
