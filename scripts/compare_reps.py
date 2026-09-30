"""Cross-representation comparison: 2D vs 2.5D vs 3D.

Reads every completed run under results/<rep>/<tag>/seed*/result.json and emits
results/COMPARISON.md plus results/comparison.json -- the tables the paper needs.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config as C
from src.common.data import load_splits
from src.common.metrics import majority_baseline

REPS = [("2d", "2D (1 slice)"), ("2p5d", "2.5D (5 slices)"),
        ("3d", "3D (104 slices)")]
MODELS = [("model_a_noaug", "Model A — supervised, no augmentation", "3class"),
          ("model_b_noaug", "Model B — MoCo v2 → fine-tune", "3class"),
          ("model_c_mild", "Model C — supervised, mild augmentation", "3class"),
          ("binary_mild", "Binary — supervised, mild augmentation", "binary")]


def collect(rep: str, tag: str):
    runs = []
    for f in sorted((C.RESULTS_DIR / rep / tag).glob("seed*/result.json")):
        r = json.loads(f.read_text())
        if r.get("test"):
            runs.append(r)
    return runs


def ms(runs, key, where="test"):
    if not runs:
        return "—"
    v = [r[where][key] for r in runs]
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) > 1 else f"{np.mean(v):.4f}"


def mean(runs, key, where="test"):
    return float(np.mean([r[where][key] for r in runs])) if runs else float("nan")


def auc_key(runs):
    return "roc_auc" if "roc_auc" in runs[0]["test"] else "roc_auc_ovr_macro"


def main() -> None:
    s3 = load_splits("3class"); sb = load_splits("binary")
    b3 = majority_baseline(s3["test"][C.LABEL_COL].to_numpy(), C.CLASS_NAMES)
    bb = majority_baseline(sb["test"][C.BINARY_LABEL_COL].to_numpy(),
                           C.BINARY_CLASS_NAMES)

    L = ["# 2D vs 2.5D vs 3D — cross-representation comparison", "",
         "All arms share one pipeline: identical HU window "
         f"[{C.HU_MIN:.0f}, {C.HU_MAX:.0f}], identical patient-level splits, "
         "identical label columns, identical loss (class-weighted cross-entropy, "
         "label smoothing 0.10), optimiser (AdamW, lr 1e-3, weight decay 0.05, "
         "cosine), early stopping on validation macro-F1, and identical metric "
         "code. Only the input representation and encoder change.", "",
         "| representation | input | encoder |", "|---|---|---|",
         "| 2D | `(1, 72, 80)` — central slice 52 | ResNet-18 |",
         "| 2.5D | `(5, 72, 80)` — slices 50–54 | ResNet-18 |",
         "| 3D | `(1, 104, 72, 80)` — full crop | 3D ResNet-10 |", "",
         "All numbers are test-set, mean ± std over 3 seeds.", "",
         "**Majority-class baselines:** 3-class macro-F1 "
         f"{b3['macro_f1']:.4f} (accuracy {b3['accuracy']:.4f}); "
         f"binary macro-F1 {bb['macro_f1']:.4f} "
         f"(accuracy {bb['accuracy']:.4f}).", ""]

    summary = {}
    for tag, title, task in MODELS:
        found = {rep: collect(rep, tag) for rep, _ in REPS}
        found = {k: v for k, v in found.items() if v}
        if not found:
            continue
        names = C.BINARY_CLASS_NAMES if task == "binary" else C.CLASS_NAMES
        L += [f"## {title}", "",
              "| metric | " + " | ".join(lbl for rep, lbl in REPS
                                         if rep in found) + " |",
              "|---" * (len(found) + 1) + "|"]
        order = [r for r, _ in REPS if r in found]
        for key, label in [("macro_f1", "**macro F1**"), ("accuracy", "accuracy"),
                           ("balanced_accuracy", "balanced accuracy"),
                           ("macro_precision", "macro precision"),
                           ("macro_recall", "macro recall")]:
            L.append(f"| {label} | " +
                     " | ".join(ms(found[r], key) for r in order) + " |")
        L.append("| ROC-AUC | " + " | ".join(
            ms(found[r], auc_key(found[r])) for r in order) + " |")
        for n in names:
            L.append(f"| F1 {n} | " +
                     " | ".join(ms(found[r], f"f1_{n}") for r in order) + " |")
        L.append("| best val macro F1 | " + " | ".join(
            f"{np.mean([x['best_val_macro_f1'] for x in found[r]]):.4f}"
            for r in order) + " |")
        L.append("| best epoch | " + " | ".join(
            str([x["best_epoch"] for x in found[r]]) for r in order) + " |")
        L.append("| parameters | " + " | ".join(
            f"{found[r][0]['params']:,}" for r in order) + " |")
        L.append("| train min/seed | " + " | ".join(
            f"{np.mean([x['train_minutes'] for x in found[r]]):.1f}"
            for r in order) + " |")
        L.append("")
        summary[tag] = {r: {"test_macro_f1_mean": mean(found[r], "macro_f1"),
                            "test_macro_f1_std": float(np.std(
                                [x["test"]["macro_f1"] for x in found[r]])),
                            "test_accuracy": mean(found[r], "accuracy"),
                            "test_balanced_accuracy": mean(found[r],
                                                           "balanced_accuracy"),
                            "test_auc": mean(found[r], auc_key(found[r])),
                            "params": found[r][0]["params"],
                            "seeds": [x["seed"] for x in found[r]]}
                       for r in order}

    out = C.RESULTS_DIR / "COMPARISON.md"
    out.write_text("\n".join(L))
    (C.RESULTS_DIR / "comparison.json").write_text(json.dumps(
        {"models": summary, "baselines": {"three_class": b3, "binary": bb}},
        indent=2))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
