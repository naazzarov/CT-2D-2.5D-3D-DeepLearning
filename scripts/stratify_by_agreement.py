"""Agreement-stratified test performance for every representation and model.

  python scripts/stratify_by_agreement.py            # all runs found
  python scripts/stratify_by_agreement.py --reps 2d --tags binary_mild

Reads the per-seed test_probabilities.npy / test_targets.npy written by the
training scripts with --eval-test, aligns them with the test split CSV (row
order is preserved because test loaders do not shuffle, and the saved targets
are checked against the CSV labels), and reports, for each stratum of
src/common/agreement.py:

  * per-seed metrics (mean +/- std over seeds), and
  * the seed-averaged ensemble with percentile-bootstrap 95% CIs, plus a
    bootstrap CI on the confident-minus-remainder difference.

Writes results/agreement/<rep>__<tag>.json and results/AGREEMENT.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import agreement as A
from src.common import config as C
from src.common.data import load_splits

REPS = ("2d", "2p5d", "3d")
TAGS = ("model_a_noaug", "model_b_noaug", "model_c_mild", "binary_mild",
        "model_c_mild_confident", "binary_mild_confident")
STRATA = ("confident", "remainder", "midpoint", "not_midpoint",
          "disagreement", "no_disagreement", "single_reader",
          "three_plus_readers")
N_BOOT = 2000


def auc(y: np.ndarray, p: np.ndarray, k: int) -> float:
    if len(np.unique(y)) < k:
        return float("nan")
    if k == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def metrics(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    pred = p.argmax(1)
    return {"n": int(len(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro",
                                       labels=range(k), zero_division=0)),
            "auc": auc(y, p, k)}


def boot(y, p, k, rng, n=N_BOOT) -> dict:
    acc, au = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        acc.append(accuracy_score(y[s], p[s].argmax(1)))
        au.append(auc(y[s], p[s], k))
    au = np.array(au)
    au = au[~np.isnan(au)]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy_ci95": ci(acc), "auc_ci95": ci(au) if len(au) else None,
            "_acc": np.array(acc), "_auc": au}


def load_run(rep: str, tag: str):
    """-> (task, cohort, list of (seed, probs, targets)) or None."""
    seeds = []
    task = cohort = None
    for d in sorted((C.RESULTS_DIR / rep / tag).glob("seed*")):
        pf, tf, rf = (d / "test_probabilities.npy", d / "test_targets.npy",
                      d / "result.json")
        if not (pf.exists() and tf.exists() and rf.exists()):
            continue
        r = json.loads(rf.read_text())
        task = r["task"]
        cohort = r.get("args", {}).get("cohort", "all")
        seeds.append((r["seed"], np.load(pf), np.load(tf)))
    return (task, cohort, seeds) if seeds else None


def analyse(rep: str, tag: str, rng) -> dict | None:
    run = load_run(rep, tag)
    if run is None:
        return None
    task, cohort, seeds = run
    spec = C.TASKS[task]
    k = spec["num_classes"]
    test = load_splits(task)["test"].reset_index(drop=True)
    if cohort == "confident":
        test = test[A.confident_mask(test)].reset_index(drop=True)
    y_csv = test[spec["label_col"]].to_numpy()
    for seed, p, y in seeds:
        if len(y) != len(y_csv) or not np.array_equal(y, y_csv):
            raise AssertionError(
                f"{rep}/{tag} seed {seed}: saved test targets do not match the "
                f"{task} test CSV row order ({len(y)} vs {len(y_csv)} rows); "
                "refusing to align predictions with metadata")

    masks = A.strata(test)
    med = A.medians(test)
    p_ens = np.mean([p for _, p, _ in seeds], axis=0)
    out = {"rep": rep, "tag": tag, "task": task, "cohort": cohort,
           "seeds": [s for s, _, _ in seeds], "n_test": int(len(y_csv)),
           "overall": {"ensemble": metrics(y_csv, p_ens, k)},
           "strata": {}, "by_median": {}}

    boots = {}
    for name in STRATA:
        m = masks[name]
        if m.sum() < 5:
            continue
        per_seed = [metrics(y_csv[m], p[m], k) for _, p, _ in seeds]
        b = boot(y_csv[m], p_ens[m], k, rng)
        boots[name] = b
        out["strata"][name] = {
            "ensemble": metrics(y_csv[m], p_ens[m], k),
            "ensemble_accuracy_ci95": b["accuracy_ci95"],
            "ensemble_auc_ci95": b["auc_ci95"],
            "per_seed_mean": {key: float(np.nanmean([s[key] for s in per_seed]))
                              for key in ("accuracy", "macro_f1", "auc")},
            "per_seed_std": {key: float(np.nanstd([s[key] for s in per_seed]))
                             for key in ("accuracy", "macro_f1", "auc")},
        }
    if "confident" in boots and "remainder" in boots:
        dacc = boots["confident"]["_acc"] - boots["remainder"]["_acc"]
        out["confident_minus_remainder_accuracy"] = {
            "point": out["strata"]["confident"]["ensemble"]["accuracy"]
            - out["strata"]["remainder"]["ensemble"]["accuracy"],
            "ci95": [float(np.percentile(dacc, 2.5)),
                     float(np.percentile(dacc, 97.5))]}
        a, r = boots["confident"]["_auc"], boots["remainder"]["_auc"]
        if len(a) and len(r):
            n = min(len(a), len(r))
            dauc = a[:n] - r[:n]
            out["confident_minus_remainder_auc"] = {
                "point": out["strata"]["confident"]["ensemble"]["auc"]
                - out["strata"]["remainder"]["ensemble"]["auc"],
                "ci95": [float(np.percentile(dauc, 2.5)),
                         float(np.percentile(dauc, 97.5))]}

    for v in np.unique(med[~np.isnan(med)]):
        m = med == v
        out["by_median"][f"{v:.1f}"] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_csv[m], p_ens[m].argmax(1)))}

    pred, conf = p_ens.argmax(1), p_ens.max(1)
    correct = pred == y_csv
    out["mean_confidence"] = {"correct": float(conf[correct].mean()),
                              "wrong": float(conf[~correct].mean())
                              if (~correct).any() else None}
    return out


def fmt(x, ci=None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def write_markdown(rows: list[dict]) -> Path:
    L = ["# Agreement-stratified test performance", "",
         "Seed-averaged ensemble, percentile-bootstrap 95% CIs. *Confident* = "
         "no midpoint median, at least three readers, no disagreement flag "
         "(`src/common/agreement.py`). Generated by "
         "`scripts/stratify_by_agreement.py`.", "",
         "| rep | model | cohort | n conf / rem | acc confident | acc remainder "
         "| Δ acc | AUC confident | AUC remainder |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["strata"]
        if "confident" not in s or "remainder" not in s:
            continue
        c, m = s["confident"], s["remainder"]
        d = r.get("confident_minus_remainder_accuracy")
        L.append(
            f"| {r['rep']} | {r['tag']} | {r['cohort']} "
            f"| {c['ensemble']['n']} / {m['ensemble']['n']} "
            f"| {fmt(c['ensemble']['accuracy'], c['ensemble_accuracy_ci95'])} "
            f"| {fmt(m['ensemble']['accuracy'], m['ensemble_accuracy_ci95'])} "
            f"| {fmt(d['point'], d['ci95']) if d else '—'} "
            f"| {fmt(c['ensemble']['auc'], c['ensemble_auc_ci95'])} "
            f"| {fmt(m['ensemble']['auc'], m['ensemble_auc_ci95'])} |")
    conf_runs = [r for r in rows if r["cohort"] == "confident"]
    if conf_runs:
        L += ["", "## Trained and tested on the confident cohort only", "",
              "| rep | model | n test | accuracy | macro-F1 | AUC |",
              "|---|---|---|---|---|---|"]
        for r in conf_runs:
            e = r["overall"]["ensemble"]
            L.append(f"| {r['rep']} | {r['tag']} | {e['n']} | "
                     f"{fmt(e['accuracy'])} | {fmt(e['macro_f1'])} | "
                     f"{fmt(e['auc'])} |")
    out = C.RESULTS_DIR / "AGREEMENT.md"
    out.write_text("\n".join(L) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(REPS))
    ap.add_argument("--tags", nargs="+", default=list(TAGS))
    args = ap.parse_args()
    # Small strata can lack a class, making AUC undefined; those are reported as —.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(0)
    outdir = C.RESULTS_DIR / "agreement"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in args.reps:
        for tag in args.tags:
            r = analyse(rep, tag, rng)
            if r is None:
                continue
            (outdir / f"{rep}__{tag}.json").write_text(json.dumps(r, indent=2))
            rows.append(r)
            c = r["strata"].get("confident", {}).get("ensemble", {})
            m = r["strata"].get("remainder", {}).get("ensemble", {})
            print(f"{rep:5s} {tag:24s} seeds {r['seeds']} | confident "
                  f"n={c.get('n')} acc {fmt(c.get('accuracy'))} | remainder "
                  f"n={m.get('n')} acc {fmt(m.get('accuracy'))}")
    if not rows:
        raise SystemExit("no runs with saved test probabilities found under "
                         f"{C.RESULTS_DIR}; train with --eval-test first")
    # Rebuild the summary from every saved JSON, so runs analysed in separate
    # sessions (e.g. the Kaggle 3D parts) all appear in AGREEMENT.md.
    order = {(r, t): i for i, (r, t) in
             enumerate((r, t) for r in REPS for t in TAGS)}
    every = [json.loads(f.read_text()) for f in outdir.glob("*.json")]
    every.sort(key=lambda r: order.get((r["rep"], r["tag"]), len(order)))
    print(f"wrote {write_markdown(every)}")


if __name__ == "__main__":
    main()
