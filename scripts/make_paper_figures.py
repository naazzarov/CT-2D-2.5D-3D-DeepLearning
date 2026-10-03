"""Figures for paper/main.tex, built from the committed per-seed results.

  python scripts/make_paper_figures.py      # -> paper/figures/*.pdf

Everything is read from results/<rep>/<tag>/seed*/result.json,
results/<rep>/moco/moco_history.json and results/agreement/*.json. Until
scripts/stratify_by_agreement.py has been run, fig_ceiling falls back to the
2D values recorded in results/2d/ANALYSIS.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"

REPS = [("2d", "2D"), ("2p5d", "2.5D"), ("3d", "3D")]
MODELS3 = [("model_a_noaug", "A (supervised)"), ("model_b_noaug", "B (MoCo)"),
           ("model_c_mild", "C (augmented)")]
REP_COLORS = {"2d": "#4C72B0", "2p5d": "#DD8452", "3d": "#55A868"}
MODEL_COLORS = {"model_a_noaug": "#8DA0CB", "model_b_noaug": "#FC8D62",
                "model_c_mild": "#66C2A5", "binary_mild": "#E78AC3"}
CLASS3 = ("benign", "indeterminate", "malignant")
MAJORITY_3CLASS_F1 = 0.2138
MAJORITY_BINARY_F1 = 0.3589

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def runs(rep: str, tag: str) -> list[dict]:
    out = []
    for f in sorted((RESULTS / rep / tag).glob("seed*/result.json")):
        r = json.loads(f.read_text())
        if r.get("test"):
            out.append(r)
    return out


def test_values(rep: str, tag: str, key: str) -> np.ndarray:
    return np.array([r["test"][key] for r in runs(rep, tag)])


def auc_key(r: dict) -> str:
    return "roc_auc" if "roc_auc" in r["test"] else "roc_auc_ovr_macro"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def grouped_bars(ax, groups, series, values, colors, labels, baseline=None,
                 ylim=None, ylabel=""):
    """values[s][g] -> per-seed array. Bars are means, error bars one std."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        means = [values[s][g].mean() for g in groups]
        stds = [values[s][g].std() for g in groups]
        ax.bar(pos, means, width * 0.92, yerr=stds, capsize=2.5,
               color=colors[s], edgecolor="black", linewidth=0.5,
               error_kw={"linewidth": 0.8}, label=labels[s])
        for p, g in zip(pos, groups):
            v = values[s][g]
            ax.scatter(np.full(len(v), p), v, s=7, color="black", zorder=3,
                       linewidths=0)
    if baseline is not None:
        ax.axhline(baseline, ls=":", c="grey", lw=1)
        ax.text(-0.45, baseline + 0.01, "majority class", va="bottom",
                ha="left", fontsize=6.5, color="dimgrey", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_xticks(x)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def fig_main_comparison() -> None:
    reps = [r for r, _ in REPS]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [3, 2]})

    tags = [t for t, _ in MODELS3]
    vals = {t: {r: test_values(r, t, "macro_f1") for r in reps} for t in tags}
    grouped_bars(axes[0], reps, tags, vals, MODEL_COLORS, dict(MODELS3),
                 baseline=MAJORITY_3CLASS_F1, ylim=(0.0, 0.72),
                 ylabel="test macro-F1")
    axes[0].set_xticklabels([lbl for _, lbl in REPS])
    axes[0].set_title("(a) Three-class, by model and representation")
    axes[0].legend(loc="lower left", ncol=3, frameon=False,
                   bbox_to_anchor=(0, 1.08, 1, 0.1), mode="expand")

    metrics = ["macro_f1", "auc"]
    bvals = {}
    for r in reps:
        rr = runs(r, "binary_mild")
        bvals[r] = {"macro_f1": np.array([x["test"]["macro_f1"] for x in rr]),
                    "auc": np.array([x["test"][auc_key(x)] for x in rr])}
    vals_b = {r: {m: bvals[r][m] for m in metrics} for r in reps}
    grouped_bars(axes[1], metrics, reps, vals_b, REP_COLORS, dict(REPS),
                 baseline=MAJORITY_BINARY_F1, ylim=(0.0, 1.0), ylabel="test score")
    axes[1].set_xticklabels(["macro-F1", "ROC-AUC"])
    axes[1].set_title("(b) Binary, by representation")
    axes[1].legend(loc="lower left", ncol=3, frameon=False,
                   bbox_to_anchor=(0, 1.08, 1, 0.1), mode="expand")
    fig.tight_layout()
    save(fig, "fig_main_comparison")


def pooled_confusion(rep: str, tag: str) -> np.ndarray:
    return np.sum([np.asarray(r["test"]["confusion_matrix"])
                   for r in runs(rep, tag)], axis=0)


def fig_confusion() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    short = ("benign", "indet.", "malig.")
    for ax, (rep, lbl) in zip(axes, REPS):
        cm = pooled_confusion(rep, "model_c_mild")
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center",
                        va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short if rep == "2d" else [""] * 3)
        ax.set_xlabel("predicted")
        if rep == "2d":
            ax.set_ylabel("true")
        ax.set_title(f"{lbl}, Model C")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig_confusion")


def agreement(rep: str, tag: str) -> dict | None:
    f = RESULTS / "agreement" / f"{rep}__{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def fig_ceiling() -> None:
    if agreement("2d", "binary_mild") is not None:
        fig_ceiling_from_analysis()
        return
    # Fallback until predictions are committed: values from results/2d/ANALYSIS.md
    # (2D binary, probabilities averaged over seeds).
    medians = ["1.0", "2.0", "3.5", "4.0", "4.5", "5.0"]
    acc = [0.898, 0.905, 0.667, 0.697, 0.917, 0.900]
    n = [49, 63, 27, 33, 12, 20]
    midpoint = {"3.5"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5),
                             gridspec_kw={"width_ratios": [3, 2]})
    colors = ["#C44E52" if m in midpoint else "#4C72B0" for m in medians]
    bars = axes[0].bar(medians, acc, color=colors, edgecolor="black", lw=0.5)
    for b, k in zip(bars, n):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"n={k}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("median radiologist malignancy rating")
    axes[0].set_ylabel("binary test accuracy")
    axes[0].set_title("(a) Accuracy by median rating")
    axes[0].grid(axis="y", alpha=0.3)
    subsets = ["confident\n(n=55)", "remainder\n(n=154)"]
    x = np.arange(2)
    axes[1].bar(x - 0.18, [0.945, 0.805], 0.34, label="accuracy",
                color="#4C72B0", edgecolor="black", lw=0.5)
    axes[1].bar(x + 0.18, [0.987, 0.902], 0.34, label="ROC-AUC",
                color="#55A868", edgecolor="black", lw=0.5)
    for xi, (a, u) in enumerate([(0.945, 0.987), (0.805, 0.902)]):
        axes[1].text(xi - 0.18, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        axes[1].text(xi + 0.18, u + 0.01, f"{u:.3f}", ha="center", fontsize=7)
    axes[1].set_xticks(x, subsets)
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_title("(b) By label confidence")
    axes[1].legend(frameon=False, loc="upper right", ncol=2)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_ceiling")


def fig_ceiling_from_analysis() -> None:
    """Binary accuracy by median rating (2D) and confident vs remainder accuracy
    for every representation, from scripts/stratify_by_agreement.py output."""
    a2 = agreement("2d", "binary_mild")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5),
                             gridspec_kw={"width_ratios": [3, 2]})
    meds = sorted(a2["by_median"], key=float)
    acc = [a2["by_median"][m]["accuracy"] for m in meds]
    n = [a2["by_median"][m]["n"] for m in meds]
    colors = ["#C44E52" if float(m) % 1 else "#4C72B0" for m in meds]
    bars = axes[0].bar(meds, acc, color=colors, edgecolor="black", lw=0.5)
    for b, k in zip(bars, n):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"n={k}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_xlabel("median radiologist malignancy rating")
    axes[0].set_ylabel("binary test accuracy")
    axes[0].set_title("(a) 2D accuracy by median rating")
    axes[0].grid(axis="y", alpha=0.3)

    reps = [(r, l) for r, l in REPS if agreement(r, "binary_mild")]
    x = np.arange(len(reps))
    for off, stratum, color in ((-0.18, "confident", "#4C72B0"),
                                (0.18, "remainder", "#DD8452")):
        vals, lo, hi = [], [], []
        for r, _ in reps:
            s = agreement(r, "binary_mild")["strata"][stratum]
            v = s["ensemble"]["accuracy"]
            ci = s["ensemble_accuracy_ci95"]
            vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        axes[1].bar(x + off, vals, 0.34, yerr=[lo, hi], capsize=2.5,
                    color=color, edgecolor="black", lw=0.5,
                    error_kw={"linewidth": 0.8}, label=stratum)
    axes[1].set_xticks(x, [l for _, l in reps])
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_ylabel("binary test accuracy")
    axes[1].set_title("(b) Confident vs remainder, 95% CI")
    axes[1].legend(frameon=False, loc="upper right", ncol=2)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_ceiling")


def fig_confident_cohort() -> None:
    """Full cohort vs confident-only cohort, test macro-F1 per representation."""
    pairs = [("model_c_mild", "model_c_mild_confident", "three-class"),
             ("binary_mild", "binary_mild_confident", "binary")]
    pairs = [p for p in pairs if any(runs(r, p[1]) for r, _ in REPS)]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.5 * len(pairs), 2.6),
                             squeeze=False)
    reps = [r for r, _ in REPS]
    for ax, (full, conf, title) in zip(axes[0], pairs):
        vals = {"full": {r: test_values(r, full, "macro_f1") for r in reps},
                "confident": {r: test_values(r, conf, "macro_f1") for r in reps}}
        vals = {k: {r: v if len(v) else np.array([np.nan]) for r, v in d.items()}
                for k, d in vals.items()}
        grouped_bars(ax, reps, ["full", "confident"], vals,
                     {"full": "#BBBBBB", "confident": "#4C72B0"},
                     {"full": "all nodules", "confident": "confident labels only"},
                     ylim=(0.0, 1.0), ylabel="test macro-F1")
        ax.set_xticklabels([l for _, l in REPS])
        ax.set_title(title)
    axes[0][0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_confident_cohort")


def fig_curves() -> None:
    configs = [("model_a_noaug", "no augmentation"),
               ("model_c_mild", "mild augmentation")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (tag, lbl) in zip(axes, configs):
        for r in runs("2d", tag):
            h = r["history"]
            a = 0.9 if r["seed"] == 0 else 0.35
            ax.plot(h["epoch"], h["train_macro_f1"], c="#C44E52", alpha=a, lw=1,
                    label="train" if r["seed"] == 0 else None)
            ax.plot(h["epoch"], h["val_macro_f1"], c="#4C72B0", alpha=a, lw=1,
                    label="validation" if r["seed"] == 0 else None)
            ax.axvline(r["best_epoch"], c="grey", ls=":", lw=0.8, alpha=a)
        ax.axhline(0.9, c="black", ls="--", lw=0.6)
        ax.set_title(f"2D Model {'A' if tag == 'model_a_noaug' else 'C'}, {lbl}")
        ax.set_xlabel("epoch")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_curves")


def fig_moco() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for rep, lbl in REPS:
        h = json.loads((RESULTS / rep / "moco" / "moco_history.json")
                       .read_text())["history"]
        axes[0].plot(h["epoch"], h["loss"], c=REP_COLORS[rep], lw=1, label=lbl)
        axes[1].plot(h["epoch"], h["acc"], c=REP_COLORS[rep], lw=1, label=lbl)
    axes[0].set_ylabel("InfoNCE loss")
    axes[1].set_ylabel("instance-discrimination accuracy")
    for ax, t in zip(axes, ["(a) Pretext loss", "(b) Pretext accuracy"]):
        ax.set_xlabel("pretraining epoch")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_moco")


def fig_cost() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    markers = {"model_a_noaug": "o", "model_b_noaug": "s", "model_c_mild": "^"}
    for rep, lbl in REPS:
        for tag, mlbl in MODELS3:
            rr = runs(rep, tag)
            t = np.mean([r["train_minutes"] for r in rr])
            f = np.mean([r["test"]["macro_f1"] for r in rr])
            s = np.std([r["test"]["macro_f1"] for r in rr])
            ax.errorbar(t, f, yerr=s, fmt=markers[tag], c=REP_COLORS[rep],
                        ms=5, capsize=2, lw=0.8, mec="black", mew=0.4)
    for rep, lbl in REPS:
        ax.scatter([], [], c=REP_COLORS[rep], label=lbl, s=20)
    for tag, mlbl in MODELS3:
        ax.scatter([], [], marker=markers[tag], c="white", edgecolors="black",
                   label=mlbl, s=20)
    ax.set_xscale("log")
    ticks = [2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("supervised training minutes per seed (log scale)")
    ax.set_ylabel("test macro-F1")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left",
              bbox_to_anchor=(0, -0.22))
    fig.tight_layout()
    save(fig, "fig_cost")


if __name__ == "__main__":
    fig_main_comparison()
    fig_confusion()
    fig_ceiling()
    fig_curves()
    fig_moco()
    fig_cost()
    fig_confident_cohort()
