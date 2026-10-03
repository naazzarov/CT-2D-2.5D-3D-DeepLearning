"""2D supervised training: Model A (random init) and Model B (MoCo fine-tune).

Both models use an identical recipe -- only the initial encoder weights differ,
so the A-vs-B gap measures the effect of self-supervised pretraining and
nothing else.

  Model A:  python src/2d/train_supervised.py --init random --seed 0
  Model B:  python src/2d/train_supervised.py --init moco \
                --moco-ckpt results/2d/moco/moco_encoder.pt --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import config as C
from src.common.data import (COHORTS, CentralSliceDataset, apply_cohort,
                             assert_no_leakage, class_weights, load_splits,
                             load_ssl_pool, train_statistics,
                             verify_binary_counts)
from src.common.metrics import bootstrap_ci, compute_all, format_report
from src.common.plots import confusion_figure, training_curves
from src.common.seed import device, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ResNet18Classifier, n_params  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, dev, criterion,
             class_names=C.CLASS_NAMES) -> tuple[dict, float, np.ndarray, np.ndarray]:
    model.eval()
    probs, targets, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        targets.append(y.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)
    y_pred = y_prob.argmax(1)
    return (compute_all(y_true, y_pred, y_prob, class_names),
            loss_sum / n, y_true, y_prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["3class", "binary"], default="3class")
    ap.add_argument("--rep", choices=["2d", "2p5d"], default="2d",
                    help="2d = central slice (1 channel); "
                         "2p5d = 5 neighbouring slices (5 channels)")
    ap.add_argument("--init", choices=["random", "moco"], default="random")
    ap.add_argument("--moco-ckpt", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--aug", choices=["none", "mild", "strong"], default="mild")
    ap.add_argument("--cohort", choices=COHORTS, default="all",
                    help="'confident' trains and evaluates only on nodules with "
                         "concordant radiologist labels (src/common/agreement.py)")
    ap.add_argument("--eval-test", action="store_true",
                    help="Evaluate the TEST split. Off during hyperparameter "
                         "search so the test set stays untouched until the "
                         "protocol is frozen.")
    args = ap.parse_args()

    spec = C.TASKS[args.task]
    num_classes, class_names = spec["num_classes"], spec["class_names"]
    default_tag = ("model_a" if args.init == "random" else "model_b")
    if args.task == "binary":
        default_tag = "binary_supervised"
    if args.cohort != "all":
        default_tag += f"_{args.cohort}"
    tag = args.tag or default_tag
    out = C.RESULTS_DIR / args.rep / tag / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dev = device()

    splits = load_splits(args.task)
    reference = load_splits("3class") if args.task == "binary" else None
    if args.task == "binary":
        verify_binary_counts(splits)
    splits = apply_cohort(splits, args.cohort, args.task)
    assert_no_leakage(splits, load_ssl_pool(), reference=reference)
    print(f"task {args.task} ({num_classes} classes: {', '.join(class_names)})")
    print(f"leakage check passed | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])}")

    # Intensity statistics always come from the 3-class TRAIN split so that the
    # binary and 3-class models see identically scaled inputs.
    mean, std = train_statistics(load_splits("3class")["train"], rep=args.rep)
    print(f"train-set windowed intensity: mean {mean:.4f} std {std:.4f}")

    ds = {
        "train": CentralSliceDataset(splits["train"], mean, std, train=True,
                                     seed=args.seed, aug=args.aug,
                                     task=args.task, rep=args.rep),
        "val": CentralSliceDataset(splits["val"], mean, std, task=args.task,
                                   rep=args.rep),
        "test": CentralSliceDataset(splits["test"], mean, std, task=args.task,
                                    rep=args.rep),
    }
    dl = {
        "train": DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                            drop_last=True, num_workers=args.workers,
                            persistent_workers=args.workers > 0),
        "val": DataLoader(ds["val"], batch_size=128, num_workers=0),
        "test": DataLoader(ds["test"], batch_size=128, num_workers=0),
    }

    in_ch = C.N_NEIGHBOUR_SLICES if args.rep == "2p5d" else 1
    model = ResNet18Classifier(in_channels=in_ch, num_classes=num_classes,
                               dropout=args.dropout).to(dev)
    if args.init == "moco":
        if not args.moco_ckpt:
            raise SystemExit("--init moco requires --moco-ckpt")
        state = torch.load(args.moco_ckpt, map_location="cpu")
        miss, unexp = model.load_encoder(state)
        print(f"loaded MoCo encoder from {args.moco_ckpt} "
              f"(missing {len(miss)}, unexpected {len(unexp)})")
    print(f"trainable params {n_params(model)/1e6:.2f}M | device {dev}")

    w = class_weights(splits["train"], args.task).to(dev)
    print(f"class weights {w.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss",
                            "train_macro_f1", "val_macro_f1", "lr")}
    best_f1, best_epoch, since = -1.0, -1, 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        ds["train"].set_epoch(ep)
        model.train()
        loss_sum, n, tr_pred, tr_true = 0.0, 0, [], []
        for x, y in dl["train"]:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            tr_pred.append(logits.detach().argmax(1).cpu().numpy())
            tr_true.append(y.cpu().numpy())
        sched.step()

        tr_f1 = f1_score(np.concatenate(tr_true), np.concatenate(tr_pred),
                         average="macro", zero_division=0)
        vm, vl, _, _ = evaluate(model, dl["val"], dev, criterion, class_names)

        hist["epoch"].append(ep)
        hist["train_loss"].append(loss_sum / n)
        hist["val_loss"].append(vl)
        hist["train_macro_f1"].append(float(tr_f1))
        hist["val_macro_f1"].append(vm["macro_f1"])
        hist["lr"].append(opt.param_groups[0]["lr"])

        flag = ""
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, since = vm["macro_f1"], ep, 0
            torch.save(model.state_dict(), out / "best.pt")
            flag = "  <- best"
        else:
            since += 1
        print(f"ep {ep:3d} | train loss {loss_sum/n:.4f} f1 {tr_f1:.4f} "
              f"| val loss {vl:.4f} f1 {vm['macro_f1']:.4f}{flag}", flush=True)

        if since >= args.patience:
            print(f"early stopping at epoch {ep} "
                  f"(no val improvement for {args.patience} epochs)")
            break

    mins = (time.time() - t0) / 60
    print(f"\ntrained {len(hist['epoch'])} epochs in {mins:.1f} min | "
          f"best val macro F1 {best_f1:.4f} @ epoch {best_epoch}")

    # ---- final test evaluation, best checkpoint only, once -------------------
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    val_m, _, _, _ = evaluate(model, dl["val"], dev, criterion, class_names)
    test_m = None
    if args.eval_test:
        test_m, _, y_true, y_prob = evaluate(model, dl["test"], dev, criterion,
                                             class_names)
        lo, hi = bootstrap_ci(y_true, y_prob.argmax(1), "macro_f1", seed=args.seed)
        test_m["macro_f1_ci95"] = [lo, hi]
        np.save(out / "test_probabilities.npy", y_prob)
        np.save(out / "test_targets.npy", y_true)

    result = {
        "tag": tag, "task": args.task, "num_classes": num_classes,
        "class_names": list(class_names), "init": args.init, "seed": args.seed,
        "epochs_run": len(hist["epoch"]), "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1, "train_minutes": mins,
        "params": n_params(model),
        "rep": args.rep, "in_channels": in_ch,
        "hu_window": [C.HU_MIN, C.HU_MAX], "central_slice": C.CENTRAL_SLICE,
        "train_mean": mean, "train_std": std,
        "args": vars(args), "validation": val_m, "test": test_m, "history": hist,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    training_curves(hist, out / "curves.png",
                    f"2D {tag} (seed {args.seed}) -- ResNet-18, central slice")
    print()
    print(format_report(f"2D {tag} seed {args.seed} -- VALIDATION", val_m,
                        class_names))
    if test_m is not None:
        confusion_figure(test_m["confusion_matrix"], out / "confusion_test.png",
                         f"2D {tag} seed {args.seed} -- test", class_names)
        print()
        print(format_report(f"2D {tag} seed {args.seed} -- TEST", test_m,
                            class_names))
        print(f"  macro F1 95% CI    [{test_m['macro_f1_ci95'][0]:.4f}, "
              f"{test_m['macro_f1_ci95'][1]:.4f}]")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
