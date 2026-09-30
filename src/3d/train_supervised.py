"""3D supervised training — same protocol as the 2D/2.5D arms.

  python src/3d/train_supervised.py --task 3class --aug mild --seed 0 --eval-test
  python src/3d/train_supervised.py --task 3class --init moco \
      --moco-ckpt results/3d/moco/moco_encoder.pt --aug none --seed 0 --eval-test
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import config as C
from src.common.data import (assert_no_leakage, class_weights, load_splits,
                             load_ssl_pool, verify_binary_counts)
from src.common.data3d import VolumeDataset, volume_train_statistics
from src.common.metrics import bootstrap_ci, compute_all, format_report
from src.common.plots import confusion_figure, training_curves
from src.common.seed import device, set_seed
from model import ResNet10Classifier, n_params  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, dev, criterion, class_names):
    model.eval()
    probs, targets, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0); n += y.size(0)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        targets.append(y.cpu().numpy())
    y_prob = np.concatenate(probs); y_true = np.concatenate(targets)
    return (compute_all(y_true, y_prob.argmax(1), y_prob, class_names),
            loss_sum / n, y_true, y_prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["3class", "binary"], default="3class")
    ap.add_argument("--init", choices=["random", "moco"], default="random")
    ap.add_argument("--moco-ckpt", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--aug", choices=["none", "mild", "strong"], default="mild")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()

    spec = C.TASKS[args.task]
    num_classes, class_names = spec["num_classes"], spec["class_names"]
    tag = args.tag or ("binary_mild" if args.task == "binary"
                       else ("model_a_noaug" if args.init == "random"
                             else "model_b_noaug"))
    out = C.RESULTS_DIR / "3d" / tag / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed); dev = device()
    splits = load_splits(args.task)
    ref = load_splits("3class") if args.task == "binary" else None
    if args.task == "binary":
        verify_binary_counts(splits)
    assert_no_leakage(splits, load_ssl_pool(), reference=ref)
    print(f"task {args.task} | rep 3d | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])} | device {dev}")

    mean, std = volume_train_statistics(load_splits("3class")["train"])
    print(f"volume intensity mean {mean:.4f} std {std:.4f}")

    ds = {"train": VolumeDataset(splits["train"], mean, std, train=True,
                                 seed=args.seed, aug=args.aug, task=args.task),
          "val": VolumeDataset(splits["val"], mean, std, task=args.task),
          "test": VolumeDataset(splits["test"], mean, std, task=args.task)}
    dl = {"train": DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0),
          "val": DataLoader(ds["val"], batch_size=args.batch_size,
                            num_workers=args.workers),
          "test": DataLoader(ds["test"], batch_size=args.batch_size,
                             num_workers=args.workers)}

    model = ResNet10Classifier(in_channels=1, num_classes=num_classes,
                               dropout=args.dropout).to(dev)
    if args.init == "moco":
        state = torch.load(args.moco_ckpt, map_location="cpu")
        miss, unexp = model.load_encoder(state)
        print(f"loaded MoCo encoder (missing {len(miss)}, unexpected {len(unexp)})")
    print(f"params {n_params(model)/1e6:.2f}M")

    w = class_weights(splits["train"], args.task).to(dev)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss",
                            "train_macro_f1", "val_macro_f1", "lr")}
    best_f1, best_epoch, since = -1.0, -1, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ds["train"].set_epoch(ep); model.train()
        ls, n, tp, tt = 0.0, 0, [], []
        for x, y in dl["train"]:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); opt.step()
            ls += loss.item() * y.size(0); n += y.size(0)
            tp.append(logits.detach().argmax(1).cpu().numpy()); tt.append(y.cpu().numpy())
        sched.step()
        tr_f1 = f1_score(np.concatenate(tt), np.concatenate(tp),
                         average="macro", zero_division=0)
        vm, vl, _, _ = evaluate(model, dl["val"], dev, criterion, class_names)
        for k, v in (("epoch", ep), ("train_loss", ls / n), ("val_loss", vl),
                     ("train_macro_f1", float(tr_f1)),
                     ("val_macro_f1", vm["macro_f1"]),
                     ("lr", opt.param_groups[0]["lr"])):
            hist[k].append(v)
        flag = ""
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, since = vm["macro_f1"], ep, 0
            torch.save(model.state_dict(), out / "best.pt"); flag = "  <- best"
        else:
            since += 1
        print(f"ep {ep:3d} | train loss {ls/n:.4f} f1 {tr_f1:.4f} | "
              f"val loss {vl:.4f} f1 {vm['macro_f1']:.4f}{flag}", flush=True)
        if since >= args.patience:
            print(f"early stopping at epoch {ep}"); break

    mins = (time.time() - t0) / 60
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

    (out / "result.json").write_text(json.dumps({
        "tag": tag, "task": args.task, "rep": "3d", "num_classes": num_classes,
        "class_names": list(class_names), "init": args.init, "seed": args.seed,
        "epochs_run": len(hist["epoch"]), "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1, "train_minutes": mins,
        "params": n_params(model), "hu_window": [C.HU_MIN, C.HU_MAX],
        "train_mean": mean, "train_std": std, "args": vars(args),
        "validation": val_m, "test": test_m, "history": hist}, indent=2))
    training_curves(hist, out / "curves.png", f"3D {tag} (seed {args.seed})")
    print(); print(format_report(f"3D {tag} seed {args.seed} -- VALIDATION",
                                 val_m, class_names))
    if test_m:
        confusion_figure(test_m["confusion_matrix"], out / "confusion_test.png",
                         f"3D {tag} seed {args.seed} -- test", class_names)
        print(); print(format_report(f"3D {tag} seed {args.seed} -- TEST",
                                     test_m, class_names))
    print(f"\nsaved -> {out}  ({mins:.1f} min)")


if __name__ == "__main__":
    main()
