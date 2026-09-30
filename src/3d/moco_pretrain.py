"""MoCo v2 self-supervised pretraining for the 3D encoder (Model B, stage 1).

Frozen team decision: MoCo v2-style contrastive pretraining, on the shared pool
`metadata/ssl_pretrain_train.csv` (5133 nodules, TRAIN patients only -- verified
to contain zero validation or test patients). No malignancy labels are used.

Single-device note: MoCo's shuffle-BN exists to stop the network cheating by
reading batch statistics shared between the query and key batches. With one
device we approximate it by randomly permuting the key batch and forwarding it
in chunks, so each chunk gets its own BN statistics.

  python src/2d/moco_pretrain.py --epochs 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import config as C
from src.common.data import assert_no_leakage, load_splits, load_ssl_pool
from src.common.data3d import VolumeMoCoDataset, volume_train_statistics
from src.common.seed import device, set_seed
from model import build_encoder  # noqa: E402


class ProjectionHead(nn.Module):
    """MoCo v2 2-layer MLP head."""

    def __init__(self, dim_in: int = 512, hidden: int = 512, dim_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, dim_out))

    def forward(self, x):
        return self.net(x)


class MoCo(nn.Module):
    def __init__(self, in_channels: int = 1, dim: int = 128, K: int = 4096,
                 m: float = 0.999, T: float = 0.2):
        super().__init__()
        self.K, self.m, self.T = K, m, T
        self.encoder_q = build_encoder(in_channels)
        self.encoder_k = build_encoder(in_channels)
        self.head_q = ProjectionHead(dim_in=self.encoder_q.out_dim, dim_out=dim)
        self.head_k = ProjectionHead(dim_in=self.encoder_k.out_dim, dim_out=dim)
        for pq, pk in ((self.encoder_q, self.encoder_k), (self.head_q, self.head_k)):
            pk.load_state_dict(pq.state_dict())
            for p in pk.parameters():
                p.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in ((self.encoder_q, self.encoder_k), (self.head_q, self.head_k)):
            for pq, pk in zip(q.parameters(), k.parameters()):
                pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)
            for bq, bk in zip(q.buffers(), k.buffers()):
                bk.data.copy_(bq.data)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor):
        b = keys.shape[0]
        ptr = int(self.ptr)
        if ptr + b <= self.K:
            self.queue[:, ptr:ptr + b] = keys.T
        else:                                     # wrap around
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :b - first] = keys[first:].T
        self.ptr[0] = (ptr + b) % self.K

    @torch.no_grad()
    def _key_forward(self, xk: torch.Tensor, chunks: int = 4) -> torch.Tensor:
        """Shuffle-BN approximation: permute, forward in chunks, unpermute."""
        n = xk.shape[0]
        perm = torch.randperm(n, device=xk.device)
        inv = torch.argsort(perm)
        outs = [self.head_k(self.encoder_k(c))
                for c in torch.chunk(xk[perm], chunks)]
        return F.normalize(torch.cat(outs), dim=1)[inv]

    def forward(self, xq: torch.Tensor, xk: torch.Tensor):
        q = F.normalize(self.head_q(self.encoder_q(xq)), dim=1)
        self._momentum_update()
        k = self._key_forward(xk)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        self._enqueue(k)
        return logits, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.004)   # MoCo v2 0.03 @ bs 256
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--T", type=float, default=0.2)
    ap.add_argument("--m", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch-size-override", type=int, default=0)
    args = ap.parse_args()

    out = C.RESULTS_DIR / "3d" / "moco"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dev = device()

    splits = load_splits()
    ssl = load_ssl_pool()
    assert_no_leakage(splits, ssl)
    print(f"SSL pool {len(ssl)} nodules / {ssl[C.PATIENT_COL].nunique()} patients "
          "- no val/test patients present")

    mean, std = volume_train_statistics(splits["train"])
    ds = VolumeMoCoDataset(ssl, mean, std, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers,
                    persistent_workers=args.workers > 0)
    print(f"usable volumes {len(ds)} | batches/epoch {len(dl)} | device {dev}")

    moco = MoCo(in_channels=1, K=args.K, m=args.m, T=args.T).to(dev)
    params = [p for p in moco.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    hist = {"epoch": [], "loss": [], "acc": [], "lr": []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ds.set_epoch(ep)
        moco.train()
        ls, correct, n = 0.0, 0, 0
        for xq, xk in dl:
            xq, xk = xq.to(dev), xk.to(dev)
            logits, labels = moco(xq, xk)
            loss = crit(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ls += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
        sched.step()
        hist["epoch"].append(ep); hist["loss"].append(ls / n)
        hist["acc"].append(correct / n); hist["lr"].append(opt.param_groups[0]["lr"])
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} | InfoNCE loss {ls/n:.4f} | "
                  f"instance-discrimination acc {correct/n:.4f} | "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    torch.save(moco.encoder_q.state_dict(), out / "moco_encoder.pt")
    (out / "moco_history.json").write_text(json.dumps(
        {"args": vars(args), "history": hist,
         "minutes": (time.time() - t0) / 60,
         "ssl_nodules": len(ds),
         "ssl_patients": int(ssl[C.PATIENT_COL].nunique())}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["epoch"], hist["loss"]); ax[0].set_title("MoCo v2 InfoNCE loss")
    ax[1].plot(hist["epoch"], hist["acc"]); ax[1].set_title("Instance-discrimination accuracy")
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "moco_curves.png", dpi=150)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out/'moco_encoder.pt'}")


if __name__ == "__main__":
    main()
