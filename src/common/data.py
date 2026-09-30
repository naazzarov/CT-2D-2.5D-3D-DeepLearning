"""Manifest loading, leakage checks and the 2D slice datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common import config as C
from src.common import cache as cache_mod
from src.common import cache25 as cache25_mod
from src.common.transforms import augment_moco, augment_supervised, window_hu


# ------------------------------------------------------------------ manifests
def load_splits(task: str = "3class") -> dict[str, pd.DataFrame]:
    """Load the frozen split CSVs for a task, exactly as provided.

    task="3class" -> supervised_{train,validation,test}.csv, label class_index_v2
    task="binary" -> binary_{train,validation,test}.csv,     label binary_class_index
    """
    spec = C.TASKS[task]
    out = {}
    for name, path in spec["csv"].items():
        df = pd.read_csv(path, low_memory=False)
        col = spec["label_col"]
        if df[col].isna().any():
            raise AssertionError(
                f"{path.name}: column '{col}' has {int(df[col].isna().sum())} "
                "empty values -- wrong label column?")
        df[col] = df[col].astype(int)
        bad = set(df[col].unique()) - set(range(spec["num_classes"]))
        if bad:
            raise AssertionError(f"{path.name}: unexpected labels {bad}")
        out[name] = df
    return out


def verify_binary_counts(splits: dict[str, pd.DataFrame]) -> None:
    """Check the binary cohort against the counts Zaineb specified."""
    expected = {"train": (1000, 622, 378), "val": (219, 134, 85),
                "test": (209, 117, 92)}
    for name, (tot, n0, n1) in expected.items():
        d = splits[name]
        c = d[C.BINARY_LABEL_COL].value_counts()
        got = (len(d), int(c.get(0, 0)), int(c.get(1, 0)))
        if got != (tot, n0, n1):
            raise AssertionError(
                f"binary {name}: expected total/benign/malignant {(tot, n0, n1)}, "
                f"got {got}")
    print("binary counts match the agreed specification "
          "(1000/219/209; 622+378, 134+85, 117+92)")


def load_ssl_pool() -> pd.DataFrame:
    return pd.read_csv(C.SSL_CSV, low_memory=False)


def assert_no_leakage(splits: dict[str, pd.DataFrame], ssl: pd.DataFrame | None = None,
                      reference: dict[str, pd.DataFrame] | None = None):
    """Hard guard: the protocol's central promise, re-checked on every run.

    Raises rather than warns -- a silent leak invalidates every number we report.
    """
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = set(splits[a][C.PATIENT_COL]), set(splits[b][C.PATIENT_COL])
            if pa & pb:
                raise AssertionError(f"patient leakage {a}/{b}: {len(pa & pb)}")
            na, nb = set(splits[a][C.ID_COL]), set(splits[b][C.ID_COL])
            if na & nb:
                raise AssertionError(f"nodule leakage {a}/{b}: {len(na & nb)}")
    if ssl is not None:
        sp = set(ssl[C.PATIENT_COL])
        for held in ("val", "test"):
            bad = sp & set(splits[held][C.PATIENT_COL])
            if bad:
                raise AssertionError(
                    f"SSL pool contains {len(bad)} {held} patients -- "
                    "pretraining would contaminate the held-out evaluation"
                )
    if reference is not None:
        # Every split must be a SUBSET of the corresponding reference split:
        # proves no patient was moved between splits when deriving the cohort.
        for name in splits:
            sub = set(splits[name][C.PATIENT_COL])
            ref = set(reference[name][C.PATIENT_COL])
            if not sub <= ref:
                raise AssertionError(
                    f"{name}: {len(sub - ref)} patients are not in the original "
                    f"{name} split -- the split was changed, not just filtered")
    return True


def class_weights(train: pd.DataFrame, task: str = "3class") -> torch.Tensor:
    """Inverse-frequency weights, N / (K * n_k), so the model cannot coast by
    predicting the majority class (indeterminate is 47% of the 3-class train
    split; benign is 62% of the binary one)."""
    spec = C.TASKS[task]
    k = spec["num_classes"]
    counts = train[spec["label_col"]].value_counts().reindex(range(k)).values
    w = counts.sum() / (k * counts)
    return torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------------- datasets
class CentralSliceDataset(Dataset):
    """CT slice(s) -> class label.

    rep="2d"    -> x of shape (1, 72, 80), the defined central slice 52.
    rep="2p5d"  -> x of shape (5, 72, 80), slices 50..54.

    Both go through identical windowing, augmentation and standardisation, so
    the only difference between the representations is how much context the
    channel dimension carries.
    """

    def __init__(self, df: pd.DataFrame, mean: float, std: float,
                 train: bool = False, seed: int = 0, aug: str = "mild",
                 task: str = "3class", rep: str = "2d"):
        arr, index = (cache25_mod.load() if rep == "2p5d" else cache_mod.load())
        self.rep = rep
        self.arr = arr
        self.rows = np.array([index[i] for i in df[C.ID_COL]], dtype=np.int64)
        self.labels = df[C.TASKS[task]["label_col"]].to_numpy(dtype=np.int64)
        self.ids = df[C.ID_COL].tolist()
        self.mean, self.std, self.train = mean, std, train
        self.aug = aug
        self._seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Augmentation must differ every epoch; the RNG seed includes the epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x = window_hu(np.asarray(self.arr[self.rows[i]]))
        x = torch.from_numpy(x)
        if self.rep == "2d":
            x = x.unsqueeze(0)                            # (1, H, W)
        # 2.5D is already (5, H, W); the spatial augmentation below applies the
        # SAME transform to every channel, so the slices stay aligned.
        if self.train:
            rng = np.random.default_rng((self._seed, self.epoch, i))
            x = augment_supervised(x, rng, strength=self.aug)
        x = (x - self.mean) / self.std
        return x, int(self.labels[i])


class MoCoPairDataset(Dataset):
    """Two independently augmented views of the same central slice, no labels."""

    def __init__(self, df: pd.DataFrame, mean: float, std: float, seed: int = 0,
                 rep: str = "2d"):
        arr, index = (cache25_mod.load() if rep == "2p5d" else cache_mod.load())
        self.rep = rep
        self.arr = arr
        self.rows = np.array(
            [index[i] for i in df[C.ID_COL] if i in index], dtype=np.int64
        )
        self.mean, self.std = mean, std
        self._seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        base = window_hu(np.asarray(self.arr[self.rows[i]]))
        base = torch.from_numpy(base)
        if self.rep == "2d":
            base = base.unsqueeze(0)
        rng = np.random.default_rng((self._seed, self.epoch, i))
        q = (augment_moco(base, rng) - self.mean) / self.std
        k = (augment_moco(base, rng) - self.mean) / self.std
        return q, k


def train_statistics(train: pd.DataFrame, rep: str = "2d") -> tuple[float, float]:
    """Mean/std of windowed intensities over the TRAINING split only."""
    arr, index = (cache25_mod.load() if rep == "2p5d" else cache_mod.load())
    rows = np.array([index[i] for i in train[C.ID_COL]], dtype=np.int64)
    vals = window_hu(np.asarray(arr[np.sort(rows)]))
    return float(vals.mean()), float(vals.std())
