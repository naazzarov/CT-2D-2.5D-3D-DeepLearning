"""Shared, frozen configuration for the 2D/2.5D/3D comparison.

Every value here is part of the team protocol. Changing any of it breaks
comparability between the three approaches, so it must be agreed jointly
(see the team rule in docs/project_proposal.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- dataset root
# Override with:  export CT_DATA_ROOT=/path/to/final_team_dataset_v2_3class
DATA_ROOT = Path(
    os.environ.get(
        "CT_DATA_ROOT",
        Path.home() / "Documents" / "final_team_dataset_v2_3class",
    )
).expanduser()

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results"

METADATA = DATA_ROOT / "metadata"
SPLIT_CSV = {
    "train": METADATA / "supervised_train.csv",
    "val": METADATA / "supervised_validation.csv",
    "test": METADATA / "supervised_test.csv",
}
SSL_CSV = METADATA / "ssl_pretrain_train.csv"

# Binary sensitivity experiment. Zaineb's package; override with CT_BINARY_ROOT.
BINARY_ROOT = Path(
    os.environ.get(
        "CT_BINARY_ROOT",
        Path.home() / "Documents" / "binary_sensitivity_team_package",
    )
).expanduser()
BINARY_CSV = {
    "train": BINARY_ROOT / "binary_train.csv",
    "val": BINARY_ROOT / "binary_validation.csv",
    "test": BINARY_ROOT / "binary_test.csv",
}

# ------------------------------------------------------------------- geometry
# Every stored crop is (104, 72, 80) at 1 mm isotropic spacing, and the nodule
# is centred: mask centre-of-mass measured over 25 random samples was
# z 51.1-52.1, y 35.2-35.7, x 39.3-39.9 -- i.e. the geometric centre.
SAMPLE_SHAPE_ZYX = (104, 72, 80)
CENTRAL_SLICE = SAMPLE_SHAPE_ZYX[0] // 2   # 52 -> the "defined central slice"
SLICE_HW = SAMPLE_SHAPE_ZYX[1:]            # (72, 80)

# 2.5D: the agreed 5 neighbouring slices, centred on CENTRAL_SLICE -> 50..54.
N_NEIGHBOUR_SLICES = 5
SLICE_WINDOW = (CENTRAL_SLICE - N_NEIGHBOUR_SLICES // 2,
                CENTRAL_SLICE + N_NEIGHBOUR_SLICES // 2 + 1)   # (50, 55)

# ------------------------------------------------------------- HU windowing
# Arrays are stored as RAW Hounsfield units (not normalised), so the window is
# our choice. Observed range across 120 random samples: minima to -3024
# (out-of-FOV padding, 16/120 samples below -1100) and maxima to +3080.
# Unclipped, those outliers dominate any normalisation, so we clip first.
# This is a standard lung window covering air, lung parenchyma and soft tissue.
HU_MIN, HU_MAX = -1000.0, 400.0

# ---------------------------------------------------------------- label space
NUM_CLASSES = 3
CLASS_NAMES = ("benign", "indeterminate", "malignant")  # class_index_v2 0,1,2
LABEL_COL = "class_index_v2"

# --- binary sensitivity experiment -------------------------------------------
# Use `binary_class_index` (0 = benign, 1 = malignant), NOT the legacy
# `binary_label` column that also exists in these CSVs: that one is inherited
# from the V2 metadata and is empty for 73% of rows (731/1000 in train), so
# reading it would silently corrupt the labels.
BINARY_NUM_CLASSES = 2
BINARY_CLASS_NAMES = ("benign", "malignant")
BINARY_LABEL_COL = "binary_class_index"

TASKS = {
    "3class": {"num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
               "label_col": LABEL_COL, "csv": SPLIT_CSV},
    "binary": {"num_classes": BINARY_NUM_CLASSES,
               "class_names": BINARY_CLASS_NAMES,
               "label_col": BINARY_LABEL_COL, "csv": BINARY_CSV},
}
ID_COL = "consensus_nodule_id"
PATIENT_COL = "patient_id"
PATH_COL = "sample_path"

# --------------------------------------------------------------- model select
PRIMARY_METRIC = "macro_f1"   # validation macro-F1 selects the checkpoint
SEEDS = (0, 1, 2, 3, 4)       # every result is reported as mean +/- std
