"""Annotator-agreement metadata: the "confident label" definition.

A nodule is *confident* when its median rating is not a midpoint (2.5 / 3.5),
at least three radiologists rated it, and it carries no disagreement flag.
The same definition drives the stratified analysis
(scripts/stratify_by_agreement.py) and the confident-only training cohort
(--cohort confident in the training scripts), so the two cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MEDIAN_COL = "malignancy_median"
MIDPOINT_COL = "midpoint_median_v2"
DISAGREEMENT_COL = "disagreement_flag"
# The reader-count column name is not fixed across metadata versions.
READER_COL_CANDIDATES = ("n_readers", "num_readers", "reader_count",
                         "n_annotations", "num_annotations", "n_raters",
                         "num_raters", "annotation_count", "n_ratings")
MIN_READERS = 3


def _flag(series: pd.Series) -> np.ndarray:
    """Truthy flag column -> bool array. Missing values count as not flagged."""
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map(
            {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0,
             "nan": 0, "": 0})
    return s.fillna(0).astype(float).to_numpy() > 0


def reader_column(df: pd.DataFrame) -> str:
    for c in READER_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        "no reader-count column found; expected one of "
        f"{READER_COL_CANDIDATES}. Columns present: {sorted(df.columns)}. "
        "Add the right name to READER_COL_CANDIDATES in src/common/agreement.py.")


def _has_all(df: pd.DataFrame) -> bool:
    if any(c not in df.columns for c in (MEDIAN_COL, MIDPOINT_COL,
                                         DISAGREEMENT_COL)):
        return False
    return any(c in df.columns for c in READER_COL_CANDIDATES)


def with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with the agreement columns, joining any that are missing from
    metadata/all_nodules_v2.csv by nodule id. Row order is preserved."""
    if _has_all(df):
        return df
    from src.common import config as C
    meta = pd.read_csv(C.METADATA / "all_nodules_v2.csv", low_memory=False)
    wanted = [c for c in (MEDIAN_COL, MIDPOINT_COL, DISAGREEMENT_COL,
                          *READER_COL_CANDIDATES)
              if c in meta.columns and c not in df.columns]
    merged = df.merge(meta[[C.ID_COL, *wanted]].drop_duplicates(C.ID_COL),
                      on=C.ID_COL, how="left", validate="many_to_one")
    assert len(merged) == len(df)
    merged.index = df.index
    return merged


def require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in (MEDIAN_COL, MIDPOINT_COL, DISAGREEMENT_COL)
               if c not in df.columns]
    if missing:
        raise KeyError(f"agreement metadata missing columns {missing}; "
                       f"columns present: {sorted(df.columns)}")
    reader_column(df)


def strata(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over the rows of a split CSV, in row order."""
    df = with_metadata(df)
    require_columns(df)
    readers = df[reader_column(df)].fillna(0).astype(int).to_numpy()
    midpoint = _flag(df[MIDPOINT_COL])
    disagree = _flag(df[DISAGREEMENT_COL])
    confident = (~midpoint) & (readers >= MIN_READERS) & (~disagree)
    return {"confident": confident, "remainder": ~confident,
            "midpoint": midpoint, "not_midpoint": ~midpoint,
            "disagreement": disagree, "no_disagreement": ~disagree,
            "single_reader": readers == 1, "three_plus_readers": readers >= 3}


def confident_mask(df: pd.DataFrame) -> np.ndarray:
    return strata(df)["confident"]


def medians(df: pd.DataFrame) -> np.ndarray:
    df = with_metadata(df)
    require_columns(df)
    return df[MEDIAN_COL].astype(float).to_numpy()
