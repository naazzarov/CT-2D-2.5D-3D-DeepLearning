# Preprocessing Contract (shared by 2D / 2.5D / 3D)

Our research question is *"does spatial context help?"*. That only holds if
context is the **only** thing that differs between the three models. Everything
on this page must therefore be identical for all three of us. Implemented once
in `src/common/` — please import it rather than re-implementing.

## 1. Source arrays

`samples/<consensus_nodule_id>.npy` — `(104, 72, 80)` float32, 1 mm isotropic,
**raw Hounsfield units**. Masks have the same shape, `uint8 {0, 1}`.

## 2. The nodule is already centred

Mask centre-of-mass over 25 random supervised samples: z 51.1–52.1,
y 35.2–35.7, x 39.3–39.9 — the geometric centre of the crop. So:

| approach | input | index |
|---|---|---|
| 2D | central slice | `vol[52]` → `(72, 80)` |
| 2.5D | 5 neighbouring slices | `vol[50:55]` → `(5, 72, 80)` |
| 3D | full crop | `vol` → `(104, 72, 80)` |

`CENTRAL_SLICE = 52` is defined in `src/common/config.py`. No recomputation from
masks is needed, and mask-based centring should be avoided: some masks contain
only 1–3 positive voxels.

## 3. HU windowing — the important one

Arrays are stored as **raw HU**, so the window is our choice, and an
unstated choice is the easiest way for our three models to become silently
incomparable.

Measured over 120 random samples, the raw range is far wider than the anatomy:
minima down to **−3024** (out-of-FOV padding; 16/120 samples fell below −1100)
and maxima up to **+3080**. Normalising without clipping lets those outliers
dominate the scale.

**Status: ratified by the team (2026-10-01).** All three members confirmed they
use this same window, so the 2D / 2.5D / 3D comparison isolates spatial context
rather than preprocessing.

**Agreed rule** (`src/common/transforms.window_hu`):

```python
x = np.clip(x, -1000, 400)
x = (x + 1000) / 1400        # -> [0, 1]
```

Then standardise with **train-split** mean/std (currently mean 0.3651,
std 0.3145 for the 2D central-slice cache; recompute per representation with
`src.common.data.train_statistics`, never using val or test).

## 4. Labels

`class_index_v2`: 0 = benign, 1 = indeterminate, 2 = malignant. Never recompute
from `malignancy_median` — use the column, so the frozen policy applies.

## 5. Splits

Read `metadata/supervised_{train,validation,test}.csv` as-is.
`src.common.data.assert_no_leakage` re-verifies on every run and **raises** on
any patient or nodule overlap, including SSL-pool contamination of val/test.
Independently confirmed clean: 0 patient overlap, 0 nodule overlap,
0 val/test patients in the SSL pool.

## 6. Class imbalance

Train counts are 622 / 876 / 378, so *indeterminate* is 47% of the data. We use
inverse-frequency class weights `N / (K · n_k)` = `[1.005, 0.714, 1.654]` in the
loss, and select on macro-F1. Without this, a model scores deceptively well by
leaning on the majority class.

## 7. Augmentation

Training split only. Validation and test are never augmented.

## 8. Reference baseline

Always predicting *indeterminate* on the test split gives macro-F1 ≈ 0.21,
accuracy ≈ 0.47. Any reported model must clear this, and it is worth stating in
the paper — accuracy near 0.50 on this 3-class task is only marginally above
the majority-class rate, while macro-F1 near 0.50 is a genuine result.
