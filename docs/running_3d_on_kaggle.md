# Running the 3D arm on a free Kaggle GPU

3D convolutions take **13.5–15 min/epoch** on an Apple M2 — roughly two weeks
for the full 3D set. Kaggle gives a free P100/T4 and **20 GB** of dataset
storage (Google Drive's free tier is only 15 GB, which the data does not fit
alongside existing files).

## 1. Build the upload folder (local)

```bash
.venv/bin/python scripts/export_volumes_subset.py ~/kaggle_upload/volumes_fp16.npy
```

This packs only the **5911 volumes the experiments actually touch** — the full
manifest has 7385, but 1474 appear in no split and no SSL pool. Result: ~6.6 GB
instead of 8.2 GB. Values are float16; the worst round-trip error is 1 HU,
which after the [−1000, 400] window is 0.018% of the range, about 100× smaller
than the Gaussian noise the augmentation already adds. The script re-measures
this and aborts if it exceeds tolerance.

Then add the metadata next to it:

```bash
mkdir -p ~/kaggle_upload/final_team_dataset_v2_3class
cp -r ~/Documents/final_team_dataset_v2_3class/metadata \
      ~/kaggle_upload/final_team_dataset_v2_3class/
cp -r ~/Documents/binary_sensitivity_team_package ~/kaggle_upload/
```

Final layout (~6.6 GB):

```
~/kaggle_upload/
├── volumes_fp16.npy
├── volumes_fp16_index.json
├── final_team_dataset_v2_3class/metadata/…
└── binary_sensitivity_team_package/…
```

## 2. Upload as a private Kaggle dataset

Get an API token first: kaggle.com → your avatar → Settings → API →
**Create New Token**. It downloads `kaggle.json`; install it:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Then:

```bash
cd ~/kaggle_upload
.venv/bin/kaggle datasets init -p .
# edit dataset-metadata.json: set "title" to "CT nodule volumes"
#                             and "id" to "<your-kaggle-username>/ct-nodule-volumes"
.venv/bin/kaggle datasets create -p . --dir-mode zip
```

The CLI resumes better than the browser uploader, which matters at 6.6 GB.

## 3. Run the notebook

Upload `notebooks/3d_kaggle.ipynb` to kaggle.com/code, then in **Settings**:

- **Accelerator**: GPU T4 x2 (or P100)
- **Internet**: On — required to clone the repo
- **Add Input**: attach **both** datasets:
  - `ct-nodule-volumes` — the packed fp16 volumes (7 GB) + binary package
  - `ct-nodule-metadata` — the 3-class split CSVs (3 MB)

  Two datasets because `qc_exclusions.csv` and `ssl_pretrain_train.csv` were
  truncated to 0 bytes while staging the 7 GB upload (iCloud placeholder files
  copied before they had materialised). Re-sending 7 GB to fix 9 MB was not
  worth it, so the CSVs live in their own dataset and the notebook reads them
  from there, asserting each one is non-empty before training.

Store your GitHub token as a Kaggle Secret named `GITHUB_TOKEN`
(*Add-ons → Secrets*) so it never appears in the notebook.

Run the cells in order. Cells 4 and 5 verify the data and benchmark a training
step — check both before starting the long runs.

## Limits worth knowing

- **12 h per session**, ~30 GPU-hours/week on the free tier.
- MoCo pretraining is the long pole, which is why it starts at 100 epochs
  rather than the 200 used for 2D. Its checkpoint is copied to
  `/kaggle/working` immediately so *Save Version* preserves it.
- If a session dies mid-way, re-running individual seeds is cheap.

## Protocol

Identical to the 2D and 2.5D arms. Two deliberate 3D-specific choices,
documented in `src/common/data3d.py`:

- **No flip or rotation along z** — CT has a consistent anatomical orientation
  in z, and our z-context is interpolated from 2.5 mm acquisitions.
- **3D ResNet-10, not ResNet-18** — a 3D ResNet-18 would carry ~33M parameters
  against 1876 training volumes.
