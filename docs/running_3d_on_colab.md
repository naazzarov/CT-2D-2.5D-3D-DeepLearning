# Running the 3D arm on a free Colab GPU

3D convolutions are far too slow on an Apple M2: measured **13.5–15 min/epoch**
for the 1876-volume training split (14.36M-parameter 3D ResNet-10, batch 4–16).
The full 3D set — Model A, Model C, binary, MoCo pretraining and Model B, three
seeds each — would take roughly two weeks of continuous running. A free Colab
T4 does the same work in hours.

## One-time preparation

**1. Pack the volumes (local, ~20 min)**

```bash
.venv/bin/python scripts/export_volumes_int16.py ~/volumes_int16.npy
```

Stored HU values are integral in [−3024, 3080], so int16 is **lossless** and
halves the size: 17 GB → ~8.5 GB. The script verifies losslessness on 25 random
volumes before writing anything, and emits `volumes_int16_index.json` mapping
`consensus_nodule_id` → row.

**2. Upload to Google Drive** into a folder such as `MyDrive/ct_project/`:

- `volumes_int16.npy` and `volumes_int16_index.json`
- the `metadata/` folder from `final_team_dataset_v2_3class`
- the `binary_sensitivity_team_package/` folder

The CT arrays themselves are never committed to git.

**3. Open `notebooks/3d_colab.ipynb` in Colab**, set *Runtime → Change runtime
type → T4 GPU*, and run the cells in order.

## What the notebook does

1. Checks the GPU
2. Clones this repo (private, so it asks for a GitHub token)
3. Mounts Drive and sets `CT_DATA_ROOT`, `CT_BINARY_ROOT`, `CT_VOLUME_PACK`
4. **Verifies the data** — asserts zero patient/nodule leakage, the agreed
   binary counts (1000/219/209), and that volumes load at the expected shape
5. Benchmarks one training step so you can confirm the GPU is being used
6. Runs Model A, Model C, binary, MoCo pretraining, Model B
7. Backs results up to Drive and commits them

## Protocol

Identical to the 2D and 2.5D arms — same HU window [−1000, 400], same splits,
same label columns, same AdamW/cosine/class-weights/label-smoothing settings,
same early stopping on validation macro-F1, same metrics. Only the encoder and
the input representation differ.

Two deliberate 3D-specific choices, both documented in `src/common/data3d.py`:

- **No flip or rotation along z.** CT has a consistent anatomical orientation
  in z, and our z-context is interpolated from 2.5 mm acquisitions.
- **3D ResNet-10, not ResNet-18.** A 3D ResNet-18 would carry ~33M parameters
  against 1876 training volumes.

## Practical notes

- **Colab disconnects.** MoCo pretraining is the long pole — the notebook copies
  the encoder to Drive immediately after it finishes. If a supervised run dies,
  re-running that seed is cheap.
- **`CT_VOLUME_PACK` matters for speed.** Reading one sequential memmap is far
  faster than 7385 individual file reads from Drive.
- Start MoCo at 100 epochs (the 2D arm used 200); raise it if the session holds.
