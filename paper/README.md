# Paper draft

`main.tex` — arXiv-ready draft. Target: **cs.CV** (or **eess.IV**) primary,
cross-listed **cs.LG**.

## Status

All three arms are filled in. Every number in Table 1 comes from the committed
runs in `results/` and can be regenerated with:

```bash
.venv/bin/python scripts/compare_reps.py     # -> results/COMPARISON.md
```

Table 2 (agreement-stratified binary results) and the median-rating breakdown
currently come from `results/2d/ANALYSIS.md`. They become reproducible once the
runbook below has been run: `scripts/stratify_by_agreement.py` recomputes them
from the committed per-seed test probabilities into `results/AGREEMENT.md`.

## The argument

The paper does **not** claim one representation beats another — the differences
are inside the confidence intervals, and saying otherwise would be
unsupportable on 396 test nodules. The claim is the *null* result plus its
explanation:

1. Spatial context (1 → 5 → 104 slices) does not improve performance.
2. MoCo v2 pretraining does not improve 2D or 2.5D, despite the pretext task
   converging well. It gives +0.043 macro-F1 in 3D, but 3D validation ranked
   configurations inconsistently with test, so this is reported as an
   observation, not a result.
3. Performance is bounded by **annotator disagreement**, quantified on the
   binary cohort: 0.945 accuracy / 0.987 AUC on unanimously labelled nodules
   versus 0.805 / 0.902 on the rest.

Point 3 is what makes this publishable rather than a negative-results note. The
representation sweep is *evidence for* point 3, not the headline.

## Before submitting

- [x] Fill in the 3D numbers
- [ ] Confirm which binary model Table 2 uses (strong-augmentation or the
      Model C recipe in Table 1), state it in the caption, and commit the
      analysis script plus the test probabilities it reads
- [ ] Re-run 3D MoCo, Model B, Model C and binary with the corrected axial
      rotation in `src/common/data3d.py`, then update Table 1 and remove the
      sagittal-rotation caveat from the Augmentation paragraph and Limitations
- [x] Add a related-work section (Section 2)
- [ ] After the runbook: update Table 1 and the abstract to five seeds, replace
      Table 2 with `results/AGREEMENT.md`, and add the confident-cohort result
      (`paper/figures/fig_confident_cohort.pdf`) as a new subsection
- [ ] Confirm co-authorship with Fatima and Zaineb — they must agree to be
      listed, not merely be listed
- [ ] Check author name spellings and affiliation
- [ ] Re-read the LIDC-IDRI / TCIA data use agreement and confirm the citations
      are the ones they ask for
- [ ] Decide whether to make the GitHub repo public at submission time
- [ ] Confirm the 2.5D runs were on the Apple M2 (cost table)
- [ ] Confirm arXiv endorsement covers the chosen primary category

## Runbook for the final version

These runs need the dataset and a GPU, so they cannot run in CI.

1. **Mac, 2D and 2.5D** (from the repo root, with `CT_DATA_ROOT` set):

   ```bash
   bash scripts/run_final_local.sh
   ```

   This exports test predictions from the existing checkpoints, adds seeds 3
   and 4 for every model, trains the confident-label cohort (seeds 0–4), then
   regenerates `results/AGREEMENT.md`, `results/COMPARISON.md` and the
   figures. Finished runs are skipped, so it is safe to restart.

2. **Kaggle, 3D** — open `notebooks/3d_kaggle_final.ipynb` and run it three
   times with `PART = 1`, `2` and `3` (each fits in the 12 h limit). Part 1
   re-pretrains MoCo with the corrected axial rotation and trains Model B;
   part 2 trains A, C and binary; part 3 trains the confident cohort. Results
   are pushed after every seed.

3. **Mac, after pulling the Kaggle results:**

   ```bash
   git pull
   .venv/bin/python scripts/stratify_by_agreement.py
   .venv/bin/python scripts/compare_reps.py
   .venv/bin/python scripts/make_paper_figures.py
   ```

   Then update the paper numbers (checklist above) and rebuild.

## Building

```bash
python scripts/make_paper_figures.py      # -> paper/figures/*.pdf
bash scripts/make_arxiv_bundle.sh         # -> paper/main.pdf and paper/arxiv_submission.tar.gz
```

Upload `arxiv_submission.tar.gz` to arXiv as the source; it includes
`main.bbl`, which arXiv needs because it does not run bibtex.

Figures are generated from the committed `results/` files. The
agreement-stratified panel (`fig_ceiling`) reads `results/agreement/` when it
exists and otherwise falls back to the values in `results/2d/ANALYSIS.md`;
`fig_confident_cohort` is produced only once confident-cohort results exist.
