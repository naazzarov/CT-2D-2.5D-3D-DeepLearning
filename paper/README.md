# Paper draft

`main.tex` — arXiv-ready draft. Target: **cs.LG** primary, cross-listed
**eess.IV** and **cs.CV**.

## Status

All three arms are filled in. Every number in Table 1 comes from the committed
runs in `results/` and can be regenerated with:

```bash
.venv/bin/python scripts/compare_reps.py     # -> results/COMPARISON.md
```

Table 2 (agreement-stratified binary results) and the median-rating breakdown
come from the analysis in `results/2d/ANALYSIS.md`; the script and the
seed-averaged test probabilities behind them are not yet committed.

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
- [ ] Add a related-work paragraph with citations for the 2D / 2.5D / 3D
      LIDC-IDRI literature the introduction refers to
- [ ] Confirm co-authorship with Fatima and Zaineb — they must agree to be
      listed, not merely be listed
- [ ] Check author name spellings and affiliation
- [ ] Re-read the LIDC-IDRI / TCIA data use agreement and confirm the citations
      are the ones they ask for
- [ ] Decide whether to make the GitHub repo public at submission time
- [ ] Confirm arXiv endorsement covers cs.LG

## Building

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
