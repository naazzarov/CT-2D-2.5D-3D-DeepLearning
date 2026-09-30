# Paper draft

`main.tex` — arXiv-ready draft. Target: **cs.LG** primary, cross-listed
**eess.IV** and **cs.CV**.

## Status

Every number in the draft comes from the committed runs in `results/`. Cells
marked `REPLACE_3D` are placeholders for the 3D arm, which is running on Kaggle
(see `docs/running_3d_on_kaggle.md`). When those land, regenerate the tables:

```bash
.venv/bin/python scripts/compare_reps.py     # -> results/COMPARISON.md
```

and copy the 3D column across.

## The argument

The paper does **not** claim one representation beats another — the differences
are inside the confidence intervals, and saying otherwise would be
unsupportable on 396 test nodules. The claim is the *null* result plus its
explanation:

1. Spatial context (1 → 5 → 104 slices) does not improve performance.
2. MoCo v2 pretraining does not improve performance, in either representation
   tested so far, despite the pretext task converging well.
3. Performance is bounded by **annotator disagreement**, quantified on the
   binary cohort: 0.945 accuracy / 0.987 AUC on unanimously labelled nodules
   versus 0.805 / 0.902 on the rest.

Point 3 is what makes this publishable rather than a negative-results note. The
representation sweep is *evidence for* point 3, not the headline.

## Before submitting

- [ ] Fill in the 3D numbers
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
