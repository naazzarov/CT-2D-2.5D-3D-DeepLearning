#!/bin/bash
# Final-version runs for the 2D and 2.5D arms (Apple M2, about 2 h):
#   0. recover test predictions for existing seeds 0-2 from best.pt
#   1. seeds 3 and 4 for Models A, B, C and binary (existing seeds 0-2 are kept)
#   2. confident-label cohort, Model C and binary recipes, seeds 0-4
#   3. agreement-stratified analysis, comparison tables and paper figures
# Same frozen protocol as run_2p5d_all.sh; only seeds and --cohort change.
# Re-running is safe: finished runs (result.json + test predictions) are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
COMMON="--epochs 80 --patience 20 --lr 1e-3 --wd 5e-2 --dropout 0.5 \
        --label-smoothing 0.10 --eval-test"

run() {  # run <rep> <tag> <seed> <args...>
  local rep=$1 tag=$2 seed=$3; shift 3
  local d="results/$rep/$tag/seed$seed"
  if [[ -f $d/result.json && -f $d/test_probabilities.npy ]]; then
    echo "skip $rep $tag seed $seed (done)"; return
  fi
  echo "########## $rep $tag seed $seed ##########"
  $PY src/2d/train_supervised.py --rep "$rep" --tag "$tag" --seed "$seed" \
      "$@" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
}

# Recover predictions for existing seeds from their checkpoints (no retraining).
$PY scripts/export_test_predictions.py --reps 2d 2p5d

for rep in 2d 2p5d; do
  moco="results/$rep/moco/moco_encoder.pt"
  [[ -f $moco ]] || { echo "missing $moco -- run MoCo pretraining first"; exit 1; }
  for s in 3 4; do
    run $rep model_a_noaug $s --task 3class --init random --aug none
    run $rep model_b_noaug $s --task 3class --init moco --moco-ckpt "$moco" --aug none
    run $rep model_c_mild  $s --task 3class --init random --aug mild
    run $rep binary_mild   $s --task binary --init random --aug mild
  done
  for s in 0 1 2 3 4; do
    run $rep model_c_mild_confident $s --task 3class --init random --aug mild \
        --cohort confident
    run $rep binary_mild_confident  $s --task binary --init random --aug mild \
        --cohort confident
  done
done

$PY scripts/stratify_by_agreement.py --reps 2d 2p5d
$PY scripts/compare_reps.py > /dev/null && echo "wrote results/COMPARISON.md"
$PY scripts/make_paper_figures.py
echo "########## LOCAL FINAL RUNS DONE ##########"
