#!/bin/bash
# Full 2.5D track, identical protocol to 2D -- only --rep changes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON="--rep 2p5d --epochs 80 --patience 20 --lr 1e-3 --wd 5e-2 --dropout 0.5 \
        --label-smoothing 0.10 --eval-test"

for s in 0 1 2; do
  echo "########## 2p5d A_noaug seed $s ##########"
  $PY src/2d/train_supervised.py --task 3class --init random --aug none \
      --tag model_a_noaug --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
for s in 0 1 2; do
  echo "########## 2p5d C_mild seed $s ##########"
  $PY src/2d/train_supervised.py --task 3class --init random --aug mild \
      --tag model_c_mild --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
for s in 0 1 2; do
  echo "########## 2p5d BINARY_mild seed $s ##########"
  $PY src/2d/train_supervised.py --task binary --init random --aug mild \
      --tag binary_mild --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
echo "########## 2p5d MoCo pretraining ##########"
$PY src/2d/moco_pretrain.py --rep 2p5d --epochs 200 2>&1 | tail -20
for s in 0 1 2; do
  echo "########## 2p5d B_noaug seed $s ##########"
  $PY src/2d/train_supervised.py --task 3class --init moco \
      --moco-ckpt results/2p5d/moco/moco_encoder.pt --aug none \
      --tag model_b_noaug --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
echo "########## 2P5D ALL DONE ##########"
