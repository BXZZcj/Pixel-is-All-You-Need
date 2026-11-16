#!/bin/bash
# Sec. 5 (Fig. 5, Tab. 7): the two scaling ladders, five sizes x five seeds, 132k steps (cool-down over the last 4k).
# Each run fits on one GPU up to 130M; the 458M (L) runs use 2-4 GPUs at the same global batch (NGPU_L).
# usage: bash scripts/sec5_scaling.sh [OUT]   (runs everything in sequence; split the loops across machines as needed)
set -euo pipefail
OUT=${1:-runs/sec5}
NGPU_L=${NGPU_L:-4}
launch() {   # launch NGPU MODULE ARGS...
  local n=$1; shift
  if [ "$n" -gt 1 ]; then torchrun --standalone --nproc_per_node "$n" -m "$@"; else python -m "$@"; fi
}
for size in XS S M B L; do
  n=1; [ $size = L ] && n=$NGPU_L
  for seed in 0 1 2 3 4; do
    # text ladder: Expressions, Multiplication and Factual Questions in one model
    R=$OUT/text_${size}_s$seed
    launch $n pixel_models.train_text train --task multi:expr:4+mul:4x4+hop:3 --arm video --size $size --seed $seed \
      --px 192 --batch 256 --steps 132000 --decay-from 128000 --kb-people 300 --out $R
    python -m pixel_models.train_text valloss --run $R --val-per-set 4096 --batch 32
    # cross-modal ladder: Mazes, Curve Tracing and Count Then Paint in one model (evaluated on Curve Tracing)
    R=$OUT/joint_${size}_s$seed
    launch $n pixel_models.train_joint train --tasks maze:5+trace:6+countpaint:tallysel:hard --per-task 88 --size $size --seed $seed \
      --steps 132000 --decay-from 128000 --chg-weight 30 --cf-paint 0 --world-arm video --out $R
    python -m pixel_models.train_joint valloss --run $R --val-per-set 4096 --tag _4k
  done
done
python -m pixel_models.scaling_fit --ladder text --runs $OUT/text_* --out $OUT/fit_text.json
python -m pixel_models.scaling_fit --ladder trace --runs $OUT/joint_* --out $OUT/fit_trace.json
