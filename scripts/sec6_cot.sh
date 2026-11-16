#!/bin/bash
# Sec. 6 (Fig. 6a): frame by frame (video) against direct answering, 40k steps each, and direct answering for 160k.
# Every accuracy is measured on 1,024 problems per test set with pixel_models.evaluate.
# usage: bash scripts/sec6_cot.sh SIZE [OUT]      SIZE: S (33M) or B (130M); XS (7.6M) for Appendix I
set -euo pipefail
SIZE=${1:?SIZE: XS, S or B}
OUT=${2:-runs/sec6}/$SIZE
SEED=${SEED:-0}
ev() { python -m pixel_models.evaluate --run "$@" --n 1024 --out "$1/final.json"; }

# Expressions (4 operations). Constant learning rate for the 40k runs; 160k runs cool down from 152k.
for arm in video direct; do
  python -m pixel_models.train_text train --task expr:4 --arm $arm --size $SIZE --seed $SEED --px 192 --batch 256 --steps 40000 --out $OUT/expr_$arm
  ev $OUT/expr_$arm
done
python -m pixel_models.train_text train --task expr:4 --arm direct --size $SIZE --seed $SEED --px 192 --batch 256 --steps 160000 --decay-from 152000 --out $OUT/expr_direct_160k
ev $OUT/expr_direct_160k

# Word Problems (3 steps, new names): 256 x 256 canvas, batch 128. Constant learning rate for the 40k runs.
for arm in video direct; do
  python -m pixel_models.text_tasks train --task wp:3 --arm $arm --size $SIZE --seed $SEED --px 256 --batch 128 --steps 40000 --out $OUT/wp_$arm
  ev $OUT/wp_$arm
done
python -m pixel_models.train_text train --task wp:3 --arm direct --size $SIZE --seed $SEED --px 256 --batch 128 --steps 160000 --decay-from 152000 --out $OUT/wp_direct_160k
ev $OUT/wp_direct_160k

# Sum Mazes (3x3 to 5x5): cool-down from 36k. Accuracy: the demonstrated path and its total ("exact"), or any
# shortest path with its own total ("valid").
for arm in video direct; do
  python -m pixel_models.train_joint train --tasks maze:5:sum --world-arm $arm --per-task 256 --size $SIZE --seed $SEED \
    --steps 40000 --decay-from 36000 --chg-weight 30 --out $OUT/summaze_$arm
  ev $OUT/summaze_$arm
done
python -m pixel_models.train_joint train --tasks maze:5:sum --world-arm direct --per-task 256 --size $SIZE --seed $SEED \
  --steps 160000 --decay-from 152000 --chg-weight 30 --out $OUT/summaze_direct_160k
ev $OUT/summaze_direct_160k

# Trace Then Sum (3, 5, 7 curves): each regime starts from a Curve Tracing model of the same regime (40k steps,
# constant learning rate), then trains 40k steps with a cool-down from 36k.
for arm in video direct; do
  python -m pixel_models.draw_tasks train --task trace:6 --arm $arm --size $SIZE --seed $SEED --steps 40000 --out $OUT/trace_$arm
  python -m pixel_models.draw_tasks train --task tracesum:7 --arm $arm --size $SIZE --seed $SEED --steps 40000 --decay-from 36000 \
    --chg-weight 30 --init $OUT/trace_$arm/ckpt.pt --out $OUT/tracesum_$arm
  ev $OUT/tracesum_$arm
done
python -m pixel_models.draw_tasks train --task tracesum:7 --arm direct --size $SIZE --seed $SEED --steps 160000 --decay-from 152000 \
  --chg-weight 30 --init $OUT/trace_direct/ckpt.pt --out $OUT/tracesum_direct_160k
ev $OUT/tracesum_direct_160k

# Count Then Paint: cool-down from 20k. The direct runs are evaluated on the frame-by-frame runs' problems.
python -m pixel_models.draw_tasks train --task countpaint:tallysel --arm video --size $SIZE --seed $SEED --steps 40000 --decay-from 20000 \
  --cf-paint 0.5 --chg-weight 30 --out $OUT/countpaint_video
ev $OUT/countpaint_video
python -m pixel_models.draw_tasks train --task countpaint --arm direct --size $SIZE --seed $SEED --steps 40000 --decay-from 20000 \
  --chg-weight 30 --out $OUT/countpaint_direct
ev $OUT/countpaint_direct --items countpaint:tallysel
python -m pixel_models.draw_tasks train --task countpaint --arm direct --size $SIZE --seed $SEED --steps 160000 --decay-from 152000 \
  --chg-weight 30 --out $OUT/countpaint_direct_160k
ev $OUT/countpaint_direct_160k --items countpaint:tallysel

# Compute Then Paint: starts from the Count Then Paint model of the same regime, cool-down from 36k.
python -m pixel_models.draw_tasks train --task countpaint:arithcheck:sel --arm video --size $SIZE --seed $SEED --steps 40000 --decay-from 36000 \
  --cf-paint 0.5 --chg-weight 30 --init $OUT/countpaint_video/ckpt.pt --out $OUT/computepaint_video
ev $OUT/computepaint_video
python -m pixel_models.draw_tasks train --task countpaint:arithcheck:sel --arm direct --size $SIZE --seed $SEED --steps 40000 --decay-from 36000 \
  --chg-weight 30 --init $OUT/countpaint_direct/ckpt.pt --out $OUT/computepaint_direct
ev $OUT/computepaint_direct
python -m pixel_models.draw_tasks train --task countpaint:arithcheck:sel --arm direct --size $SIZE --seed $SEED --steps 160000 --decay-from 152000 \
  --chg-weight 30 --init $OUT/countpaint_direct/ckpt.pt --out $OUT/computepaint_direct_160k
ev $OUT/computepaint_direct_160k
