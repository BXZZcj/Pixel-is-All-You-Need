#!/bin/bash
# Sec. 7 (Fig. 6b): RL against supervised fine-tuning from the same start model.
#   text tasks          start: frame by frame with 1-3 steps per frame (--per-frame 3); GRPO for 1,500 iterations;
#                       SFT on optimal (1-3 steps per frame) or non-optimal (--detour 0.5) demonstrations
#   cross-modal tasks   start: trained on non-optimal demonstrations (:detour); RFT for 600 iterations; SFT on the
#                       non-optimal (:detour) or optimal (:clean) demonstrations for the same 600 iterations
# usage: bash scripts/sec7_rl.sh SIZE [OUT] [SEC6_OUT]     SIZE: S (33M) or B (130M)
# The Trace Then Sum start continues the Curve Tracing model of scripts/sec6_cot.sh; the 33M Compute Then Paint start
# continues its Count Then Paint model (SEC6_OUT).
set -euo pipefail
SIZE=${1:?SIZE: S or B}
OUT=${2:-runs/sec7}/$SIZE
SEC6=${3:-runs/sec6}/$SIZE
SEED=${SEED:-0}
ev() { python -m pixel_models.evaluate --run "$@" --n 1024 --out "$1/final.json"; }

# ---- Expressions: GRPO on fresh k4 problems, at most 10 frames
S=$OUT/expr_start
python -m pixel_models.text_tasks train --task expr:4 --arm video --per-frame 3 --size $SIZE --seed $SEED --steps 40000 --out $S
ev $S
python -m pixel_models.rl_text --init $S/ckpt.pt --task expr:4 --prompt-set k4 --max-frames 10 --size $SIZE --iters 1500 --out $OUT/expr_grpo
ev $OUT/expr_grpo
python -m pixel_models.text_tasks train --task expr:4 --arm video --per-frame 3 --size $SIZE --seed $SEED --steps 65000 \
  --init-from $S/ckpt.pt --out $OUT/expr_sft_optimal
python -m pixel_models.text_tasks train --task expr:4 --arm video --per-frame 3 --detour 0.5 --size $SIZE --seed $SEED --steps 65000 \
  --init-from $S/ckpt.pt --out $OUT/expr_sft_nonoptimal
ev $OUT/expr_sft_optimal; ev $OUT/expr_sft_nonoptimal

# ---- Word Problems: the 40k frame-by-frame model of Sec. 6 continued 10k steps with 1-3 steps per frame; GRPO on
# training problems with the most steps, at most 8 frames
S=$OUT/wp_start
python -m pixel_models.text_tasks train --task wp:3 --arm video --per-frame 3 --px 256 --batch 128 --size $SIZE --seed $SEED --steps 50000 \
  --init-from $SEC6/wp_video/ckpt.pt --out $S
ev $S
python -m pixel_models.rl_text --init $S/ckpt.pt --task wp:3 --hard-only --max-frames 8 --px 256 --size $SIZE --iters 1500 --out $OUT/wp_grpo
ev $OUT/wp_grpo
python -m pixel_models.text_tasks train --task wp:3 --arm video --per-frame 3 --px 256 --batch 128 --size $SIZE --seed $SEED --steps 75000 \
  --init-from $S/ckpt.pt --out $OUT/wp_sft_optimal
python -m pixel_models.text_tasks train --task wp:3 --arm video --per-frame 3 --detour 0.5 --px 256 --batch 128 --size $SIZE --seed $SEED --steps 75000 \
  --init-from $S/ckpt.pt --out $OUT/wp_sft_nonoptimal
ev $OUT/wp_sft_optimal; ev $OUT/wp_sft_nonoptimal

# ---- cross-modal start models (non-optimal demonstrations)
python -m pixel_models.train_joint train --tasks maze:5:sum:detour --per-task 256 --size $SIZE --seed $SEED --steps 40000 --decay-from 36000 \
  --chg-weight 30 --out $OUT/summaze_start
python -m pixel_models.draw_tasks train --task tracesum:7:detour --arm video --size $SIZE --seed $SEED --steps 40000 --decay-from 36000 \
  --chg-weight 30 --init $SEC6/trace_video/ckpt.pt --out $OUT/tracesum_start
python -m pixel_models.draw_tasks train --task countpaint:tallysel:detour --arm video --size $SIZE --seed $SEED --steps 40000 --decay-from 20000 \
  --chg-weight 30 --out $OUT/countpaint_start
if [ $SIZE = S ]; then   # 33M: from the Count Then Paint model; 130M: from scratch
  python -m pixel_models.draw_tasks train --task countpaint:arithcheck:sel:detour --arm video --size $SIZE --seed $SEED --steps 40000 \
    --decay-from 36000 --chg-weight 30 --init $SEC6/countpaint_video/ckpt.pt --out $OUT/computepaint_start
else
  python -m pixel_models.draw_tasks train --task countpaint:arithcheck:sel:detour --arm video --size $SIZE --seed $SEED --steps 40000 \
    --decay-from 20000 --chg-weight 30 --out $OUT/computepaint_start
fi

# ---- RFT, non-optimal SFT and optimal SFT, 600 iterations each
for t in "summaze maze:5:sum:detour maze:5:sum:canon:as-detour" \
         "tracesum tracesum:7:detour tracesum:7:clean" \
         "countpaint countpaint:tallysel:detour countpaint:tallysel:clean" \
         "computepaint countpaint:arithcheck:sel:detour countpaint:arithcheck:sel:clean"; do
  set -- $t
  name=$1 detour=$2 clean=$3
  S=$OUT/${name}_start
  ev $S
  python -m pixel_models.rl_draw --init $S/ckpt.pt --task $detour --algo rft --size $SIZE --iters 600 --out $OUT/${name}_rft
  python -m pixel_models.rl_draw --init $S/ckpt.pt --task $detour --algo sft --size $SIZE --iters 600 --out $OUT/${name}_sft_nonoptimal
  python -m pixel_models.rl_draw --init $S/ckpt.pt --task $clean --algo sft --size $SIZE --iters 600 --out $OUT/${name}_sft_optimal
  for arm in rft sft_nonoptimal sft_optimal; do ev $OUT/${name}_$arm; done
done
