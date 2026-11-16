# Pixel is All You Need

Code for the paper *Pixel is All You Need*.

The repository has two parts that follow the paper:

| Part | Paper | Directory |
|---|---|---|
| Benchmark of current image editing, video and unified multimodal generators | Sec. 3 | `generator_benchmark/` |
| Pixel-space generative models trained from scratch with no text pathway | Secs. 4 to 7 | `pixel_models/` |

## Setup

```bash
pip install -r requirements.txt          # pixel models: torch, numpy, pillow
pip install -r requirements-benchmark.txt  # Sec. 3 only: diffusers, transformers, ...
python -m unittest discover tests        # CPU checks of the tasks, the reader and the LoRA
```

The fonts in `assets/` (Liberation, SIL Open Font License) are used to draw every canvas. Pillow's basic text layout
is pinned in the code, so canvases are pixel-identical across machines.

## Pixel-space models (Secs. 4 to 7)

A flow-matching transformer that works directly on 8x8 pixel patches (`pixel_models/model.py`). It reads the
question from pixels and writes the steps and the answer as pixels. It never sees character codes, and answers are
read back from the canvas by nearest-glyph matching, with no judge model (`pixel_models/canvas.py`).

| Module | What it holds |
|---|---|
| `model.py` | The flow ViT (Sec. 4.2, Tab. 6 sizes XS 7.6M, S 33M, M 58M, B 130M, L 458M) |
| `canvas.py` | The 192x192 canvas: 24x12 cells of 8x16 px, a rule, a two-row answer box, and the reader |
| `text_tasks.py` | Expressions, Multiplication, Word Problems, Factual Questions; their single-GPU trainer |
| `draw_tasks.py` | Curve Tracing, Trace Then Sum, Count Then Paint, Compute Then Paint; their trainer |
| `maze_tasks.py` | Mazes and Sum Mazes |
| `train_text.py` | Multi-GPU trainer of the text tasks (text scaling ladder, 160k-step runs), per-step metrics |
| `train_joint.py` | Several tasks in one model (cross-modal scaling ladder, Sum Mazes), per-step metrics |
| `rl_text.py` | GRPO on frame-by-frame rollouts (Sec. 7, text tasks) |
| `rl_draw.py` | RFT and the SFT controls (Sec. 7, cross-modal tasks) |
| `evaluate.py` | Accuracy on 1,024 fixed problems per test set (every main-text number) |
| `scaling_fit.py` | Power-law fits with a held-out 458M model and seed-bootstrap intervals (Sec. 5.2, Tab. 7) |

Training problems are generated on the fly; test problems come from fixed seeds, so every run of a task sees the
same test items. Two regimes share the model, data and budget: `--arm direct` maps the problem canvas to the final
canvas in one generation, and `--arm video` generates frame by frame, feeding each generated frame back as the next
input until the model writes an answer.

### Task specs

| Task | Spec | Trainer |
|---|---|---|
| Expressions | `expr:4` | `train_text` / `text_tasks` |
| Multiplication | `mul:4x4` | `train_text` |
| Word Problems | `wp:3` (256x256 canvas) | `text_tasks` / `train_text` |
| Factual Questions | `hop:3` | `train_text` |
| Mazes | `maze:5` | `train_joint` |
| Sum Mazes | `maze:5:sum` | `train_joint` |
| Curve Tracing | `trace:6` | `draw_tasks` / `train_joint` |
| Trace Then Sum | `tracesum:7` | `draw_tasks` |
| Count Then Paint | `countpaint:tallysel` (direct: `countpaint`) | `draw_tasks` |
| Compute Then Paint | `countpaint:arithcheck:sel` | `draw_tasks` |

Word Problems use a 256x256 canvas (`--px 256`, batch 128), because the longer problems need more rows; every other
task uses 192x192.

### Reproducing the main text

Each script runs one section in sequence. They are written for clarity, not for scheduling, so split the loops across
GPUs as needed. Every run directory holds `config.json`, `ckpt.pt` (model, EMA, optimizer, step; a restarted run
resumes) and `evals.jsonl`. `evaluate.py` writes `final.json`.

| Paper | Script | Notes |
|---|---|---|
| Sec. 5, Fig. 5, Tab. 7 | `scripts/sec5_scaling.sh` | 5 sizes x 5 seeds x 2 ladders, 132k steps; 458M on 2 to 4 GPUs |
| Sec. 6, Fig. 6a | `scripts/sec6_cot.sh S` and `B` | 40k steps per regime, direct answering also 160k |
| Appendix I | `scripts/sec6_cot.sh XS` | 7.6M |
| Sec. 7, Fig. 6b | `scripts/sec7_rl.sh S` and `B` | needs the Sec. 6 runs of the same size (start models) |

Single commands, for example:

```bash
# frame by frame on Expressions, 33M, 40k steps
python -m pixel_models.train_text train --task expr:4 --arm video --size S --out runs/expr_video
python -m pixel_models.evaluate --run runs/expr_video --n 1024 --out runs/expr_video/final.json

# the 458M point of the text ladder on 4 GPUs, then its per-step error
torchrun --standalone --nproc_per_node 4 -m pixel_models.train_text train --task multi:expr:4+mul:4x4+hop:3 \
    --arm video --size L --steps 132000 --decay-from 128000 --seed 0 --out runs/text_L_s0
python -m pixel_models.train_text valloss --run runs/text_L_s0 --val-per-set 4096 --batch 32

# GRPO from a start model trained with 1-3 steps per frame
python -m pixel_models.rl_text --init runs/expr_start/ckpt.pt --task expr:4 --prompt-set k4 --size S --out runs/expr_grpo
```

### Where the numbers are in the output

| Paper number | Key in `final.json` (`sets.SET.KEY`) |
|---|---|
| Expressions, Word Problems (Sec. 6) | `k4.exact`, `new_k3.exact` |
| Expressions, Word Problems (Sec. 7) | `k4.free`, `new_k3.free` (the model chooses how many frames it draws) |
| Sum Mazes (Sec. 6) | `n3`, `n4`, `n5`: `exact` (the demonstrated path and its total); `valid` accepts any shortest path |
| Sum Mazes (Sec. 7) | `n3`, `n4`, `n5`: `valid` (a shortest path with that path's own total) |
| Trace Then Sum (Sec. 6) | `k3`, `k5`, `k7`: `exact` |
| Count Then Paint, Compute Then Paint (Sec. 6) | `k3.solved`, `arith.solved` (letter, counts, painting IoU >= 0.8) |
| Count Then Paint, Compute Then Paint, Trace Then Sum (Sec. 7) | `minimal`: solved in the fewest frames (Trace Then Sum: the mean over `k3`, `k5`, `k7`) |
| Per-step error (Sec. 5) | `valloss.json` / `valloss_4k.json`, fitted by `scaling_fit.py` |

## Benchmark of current generators (Sec. 3)

Each item is a 1024x1024 canvas with a title, a printed question and an empty answer box. Six tasks span decreasing
answer visibility: Copy, Instruction, Context, Visual, Arithmetic and Factual (`generator_benchmark/tasks.py`,
`cards.py`, `render.py`). A multimodal judge reads the answer box (`generator_benchmark/judge/`).

| Step | Module |
|---|---|
| Open image editing models, Default / Rewrite / Delete (Tab. 1) | `generate.py` (`--route native | rewrite`), `models.py` |
| Prompt rewriters and answer deletion | `rewrite.py`, `delete_answer.py`, `rewrite_prompts/` |
| Closed image editing models | `api_generate.py`, `api_image.py` |
| Video generation models | `video_generate.py`, `video_eval.py` (judged on the final frame) |
| Unified multimodal models, Default / Render / Think | `umm_generate.py`, `bagel_generate.py` |
| LoRA post-training of the generation module (Sec. 3.4, Fig. 3) | `train.py`, `lora.py` |
| BAGEL fine-tuning (Sec. 3.4) | `bagel_train.py`, `bagel_cards.py`, `bagel_replay.py`, `replay_prompts.py`, `bagel_bench.py` |
| Judging and scores | `cases.py`, `judge/run_text.py`, `judge_summary.py` |

Model weights are not included. Set `UMM_MODEL_DIR` to the directory holding them and `UMM_SRC_DIR` to the upstream
source trees that are not pip packages (see `generator_benchmark/paths.py` for the expected sub-directory names).
API endpoints are read from environment variables and have no defaults:

| Variable | Used for |
|---|---|
| `JUDGE_BASE_URL`, `JUDGE_API_KEY` | the multimodal judge |
| `REWRITER_BASE_URL`, `REWRITER_API_KEY` | prompt rewriters and the LLM pass of the answer deletion |
| `IMAGE_API_BASE_URL`, `IMAGE_API_KEY` | closed image editing models |
| `VIDEO_API_BASE_URL`, `VIDEO_API_KEY` | closed video generation models |

## Citation

```bibtex
@inproceedings{pixel2027,
  title     = {Pixel is All You Need},
  booktitle = {CVPR},
  year      = {2027}
}
```

## License

The code is released under the MIT License (`LICENSE`). The fonts in `assets/` are under the SIL Open Font License.
