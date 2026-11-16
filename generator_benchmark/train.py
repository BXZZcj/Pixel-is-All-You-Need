"""LoRA post-training of an image editing model on one GPU (Sec. 3, one LoRA per task).

Recipe: AdamW lr 1e-4, weight decay 1e-4, rank 16, alpha 16, microbatch 1 with 4-step gradient accumulation,
flow-matching MSE on the target latent (--roi-weight > 1 gives the answer box a larger weight), clip 1.0, 2500 updates.

usage: python -m generator_benchmark.train --model flux2_klein_base --tasks qa --updates 2500 --roi-weight W --out RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from . import lora, render, tasks
from .models import ADAPTERS


def code_commit() -> str:
    """The git commit of this checkout, recorded with the run ("" outside a git checkout)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def roi_weight_map(item: dict, shape, w: float, device) -> torch.Tensor:
    """Weights on the latent grid: w inside the answer area, 1 elsewhere (broadcast over channels/frames)."""
    from . import cards
    h, w_lat = shape[-2], shape[-1]
    if item["mode"] == "select":
        x0, y0 = cards.OPTION_ROIS[0][0], cards.OPTION_ROIS[0][1]
        x1, y1 = cards.OPTION_ROIS[2][2], cards.OPTION_ROIS[0][3]
    else:
        x0, y0, x1, y1 = cards.ANSWER_ROI
    sy, sx = h / cards.SIZE, w_lat / cards.SIZE
    m = torch.ones(h, w_lat, device=device)
    m[int(y0 * sy):int(-(-y1 * sy // 1)), int(x0 * sx):int(-(-x1 * sx // 1))] = w
    return m.view(*([1] * (len(shape) - 2)), h, w_lat).expand(*shape)


def training_items(task_names: list[str], n_per_task: int, seed: int) -> list[dict]:
    """Deterministic: the first n of each task's frozen training list (the data itself is seeded by tasks.SEED)."""
    data = tasks.build_all()
    items: list[dict] = []
    for name in task_names:
        rows = data[name]["train"]
        items += rows if n_per_task <= 0 else rows[:n_per_task]
    return items


def presentation_order(items: list[dict], presentations: int, seed: int) -> list[int]:
    """Shuffled epochs, repeated until the budget is used, so every item is seen as evenly as possible."""
    rng = random.Random(seed)
    order: list[int] = []
    while len(order) < presentations:
        epoch = list(range(len(items)))
        rng.shuffle(epoch)
        order += epoch
    return order[:presentations]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--tasks", required=True, nargs="+", help="task names from tasks.build_all()")
    ap.add_argument("--n-per-task", type=int, default=0, help="first n training items per task; 0 = all")
    ap.add_argument("--updates", type=int, default=2500)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-every", type=int, default=0, help="also save a checkpoint every N updates")
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="write a resumable checkpoint (LoRA + optimizer + RNG) every N updates; a rerun resumes from it")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--roi-weight", type=float, default=1.0,
                    help="loss weight of latents inside the answer area (answer box / option row) relative to 1 elsewhere")
    ap.add_argument("--init-lora", type=Path, help="continue training this LoRA (curriculum); rank must match")
    ap.add_argument("--extra-lora", nargs="*", default=[], help="extra DiT linear-layer suffixes (e.g. connector)")
    ap.add_argument("--encoder-lora-layers", type=int, default=0,
                    help="also train LoRA on the attention of the last N text-encoder decoder layers (Qwen-Image-2.1)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "lora.safetensors").exists():
        raise SystemExit(f"{a.out} already has a finished run")

    items = training_items(a.tasks, a.n_per_task, a.seed)
    presentations = a.updates * a.accum
    order = presentation_order(items, presentations, a.seed)
    ckpt_path = a.out / "resume.pt"
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    run_file = a.out / ("run.json" if resume is None else f"run_resume_{resume['update']:06d}.json")
    run_file.write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "git_commit": code_commit(),
        "prompt": tasks.GENERIC_PROMPT, "per_item_prompts": any("prompt" in i for i in items), "n_items": len(items), "presentations": presentations,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": subprocess.getoutput("hostname"),
        "gpu": subprocess.getoutput("nvidia-smi -L | head -1")}, indent=1))

    torch.manual_seed(a.seed)
    generator = torch.Generator().manual_seed(a.seed)
    adapter = ADAPTERS[a.model]("cuda").load()
    if resume is not None:  # resume: the checkpoint holds the full LoRA state, so it replaces --init-lora
        names = lora.inject(adapter.dit, lambda n: n in set(resume["dit_names"]), a.rank, a.rank)
    elif a.init_lora:
        print("init lora", lora.load({"dit": adapter.dit, "encoder": adapter.encoder}, a.init_lora, a.rank, a.rank), flush=True)
        names = [n for n, _ in lora.lora_modules(adapter.dit)]
    else:
        names = lora.inject(adapter.dit, lora.default_matcher(tuple(adapter.lora_suffixes) + tuple(a.extra_lora)),
                            a.rank, a.rank)
    lora.freeze_base(adapter.dit)
    enc_names: list[str] = []
    if a.encoder_lora_layers:
        if a.model != "qwen_image_21":
            raise SystemExit("encoder LoRA is only wired for qwen_image_21 (LongCat detaches its encoder output)")
        layers = adapter.encoder.config.text_config.num_hidden_layers
        keep = {f"model.language_model.layers.{i}.self_attn.{p}" for i in range(layers - a.encoder_lora_layers, layers)
                for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
        enc_names = lora.inject(adapter.encoder, lambda n: n in keep, a.rank, a.rank)
        lora.freeze_base(adapter.encoder)
        adapter.encoder.gradient_checkpointing_enable()
        adapter.encoder.config.use_cache = False
    if a.grad_checkpointing:
        adapter.dit.enable_gradient_checkpointing()
    params = lora.trainable_parameters(adapter.dit) + lora.trainable_parameters(adapter.encoder)
    print(f"items={len(items)} presentations={presentations} lora_layers={len(names)} encoder_lora={len(enc_names)} "
          f"lora_params={sum(p.numel() for p in params)}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    roots = {"dit": adapter.dit, **({"encoder": adapter.encoder} if enc_names else {})}

    start_micro, elapsed_before = 0, 0.0
    if resume is not None:
        for prefix, root in roots.items():
            for name, module in lora.lora_modules(root):
                module.down.weight.data.copy_(resume["lora"][f"{prefix}/{name}/down"])
                module.up.weight.data.copy_(resume["lora"][f"{prefix}/{name}/up"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["generator"])
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"])
        start_micro, elapsed_before = resume["update"] * a.accum, resume["elapsed"]
        print(f"resumed from update {resume['update']}", flush=True)

    def save_checkpoint(update: int) -> None:
        state = {"update": update, "elapsed": elapsed_before + time.time() - started, "optimizer": optimizer.state_dict(),
                 "generator": generator.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "dit_names": [n for n, _ in lora.lora_modules(adapter.dit)],
                 "lora": {f"{prefix}/{name}/{part}": getattr(module, part).weight.detach().cpu()
                          for prefix, root in roots.items() for name, module in lora.lora_modules(root)
                          for part in ("down", "up")}}
        tmp = a.out / "resume.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)

    log = (a.out / "loss.jsonl").open("a")
    started, running = time.time(), 0.0
    adapter.dit.train()
    for micro, index in enumerate(order, start=1):
        if micro <= start_micro:
            continue
        item = items[index]
        source, target = render.render_input(item), render.render_target(item)
        with torch.no_grad():
            z0, z_cond = adapter.latent(target), adapter.latent(source)
        with torch.set_grad_enabled(bool(enc_names)):
            cond = adapter.condition(item.get("prompt", tasks.GENERIC_PROMPT), source)
        noise = torch.randn(z0.shape, generator=generator).to(z0.device, z0.dtype)
        timestep, sigma = adapter.sample_time(generator)
        sigma = sigma.to(z0.dtype).reshape(1, *([1] * (z0.ndim - 1)))
        zt = (1 - sigma) * z0 + sigma * noise
        pred = adapter.predict(zt, timestep, z_cond, cond)
        err = (pred.float() - (noise - z0).float()) ** 2
        if a.roi_weight != 1.0:
            weight = roi_weight_map(item, z0.shape, a.roi_weight, z0.device)
            loss = (err * weight).sum() / weight.sum()
        else:
            loss = err.mean()
        (loss / a.accum).backward()
        running += loss.item()
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update = micro // a.accum
            if update % 10 == 0 or update == a.updates:
                log.write(json.dumps({"update": update, "loss": running / a.accum, "t": round(time.time() - started, 1)}) + "\n")
                log.flush()
            if update % 50 == 0:
                print(f"update {update}/{a.updates} loss {running / a.accum:.4f} {time.time() - started:.0f}s "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            running = 0.0
            if a.ckpt_every and update % a.ckpt_every == 0 and update != a.updates:
                save_checkpoint(update)
            if a.save_every and update % a.save_every == 0 and update != a.updates:
                lora.save({"dit": adapter.dit, **({"encoder": adapter.encoder} if enc_names else {})},
                          a.out / f"lora_{update:06d}.safetensors", {"update": update, "rank": a.rank})
    lora.save(roots, a.out / "lora.safetensors", {"update": a.updates, "rank": a.rank, "model": a.model})
    (a.out / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                 "seconds": elapsed_before + time.time() - started,
                                                 "resumed_from": resume["update"] if resume else 0,
                                                 "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=1))
    ckpt_path.unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
