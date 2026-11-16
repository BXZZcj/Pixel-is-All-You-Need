"""Generate test outputs for one image editing model (+ an optional LoRA from train.py) on one GPU; resumable, one PNG
per item. The seed comes from the item id.

usage: python -m generator_benchmark.generate --model M [--lora RUN/lora.safetensors] --tasks qa arithmetic --out DIR
Outputs: DIR/<item_id>/{output.png,source.png} and DIR/manifest.jsonl.

--route decides what the prompt is:
  native            the fixed instruction (Default; also the LoRA evaluation)
  rewrite           a rewriter's output for this item (--rewrites jsonl from rewrite.py / delete_answer.py, field
                    --rewrite-field prompt (Rewrite) | prompt_noans (Delete)); only the items in that file are generated
  render            the correct answer itself is written in the prompt
  render_q          the same with the answer in double quotes (LongCat-Image's model card: only quoted text is
                    encoded character by character)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import lora, render, tasks
from .models import ADAPTERS


ROUTES = ("native", "rewrite", "render", "render_q")
REWRITTEN: dict[str, str] = {}


def route_inputs(item: dict, route: str, base_prompt: str):
    """Returns (prompt, canvas given to the generator)."""
    real = render.render_input(item)
    if route == "native":
        return base_prompt, real
    if route == "rewrite":  # a rewriter's output (or its answer-deleted version) in place of the fixed instruction
        return REWRITTEN[item["item_id"]], real
    if item["mode"] != "write":
        raise ValueError("render routes are defined for write items only")
    q = '"' if route.endswith("_q") else ""
    return base_prompt + tasks.RENDER_INSTRUCTION.format(answer=q + item["target"] + q), real


def safe_open_meta(path: Path) -> str:
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return (f.metadata() or {}).get(lora.META_KEY, "{}")


item_seed = tasks.item_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--lora", type=Path)
    ap.add_argument("--rank", type=int, default=0, help="default: read from the LoRA file")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--levels", nargs="*", help="only these test levels")
    ap.add_argument("--split", choices=("test", "train"), default="test", help="train: recall of trained items")
    ap.add_argument("--limit", type=int, default=0, help="at most N items per task, in frozen order")
    ap.add_argument("--per-level", action="store_true",
                    help="--limit applies to every test level of a task: the first N items of each level, in frozen "
                         "order (Table 1: --per-level --limit 60)")
    ap.add_argument("--offset", type=int, default=0, help="with --per-level: skip the first N items of every level "
                    "(--per-level --limit 1000000 --offset 60 = the rest of each level)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--prompt", default=tasks.GENERIC_PROMPT)
    ap.add_argument("--route", default="native", choices=ROUTES)
    ap.add_argument("--rewrites", type=Path, nargs="*", default=[], help="rewrite rows (rewrite.py format) for --route rewrite")
    ap.add_argument("--rewrite-field", default="prompt", choices=("prompt", "prompt_noans"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    data = tasks.build_all()
    items = []
    for name in a.tasks:
        rows = [i for i in data[name][a.split] if not a.levels or i["level"] in a.levels]
        if a.per_level and a.limit:   # the first N items of every level
            for lv in dict.fromkeys(i["level"] for i in rows):
                items += [i for i in rows if i["level"] == lv][a.offset:a.limit]
        else:
            items += rows[:a.limit] if a.limit else rows
    if a.route == "rewrite":
        for f in a.rewrites:
            for line in f.read_text().splitlines():
                row = json.loads(line)
                if row.get(a.rewrite_field):
                    REWRITTEN[row["item_id"]] = row[a.rewrite_field]
        index = {i["item_id"]: i for name in a.tasks for i in data[name]["test"]}
        items = [index[k] for k in REWRITTEN if k in index]
    a.out.mkdir(parents=True, exist_ok=True)
    adapter = ADAPTERS[a.model]("cuda").load()
    if a.lora:
        rank = a.rank or json.loads(safe_open_meta(a.lora)).get("rank", 16)
        print("lora layers", lora.load({"dit": adapter.dit, "encoder": adapter.encoder}, a.lora, rank, rank), flush=True)
    adapter.dit.eval()
    started = time.time()
    with (a.out / "manifest.jsonl").open("a") as manifest:
        for n, item in enumerate(items, start=1):
            d = a.out / item["item_id"]
            if (d / "output.png").exists():
                continue
            d.mkdir(exist_ok=True)
            prompt, source = route_inputs(item, a.route, a.prompt)
            source.save(d / "source.png")
            image = adapter.generate(prompt, source, item_seed(item["item_id"]), a.steps)
            if image.size != (1024, 1024):
                raise RuntimeError(f"generator returned {image.size}")
            image.convert("RGB").save(d / ".output.tmp.png")
            (d / ".output.tmp.png").rename(d / "output.png")
            manifest.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                       "seed": item_seed(item["item_id"]), "model": a.model, "route": a.route,
                                       "rewrites": [str(f) for f in a.rewrites] or None, "rewrite_field": a.rewrite_field,
                                       "lora": str(a.lora) if a.lora else None}) + "\n")
            manifest.flush()
            if n % 20 == 0:
                print(f"{n}/{len(items)} {time.time() - started:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
