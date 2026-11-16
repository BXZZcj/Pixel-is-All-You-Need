"""Training canvases for the BAGEL fine-tuning, and BAGEL's own thinks on them.

train_items() is the training mix of bagel_train.py. The `think` step runs BAGEL's official think pipeline (as
bagel_generate --mode think, greedy, without generating the image) on every training canvas; bagel_train.py keeps, for
its pixel arm, the canvases whose think states the answer (states), i.e. the items BAGEL's text module can answer, and
from whose think every answer-stating sentence can be removed (drop_answer).

usage: python -m generator_benchmark.bagel_cards think --shard I --nshards N --out DIR      (DIR/think.jsonl, resumable)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from . import render, tasks

# training mix: name -> cap (None = all). qa_big already contains every qa training fact (more phrasings).
TRAIN_MIX = {"qa_big": 2000, "arithmetic": None, "instruction": None, "context": None, "visual": None,
             "select": 600, "copy_complete": 400}
MIX_SEED = tasks.SEED + 1600


def train_items() -> list[dict]:
    data = tasks.build_all()
    rng = random.Random(MIX_SEED)
    out, seen = [], set()
    for name, cap in TRAIN_MIX.items():
        pool = list(data[name]["train"])
        if cap is not None and len(pool) > cap:
            pool = rng.sample(pool, cap)
        for item in pool:
            if item["item_id"] not in seen:
                seen.add(item["item_id"])
                out.append(item)
    return out


def states(text: str, answer: str) -> bool:
    """The answer occurs in the text as a whole word/number (case-insensitive; digits need digit boundaries)."""
    answer = answer.strip()
    if not answer:
        return False
    if answer.isdigit():
        return re.search(rf"(?<![0-9]){re.escape(answer)}(?![0-9])", text) is not None
    return tasks.mentions(text, answer)


# sentence boundary: . ! ? (optionally closing quote/bracket) + space, next sentence starts upper-case/digit/quote
SENTENCE_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019)\]]))\s+(?=[\"'\u201c(\[]?[A-Z0-9])")
NOANS_FALLBACK = "The answer should be written in the answer box, and the rest of the card should remain unchanged."


def drop_answer(think: str, answer: str) -> str:
    """The noans think: remove every sentence that states the answer (keeps the <think> tags and the rest verbatim)."""
    body = think.strip()
    head, tail = "", ""
    if body.startswith("<think>"):
        head, body = "<think>\n", body[len("<think>"):]
    if body.endswith("</think>"):
        body, tail = body[: -len("</think>")], "\n</think>"
    kept = [s.strip() for s in SENTENCE_SPLIT.split(body.strip()) if s.strip() and not states(s, answer)]
    return head + (" ".join(kept) or NOANS_FALLBACK) + tail


def think_main(a: argparse.Namespace) -> None:
    import torch

    from .bagel_generate import load  # puts the BAGEL source on sys.path
    from data.data_utils import pil_img2rgb
    from inferencer import GEN_THINK_SYSTEM_PROMPT
    items = [it for n, it in enumerate(train_items()) if n % a.nshards == a.shard]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "think.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["item_id"] for line in path.read_text().splitlines() if line.strip()}
    todo = [it for it in items if it["item_id"] not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        print("done", flush=True)
        return
    inf = load()
    started = time.time()
    with path.open("a") as f:
        for n, item in enumerate(todo, start=1):
            t0 = time.time()
            ctx = inf.init_gen_context()
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                ctx = inf.update_context_text(GEN_THINK_SYSTEM_PROMPT, ctx)
                image = inf.vae_transform.resize_transform(pil_img2rgb(render.render_input(item)))
                ctx = inf.update_context_image(image, ctx, vae=True, vit=True)
                ctx = inf.update_context_text(tasks.GENERIC_PROMPT, ctx)
                think = inf.gen_text(ctx, max_length=a.max_tokens, do_sample=False, temperature=0.3)
            f.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                "mode": item["mode"], "expected": item["expected"], "think": think,
                                "states_answer": states(think, item["expected"]),
                                "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{n}/{len(todo)} {time.time() - started:.0f}s", flush=True)
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("think")
    t.add_argument("--shard", type=int, default=0)
    t.add_argument("--nshards", type=int, default=1)
    t.add_argument("--max-tokens", type=int, default=1000)
    t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "think":
        think_main(a)


if __name__ == "__main__":
    main()
