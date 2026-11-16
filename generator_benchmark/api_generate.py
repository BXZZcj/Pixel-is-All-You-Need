"""The benchmark canvases through closed image editing models. Same canvases and fixed instruction as the open models,
no answer in the prompt (their built-in rewriter cannot be disabled: the Rewrite column of Table 1). Output layout is
that of generate.py, so cases.py works unchanged. Gateway: IMAGE_API_BASE_URL (OpenAI-compatible), key: IMAGE_API_KEY.

usage: python -m generator_benchmark.api_generate --model MODEL --out DIR --per-level 60 [--workers 6]
       (models whose name starts with gpt-image use /v1/images/edits, others /v1/chat/completions)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import render, tasks
from .api_image import ChatImageClient, ImagesEditClient
item_seed = tasks.item_seed

LEVELS = {"qa": ("L1", "L2R", "L3", "L4", "L2F"), "context": ("L5A", "L5B"), "arithmetic": ("A1", "A2"),
          "instruction": ("I1", "I2"), "visual": ("L6A", "L6I", "L6B"), "copy_complete": ("copy",)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-level", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N items of every level (Table 1 top-up)")
    ap.add_argument("--api-key-env", default="IMAGE_API_KEY")
    a = ap.parse_args()
    config = {"api_model": a.model, "api_base_url": os.environ["IMAGE_API_BASE_URL"], "api_key_env": a.api_key_env,
              "api_timeout_seconds": 600, "api_max_attempts": 5}
    client = (ImagesEditClient if a.model.startswith("gpt-image") else ChatImageClient)(config)
    data = tasks.build_all()
    items = []
    for task, levels in LEVELS.items():
        for level in levels:
            items += [i for i in data[task]["test"] if i["level"] == level][a.offset:a.offset + a.per_level]
    a.out.mkdir(parents=True, exist_ok=True)
    lock, done = threading.Lock(), [0]
    manifest = (a.out / "manifest.jsonl").open("a")

    def one(item):
        d = a.out / item["item_id"]
        if (d / "output.png").exists():
            return
        d.mkdir(exist_ok=True)
        source = render.render_input(item)
        source.save(d / "source.png")
        try:
            image = client.edit(input_path=d / "source.png", prompt=tasks.GENERIC_PROMPT, receipt_dir=d, size=(1024, 1024))
        except Exception as exc:  # noqa: BLE001 - recorded, counted as missing
            (d / "error.txt").write_text(f"{type(exc).__name__}: {exc}")
            return
        if image.size != (1024, 1024):
            (d / "error.txt").write_text(f"size {image.size}")
            return
        image.save(d / ".output.tmp.png")
        (d / ".output.tmp.png").rename(d / "output.png")
        with lock:
            manifest.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                       "seed": item_seed(item["item_id"]), "model": a.model, "route": "native"}) + "\n")
            manifest.flush()
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"{done[0]}/{len(items)} {time.strftime('%H:%M:%S')}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, items))
    print("done", flush=True)


if __name__ == "__main__":
    main()
