"""Cosmos3's own prompt upsampler as the Rewrite of Cosmos3 (video).

The upsampler is cosmos-framework's `cosmos_framework.inference.prompt_upsampling` (NVIDIA/cosmos-framework, image2video
mode, its built-in external-API templates; qwen3.8-max in place of the example's model, as for Wan2.2's extension). It
runs in cosmos-framework's own environment, so this module only writes its inputs and collects its outputs:

  python -m generator_benchmark.cosmos_upsample inputs --set t1 --work W          (canvases, prompt and image lists)
  # in the cosmos-framework env, for each chunk k (the commands are printed by `inputs`):
  PROMPT_UPSAMPLER_ENDPOINT_URL=$REWRITER_BASE_URL/v1 PROMPT_UPSAMPLER_API_TOKEN=$REWRITER_API_KEY \
    python -m cosmos_framework.inference.prompt_upsampling --input W/prompts_rw_k.txt --image-list W/images_k.txt \
    --output W/out_rw_k --mode image2video --model qwen3.8-max --resolution 720 --aspect-ratio "1,1" --duration "4s" \
    --fps 24 --timeout-s 600
  python -m generator_benchmark.cosmos_upsample collect --set t1 --work W --out rewrites/cosmos3.jsonl

`collect` writes rewrite.py-style rows (prompt = the upsampled default instruction; prompt_noans = the rule-based
deletion, Factual and Arithmetic only, to be passed through delete_answer.py --mode rewrite). Generate with
video_generate --model cosmos3 --variant sq_json --route rewrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import render, rewrite, video_generate as V

CHUNK = 30
UPSAMPLE = ("python -m cosmos_framework.inference.prompt_upsampling --input {w}/prompts_rw_{k}.txt "
            "--image-list {w}/images_{k}.txt --output {w}/out_rw_{k} --mode image2video --model qwen3.8-max "
            "--resolution 720 --aspect-ratio \"1,1\" --duration \"4s\" --fps 24 --timeout-s 600")


def inputs(a: argparse.Namespace) -> None:
    w = a.work.resolve()
    (w / "cards").mkdir(parents=True, exist_ok=True)
    items = V.select(a.set)
    for c in range(0, len(items), CHUNK):
        ch, k = items[c:c + CHUNK], f"{c // CHUNK:02d}"
        imgs = []
        for it in ch:
            p = w / "cards" / f"{it['item_id']}.png"
            if not p.exists():
                render.render_input(it).save(p)
            imgs.append(str(p))
        (w / f"ids_{k}.txt").write_text("\n".join(i["item_id"] for i in ch) + "\n")
        (w / f"images_{k}.txt").write_text("\n".join(imgs) + "\n")
        (w / f"prompts_rw_{k}.txt").write_text("\n".join(V.VIDEO_PROMPT for _ in ch) + "\n")
        print(UPSAMPLE.format(w=w, k=k))
    print(f"# {len(items)} canvases", flush=True)


def collect(a: argparse.Namespace) -> None:
    w = a.work.resolve()
    items = {i["item_id"]: i for i in V.select(a.set)}
    rows = []
    for f in sorted(w.glob("ids_*.txt")):
        k = f.stem.split("_")[1]
        for j, iid in enumerate(f.read_text().split()):
            p = w / f"out_rw_{k}" / f"prompt_{j}.json"
            if not p.exists():
                continue
            prompt, it = json.loads(p.read_text())["prompt"], items[iid]
            noans, n = rewrite.strip_answer(prompt, it["target"])
            rows.append({"item_id": iid, "task": it["task"], "level": it["level"], "target": it["target"],
                         "rewriter": "cosmos3_upsampler", "backend": "api", "prompt": prompt,
                         "prompt_noans": noans if it["task"] in rewrite.ABLATE_TASKS else None, "n_removed": n})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} of {len(items)} upsampled", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inputs", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--set", default="t1", choices=sorted(V.SETS))
        s.add_argument("--work", type=Path, required=True)
        if name == "collect":
            s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"inputs": inputs, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
