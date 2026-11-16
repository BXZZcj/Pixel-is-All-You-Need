"""Prompts for the BAGEL replay data: an LLM writes varied text-to-image prompts, each with one image-edit
instruction that refers only to what the prompt describes. Base BAGEL draws them (bagel_replay.py) as prior-preservation
replay for the fine-tuning.

Each row also fixes how base BAGEL draws it, so the replay covers every inference format: t2i_mode plain (GenEval path:
[prompt] -> image) or think ([system][prompt][think] -> image, the WISE path), edit_mode plain (GEdit path:
[image][instruction]) or think ([system][image][instruction][think], the RISE path), and the t2i size. The patterns
cycle every 16 rows, so any contiguous block of rows has the same mix. Prompts equal to a GenEval or WISE prompt (after
lowercasing) are dropped.
The LLM is reached through an OpenAI-compatible endpoint: JUDGE_BASE_URL and JUDGE_API_KEY (the judge's endpoint),
model --model (default qwen3.8-max).
usage: python -m generator_benchmark.replay_prompts --out PROMPTS.jsonl [--per-category 32] [--workers 16]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
from pathlib import Path

from .paths import src

BAGEL_SRC = os.environ.get("BAGEL_SRC", src("Bagel"))
BENCH_PROMPTS = (f"{BAGEL_SRC}/eval/gen/geneval/prompts/evaluation_metadata.jsonl", f"{BAGEL_SRC}/eval/gen/wise/final_data.json")
CATEGORIES = [
    "close-up portraits of people of varied ages and backgrounds", "people doing everyday activities at home or outdoors",
    "sports and athletes in action", "children and families", "people at work in different professions",
    "dogs, cats and other pets", "wild animals in their habitats", "birds", "marine life and underwater scenes",
    "insects, reptiles and small creatures", "plated food and dishes from many cuisines", "desserts, pastries and drinks",
    "fruits, vegetables and market stalls", "city streets and buildings", "home interiors: living rooms, kitchens, bedrooms",
    "mountains, lakes and valleys", "beaches, coasts and seascapes", "forests, gardens and flowers",
    "weather and seasons: snow, rain, fog, autumn", "night scenes and illuminated cityscapes", "cars, motorcycles and bicycles",
    "trains, airplanes and boats", "product shots of gadgets and household items", "clothing, shoes and accessories",
    "furniture and decor objects", "toys and board games", "musical instruments", "books, stationery and desks",
    "shop fronts and street signs with a short word or two of text", "posters, cards and packaging with a short title",
    "paintings in classical styles (oil, watercolor, ink, impressionist)", "illustrations: anime, cartoon, comic, storybook",
    "3D renders, low-poly, clay and miniature dioramas", "fantasy and science-fiction scenes",
    "historical scenes, costumes and old towns", "still lifes of several objects arranged in clear spatial relations",
    "groups of objects with specific counts and colors", "macro close-ups of textures and materials",
    "aerial and bird's-eye views", "minimalist designs, patterns and flat graphic art",
]
EDIT_TYPES = {
    "background_change": "replace or change the background", "color_alter": "change the color of an object",
    "material_alter": "change what an object is made of", "motion_change": "change a person's or animal's pose or action",
    "ps_human": "retouch a person (expression, hair, makeup, age)", "style_change": "render the image in another style",
    "subject-add": "add an object or creature", "subject-remove": "remove an object or creature",
    "subject-replace": "replace an object or creature with another", "text_change": "change the quoted text",
    "tone_transfer": "change the lighting, weather, season or time of day",
}
ASK = """Write {k} varied text-to-image prompts in this category: {category}.

Rules for the prompts:
- Describe directly what should be visible. No riddles, puzzles or questions, nothing that needs outside knowledge to draw.
- Lengths: about a third short (5-12 words), a third medium (15-35 words), a third long and detailed (40-80 words).
- Vary the subjects, settings, viewpoints, lighting, color palettes and media (photo, painting, illustration, render) as the category allows.
- No real people's names, real brands or logos, copyrighted characters, or unsafe content. Do not start with "a photo of".
- Any text that should appear in the image is short (one to three words) and in double quotes.

For each prompt also write one instruction to edit the image that the prompt would produce, like an image-editing benchmark instruction: one clear change, imperative, 4-20 words, referring only to things the prompt describes or to the image as a whole. Give its type, one of:
{types}
Vary the types across the {k} prompts. Use ps_human only for clearly visible people, text_change only when the prompt has quoted text, motion_change only for people or animals.

Output exactly {k} lines and nothing else, each one JSON object: {{"prompt": "...", "edit": "...", "edit_type": "..."}}"""
# (t2i_mode, edit_mode, (height, width)) by row index; every 16 rows cover each combination once
SIZES = [(1024, 1024), (1024, 1024), (768, 1024), (1024, 768)]


def row_plan(i: int) -> dict:
    return {"t2i_mode": ("plain", "think")[i % 2], "edit_mode": ("plain", "think")[(i // 2) % 2],
            "size": list(SIZES[(i // 4) % 4])}


def bench_prompts() -> set[str]:
    out = set()
    for path in BENCH_PROMPTS:
        if not os.path.exists(path):
            print(f"warning: {path} not found, not checked", flush=True)
            continue
        text = Path(path).read_text()
        rows = json.loads(text) if path.endswith(".json") else [json.loads(x) for x in text.splitlines() if x.strip()]
        out |= {str(r.get("prompt") or r.get("Prompt")).strip().lower() for r in rows}
    return out


def parse(reply: str) -> list[dict]:
    rows = []
    for line in (reply or "").splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict) or r.get("edit_type") not in EDIT_TYPES:
            continue
        p, e = str(r.get("prompt", "")).strip(), str(r.get("edit", "")).strip()
        if 3 <= len(p.split()) <= 120 and 2 <= len(e.split()) <= 40:
            rows.append({"prompt": p, "edit": e, "edit_type": r["edit_type"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--categories", type=int, default=None, help="only the first K categories (smoke test)")
    ap.add_argument("--model", default="qwen3.8-max")
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["JUDGE_BASE_URL"].rstrip("/") + "/v1", api_key=os.environ["JUDGE_API_KEY"])
    types = "\n".join(f"- {k}: {v}" for k, v in EDIT_TYPES.items())
    raw = a.out.with_suffix(".raw.jsonl")

    def one(category: str) -> list[dict]:
        best = []
        for t in range(a.tries):
            msg = [{"role": "user", "content": ASK.format(k=a.per_category, category=category, types=types)}]
            try:
                reply = client.chat.completions.create(model=a.model, messages=msg).choices[0].message.content
            except Exception as exc:   # noqa: BLE001
                print(f"{category[:40]}: try {t}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            rows = parse(reply)
            with open(raw, "a") as f:
                f.write(json.dumps({"category": category, "try": t, "n": len(rows), "reply": reply}) + "\n")
            if len(rows) > len(best):
                best = rows
            if len(rows) >= 0.9 * a.per_category:
                break
        print(f"{category[:50]}: {len(best)} rows", flush=True)
        return [dict(r, category=category) for r in best[:a.per_category]]

    with cf.ThreadPoolExecutor(a.workers) as ex:
        per_cat = list(ex.map(one, CATEGORIES[:a.categories]))
    banned, seen, rows = bench_prompts(), set(), []
    dropped = {"duplicate": 0, "bench": 0}
    for cat_rows in per_cat:
        for r in cat_rows:
            key = re.sub(r"\s+", " ", r["prompt"].lower()).strip()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            if key in banned or key.rstrip(".") in banned:
                dropped["bench"] += 1
                continue
            seen.add(key)
            rows.append(r)
    random.Random(0).shuffle(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(dict({"id": f"r{i:05d}"}, **r, **row_plan(i)), ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    by_type = {}
    for r in rows:
        by_type[r["edit_type"]] = by_type.get(r["edit_type"], 0) + 1
    print(json.dumps({"rows": len(rows), "dropped": dropped, "edit_types": by_type}), flush=True)


if __name__ == "__main__":
    main()
