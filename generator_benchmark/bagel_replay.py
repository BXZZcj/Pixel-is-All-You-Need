"""Replay data for the BAGEL fine-tuning: base BAGEL draws its own prior-preservation samples (DreamBooth-style: the
replay targets are what the untrained model itself produces, so replay can only hold the model where it was, never
teach it new skills). Rows come from replay_prompts.py (prompt, edit instruction, modes, size):
  t2i   plain = the GenEval path ([prompt] -> image), think = the WISE path ([system][prompt][think] -> image)
  edit  of that t2i image: plain = the GEdit path (cfg_img 1.5 on [0.4, 1], no think),
        think = the RISE path (a sampled think, cfg_img 2)
Outputs DEST/t2i/<id>.png and DEST/edit/<id>.png, each with a .json (mode, think, seed, ...; written before the png,
so a png marks a finished row). Row r<i> is seeded from i (t2i 7000000 + i, edit 8000000 + i), so shards and resumes
change nothing.
`pack` checks every row and writes DEST/replay.jsonl (one t2i and one edit sample per row, paths relative to DEST),
DEST/pack.json and DEST/_done/complete, which `bagel_train prepare --replay DEST` requires.
usage: python -m generator_benchmark.bagel_replay gen --prompts P --dest DEST [--shard I --nshards N] [--limit K] --out LOGDIR
       python -m generator_benchmark.bagel_replay pack --prompts P --dest DEST --nshards N [--partial] --out LOGDIR
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from .bagel_bench import CFG_IMG, GEDIT, GENEVAL, IMAGE_SEED_OFFSET, SYSTEM_PROMPT, WISE, Bagel, seed_all

T2I_SEED, EDIT_SEED = 7000000, 8000000


class ReplayBagel(Bagel):
    def t2i_plain(self, prompt, seed, h, w):
        """= Bagel.t2i_batch (GenEval's generate_image) for one prompt at size (h, w)."""
        m, c = self.model, GENEVAL
        gi, lens, rope = m.prepare_prompts(curr_kvlens=[0], curr_rope=[0], prompts=[prompt], tokenizer=self.tok,
                                           new_token_ids=self.ids)
        with self.torch.amp.autocast("cuda", enabled=True, dtype=self.torch.float16):   # sic, as the official script
            past = m.forward_cache_update_text(self.cache(), **self.dev(gi))
        seed_all(seed + IMAGE_SEED_OFFSET)
        gi = self.dev(m.prepare_vae_latent(curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)], new_token_ids=self.ids))
        cfg = self.dev(m.prepare_vae_latent_cfg(curr_kvlens=[0], curr_rope=[0], image_sizes=[(h, w)]))
        with self.ac():
            latents = m.generate_image(
                past_key_values=past, num_timesteps=c["steps"], cfg_text_scale=c["cfg"],
                cfg_interval=list(c["cfg_interval"]), cfg_renorm_min=0.0, timestep_shift=c["shift"],
                cfg_text_past_key_values=self.cache(),
                cfg_text_packed_position_ids=cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=cfg["cfg_packed_key_value_indexes"], **gi)
        return self.decode([latents[0]], h, w)

    def t2i_think(self, prompt, seed, h, w, max_len=2048):
        """= Bagel.t2i with a generated think (WISE's generate_image_with_think) at size (h, w)."""
        m, c = self.model, WISE
        past, lens, rope = self.text(self.cache(), [0], [0], SYSTEM_PROMPT)
        cfg = self.dev(m.prepare_vae_latent_cfg(curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)]))
        past, lens, rope = self.text(past, lens, rope, prompt)
        seed_all(seed)
        think = self.think(copy.deepcopy(past), lens, rope, True, max_len)
        past, lens, rope = self.text(past, lens, rope, think)
        seed_all(seed + IMAGE_SEED_OFFSET)
        gi = self.dev(m.prepare_vae_latent(curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)], new_token_ids=self.ids))
        with self.ac():
            latent = m.generate_image(
                past_key_values=past, num_timesteps=c["steps"], cfg_text_scale=c["cfg"],
                cfg_interval=list(c["cfg_interval"]), timestep_shift=c["shift"], cfg_renorm_min=0.0,
                cfg_renorm_type=c["renorm"], cfg_text_past_key_values=None,
                cfg_text_packed_position_ids=cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=cfg["cfg_packed_key_value_indexes"], **gi)
        return self.decode(latent, h, w), think


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def save(png: Path, image, meta: dict) -> None:
    png.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    tmp = png.with_name(png.stem + ".tmp.png")
    image.save(tmp)
    os.replace(tmp, png)


def gen_main(a: argparse.Namespace) -> None:
    from . import bagel_generate  # noqa: F401  (puts the flash-attn shim and the BAGEL source on sys.path)
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from data.data_utils import pil_img2rgb
    from PIL import Image
    rows = load_rows(a.prompts)
    lo, hi = a.shard * len(rows) // a.nshards, (a.shard + 1) * len(rows) // a.nshards   # contiguous: same mode mix
    mine = rows[lo:hi][:a.limit]
    for d in ("t2i", "edit"):
        (a.dest / d).mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    left = [r for r in mine if not (a.dest / "edit" / f"{r['id']}.png").exists()]
    print(f"shard {a.shard}/{a.nshards}: rows {lo}-{hi}, {len(mine)} rows, {len(left)} to do", flush=True)
    model = ReplayBagel(None) if left else None
    started = time.time()
    for k, r in enumerate(left, start=1):
        t0, i = time.time(), int(r["id"][1:])
        t2i_png = a.dest / "t2i" / f"{r['id']}.png"
        if not t2i_png.exists():
            h, w = r["size"]
            seed = T2I_SEED + i
            if r["t2i_mode"] == "think":
                image, think = model.t2i_think(r["prompt"], seed, h, w)
            else:
                image, think = model.t2i_plain(r["prompt"], seed, h, w), None
            save(t2i_png, image, {"id": r["id"], "mode": r["t2i_mode"], "prompt": r["prompt"], "think": think,
                                  "seed": seed, "size": [h, w]})
        t1 = time.time()
        source = pil_img2rgb(Image.open(t2i_png))
        seed = EDIT_SEED + i
        if r["edit_mode"] == "think":
            image, think = model.edit([source], r["edit"], None, True, seed, CFG_IMG["rise"])
            cfg = {"cfg_img": CFG_IMG["rise"], "cfg_interval": [0.0, 1.0]}
        else:
            image, think = model.edit([source], r["edit"], None, False, seed, GEDIT["cfg_img"],
                                      cfg_interval=GEDIT["cfg_interval"])
            cfg = {"cfg_img": GEDIT["cfg_img"], "cfg_interval": list(GEDIT["cfg_interval"])}
        save(a.dest / "edit" / f"{r['id']}.png", image.convert("RGB"),
             {"id": r["id"], "mode": r["edit_mode"], "instruction": r["edit"], "edit_type": r["edit_type"],
              "think": think, "seed": seed, "size": list(image.size[::-1]), **cfg})
        if k % 5 == 0 or k == len(left):
            print(f"{k}/{len(left)} {time.time() - started:.0f}s (last: t2i {r['t2i_mode']} {t1 - t0:.1f}s, "
                  f"edit {r['edit_mode']} {time.time() - t1:.1f}s)", flush=True)
    done = a.dest / "_done"
    done.mkdir(exist_ok=True)
    tag = f"s{a.shard}of{a.nshards}" + (f"_limit{a.limit}" if a.limit is not None else "")
    (done / tag).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("done", flush=True)


def pack_main(a: argparse.Namespace) -> None:
    from PIL import Image, ImageChops, ImageStat
    rows = load_rows(a.prompts)   # complete = every row has both pngs (shards may be split finer, e.g. 32-39 of 40)
    a.out.mkdir(parents=True, exist_ok=True)
    out, stats = [], {"rows": len(rows), "missing": 0, "blank": 0, "unchanged_edits": 0, "by_mode": {}}
    for r in rows:
        t2i, edit = a.dest / "t2i" / f"{r['id']}.png", a.dest / "edit" / f"{r['id']}.png"
        if not (t2i.exists() and edit.exists()):
            stats["missing"] += 1
            continue
        ti, ei = Image.open(t2i).convert("RGB"), Image.open(edit).convert("RGB")
        if min(ImageStat.Stat(ti.convert("L")).stddev[0], ImageStat.Stat(ei.convert("L")).stddev[0]) < 2:
            stats["blank"] += 1   # a flat image: a failed draw, not a sample of the model's distribution
            continue
        if ti.size == ei.size and max(ImageStat.Stat(ImageChops.difference(ti, ei)).mean) < 2:
            stats["unchanged_edits"] += 1   # kept: base BAGEL does this, and replay preserves what base does
        tm, em = json.loads(t2i.with_suffix(".json").read_text()), json.loads(edit.with_suffix(".json").read_text())
        out.append({"id": r["id"], "kind": "t2i", "mode": tm["mode"], "prompt": r["prompt"], "think": tm["think"],
                    "image": f"t2i/{r['id']}.png", "category": r["category"]})
        out.append({"id": r["id"], "kind": "edit", "mode": em["mode"], "prompt": r["edit"], "think": em["think"],
                    "input": f"t2i/{r['id']}.png", "output": f"edit/{r['id']}.png", "edit_type": r["edit_type"]})
        for x in out[-2:]:
            key = f"{x['kind']}_{x['mode']}"
            stats["by_mode"][key] = stats["by_mode"].get(key, 0) + 1
    stats["samples"] = len(out)
    if stats["missing"] and not a.partial:
        sys.exit(f"{stats['missing']} of {len(rows)} rows missing")
    tmp = a.dest / "replay.tmp.jsonl"
    with tmp.open("w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, a.dest / "replay.jsonl")
    (a.dest / "pack.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)
    if not a.partial:
        (a.dest / "_done" / "complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--limit", type=int, help="only the first K rows of the shard (smoke tests)")
    p = sub.add_parser("pack")
    p.add_argument("--partial", action="store_true", help="pack what is there, no _done/complete (smoke tests)")
    for s in (g, p):
        s.add_argument("--prompts", type=Path, required=True)
        s.add_argument("--dest", type=Path, required=True)
        s.add_argument("--nshards", type=int, default=1)
        s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"gen": gen_main, "pack": pack_main}[a.cmd](a)


if __name__ == "__main__":
    main()
