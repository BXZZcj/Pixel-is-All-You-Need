"""BAGEL (base, or with a fine-tuned checkpoint from bagel_train.py) on the held-out benchmark canvases.

The generation path mirrors BAGEL's official think-mode edit script (eval/gen/gen_images_mp_rise.py: same contexts,
CFG branches and settings), so base and fine-tuned weights are compared under one fixed protocol:
  cards         [system][canvas vae+vit][instruction][think] -> canvas; cfg_text [system][canvas], cfg_img
                [system][instruction][think]; cfg 4 / 2, serial_text_img, text_channel renorm, shift 3, 50 steps.
                The think is BAGEL's own greedy think (at most 1000 tokens). With --plain: the no-think path
                [canvas][instruction] (the Default condition).
  cards_render  the same canvases with the correct answer, quoted, appended to the instruction; no think (Render).
  und           BAGEL answers each canvas in text (understanding mode, greedy): what its text module knows.
Every random draw is seeded per item (42 + index: think; + 1000003: image noise; + 2000003: the VAE posterior sample of
the input image), so an output depends only on the item and the weights, and sharding or resuming changes nothing.
Outputs: DEST/<bench>[_plain]/gen/<item_id>/{output.png, source.png, think.txt} + gen/manifest.jsonl (the layout read
by cases.py), or DEST/und/chat_s<k>of<n>.jsonl.

usage: python -m generator_benchmark.bagel_bench --bench cards,cards_render --dest DIR [--ckpt gen.safetensors]
           [--plain] [--per-level 30] [--shard I --nshards N] [--limit K] --out LOGDIR
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

# byte-exact copy of the official eval scripts' prompt (= inferencer.GEN_THINK_SYSTEM_PROMPT; note the space before \n)
SYSTEM_PROMPT = ("You should first think about the planning process in the mind and then generate the image. \n"
                 "The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here "
                 "</think> image here")
EDIT = dict(cfg_text=4.0, cfg_interval=(0.0, 1.0), cfg_type="serial_text_img", renorm="text_channel", shift=3.0,
            steps=50, max_size=1024, min_size=512)
CFG_IMG = {"rise": 2.0, "cards": 2.0, "cards_render": 2.0}
WISE = dict(cfg=4.0, cfg_interval=(0.4, 1.0), shift=3.0, steps=50, res=1024, renorm="global")
GEDIT = dict(cfg_img=1.5, cfg_interval=(0.4, 1.0))   # the rest as EDIT
GENEVAL = dict(cfg=4.0, cfg_interval=(0.0, 1.0), shift=3.0, steps=50, res=1024, n=4)   # renorm: the default (global)
NO_THINK = ("cards_render", "und")
BENCHES = ("cards", "cards_render", "und")
CARDS = ("cards", "cards_render")
UND_Q = "Answer the question on the card. Answer with only the answer, in as few words as possible."
IMAGE_SEED_OFFSET = 1000003
VAE_SEED_OFFSET = 2000003


def items(bench: str, a: argparse.Namespace) -> list[dict]:
    """Each item: key (relative output path without .png), prompt, item, images (callable -> list of PIL)."""
    from . import render, tasks
    from .api_generate import LEVELS
    data = tasks.build_all()
    out = []
    for task in a.tasks or sorted(LEVELS):
        for level in LEVELS[task]:
            for it in [i for i in data[task]["test"] if i["level"] == level][:a.per_level]:
                prompt = tasks.GENERIC_PROMPT
                if bench == "cards_render":
                    prompt += tasks.RENDER_INSTRUCTION.format(answer=f'"{it["target"]}"')
                out.append({"key": f"gen/{it['item_id']}/output", "prompt": prompt, "item": it,
                            "images": (lambda it=it: [render.render_input(it)])})
    return out


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_scale(width, height, scale):   # = eval/gen/gen_images_mp_rise.py
    def _make_divisible(value, stride):
        return max(stride, int(round(value / stride) * stride))
    return _make_divisible(round(width * scale), 16), _make_divisible(round(height * scale), 16)


class Bagel:
    def __init__(self, ckpt: Path | None):
        import torch
        from .bagel_generate import load   # first: puts the flash-attn shim and the BAGEL source on sys.path
        from data.transforms import ImageTransform
        from modeling.bagel.qwen2_navit import NaiveCache
        inf = load()
        from inferencer import GEN_THINK_SYSTEM_PROMPT
        assert SYSTEM_PROMPT == GEN_THINK_SYSTEM_PROMPT, "think system prompt differs from BAGEL's"
        self.inf = inf
        self.model, self.vae, self.tok, self.ids = inf.model, inf.vae_model, inf.tokenizer, inf.new_token_ids
        self.vae_t, self.vit_t = ImageTransform(1024, 512, 16), ImageTransform(980, 378, 14)   # official eval
        self.cache = lambda: NaiveCache(self.model.config.llm_config.num_hidden_layers)
        self.torch = torch
        self.overlay = None
        if ckpt is not None:
            from safetensors.torch import load_file
            sd = load_file(str(ckpt), device="cuda:0")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            assert not unexpected, f"checkpoint keys not in BAGEL: {unexpected[:5]}"
            self.overlay = {"ckpt": str(ckpt), "tensors": len(sd)}
            print(f"overlay {ckpt}: {len(sd)} tensors", flush=True)
            del sd

    def dev(self, d: dict) -> dict:
        return {k: (v.to("cuda:0") if isinstance(v, self.torch.Tensor) else v) for k, v in d.items()}

    def ac(self):
        return self.torch.amp.autocast("cuda", enabled=True, dtype=self.torch.bfloat16)

    def text(self, past, lens, rope, text):
        gi, lens, rope = self.model.prepare_prompts(curr_kvlens=lens, curr_rope=rope, prompts=[text],
                                                    tokenizer=self.tok, new_token_ids=self.ids)
        with self.ac():
            past = self.model.forward_cache_update_text(past, **self.dev(gi))
        return past, lens, rope

    def think(self, past, lens, rope, do_sample: bool, max_len: int) -> str:
        gi = self.dev(self.model.prepare_start_tokens(lens, rope, self.ids))
        with self.ac():
            out = self.model.generate_text(past_key_values=past, max_length=max_len, do_sample=do_sample,
                                           temperature=0.3, end_token_id=self.ids["eos_token_id"], **gi)
        return self.tok.decode(out[:, 0]).split("<|im_end|>")[0].split("<|im_start|>")[1]

    def decode(self, latent, h, w):
        import torch
        from PIL import Image
        latent = latent[0].reshape(1, h // 16, w // 16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent).reshape(1, 16, h // 8, w // 8)
        with torch.no_grad():
            image = self.vae.decode(latent)
        return Image.fromarray(((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy())

    def edit(self, images, prompt, think, gen_think, seed, cfg_img, do_sample=True, max_len=2048,
             cfg_interval=EDIT["cfg_interval"]):
        """= editing_image_with_think / editing_image of eval/gen/gen_images_mp_rise.py (think=None, gen_think=False:
        the no-think path, = editing_with_text_img_cfg of gen_images_gedit.py with its cfg_interval)."""
        m, e = self.model, EDIT
        w, h = images[0].size
        scale = max(min(e["max_size"] / max(w, h), 1.0), e["min_size"] / min(w, h))
        w, h = apply_scale(w, h, scale)
        if max(w, h) > e["max_size"]:
            w, h = apply_scale(w, h, e["max_size"] / max(w, h))
        use_think = think is not None or gen_think
        past, lens, rope = self.cache(), [0], [0]
        if use_think:
            past, lens, rope = self.text(past, lens, rope, SYSTEM_PROMPT)
        seed_all(seed + VAE_SEED_OFFSET)   # the VAE encoder samples its posterior (DiagonalGaussian(sample=True))
        for image in images:
            gi, lens, rope = m.prepare_vae_images(curr_kvlens=lens, curr_rope=rope, images=[image],
                                                  transforms=self.vae_t, new_token_ids=self.ids)
            with self.ac():
                past = m.forward_cache_update_vae(self.vae, past, **self.dev(gi))
            gi, lens, rope = m.prepare_vit_images(curr_kvlens=lens, curr_rope=rope, images=[image],
                                                  transforms=self.vit_t, new_token_ids=self.ids)
            with self.ac():
                past = m.forward_cache_update_vit(past, **self.dev(gi))
        if gen_think:
            seed_all(seed)
            tmp, tl, tr = self.text(copy.deepcopy(past), copy.deepcopy(lens), copy.deepcopy(rope), prompt)
            think = self.think(tmp, tl, tr, do_sample, max_len)
            del tmp
        cfg_text_past, tl, tr = copy.deepcopy(past), lens, rope
        cfg_text = self.dev(m.prepare_vae_latent_cfg(curr_kvlens=tl, curr_rope=tr, image_sizes=[(h, w)]))
        cfg_img_past, cl, cr = self.cache(), [0], [0]
        for t in ([SYSTEM_PROMPT, prompt, think] if use_think else [prompt]):
            cfg_img_past, cl, cr = self.text(cfg_img_past, cl, cr, t)
        cfg_img_in = self.dev(m.prepare_vae_latent_cfg(curr_kvlens=cl, curr_rope=cr, image_sizes=[(h, w)]))
        for t in ([prompt, think] if use_think else [prompt]):
            past, lens, rope = self.text(past, lens, rope, t)
        seed_all(seed + IMAGE_SEED_OFFSET)   # the initial noise is drawn in prepare_vae_latent
        gi = self.dev(m.prepare_vae_latent(curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)],
                                           new_token_ids=self.ids))
        with self.ac():
            latent = m.generate_image(
                past_key_values=past, cfg_text_past_key_values=cfg_text_past, cfg_img_past_key_values=cfg_img_past,
                num_timesteps=e["steps"], cfg_text_scale=e["cfg_text"], cfg_img_scale=cfg_img,
                cfg_type=e["cfg_type"],
                cfg_interval=list(cfg_interval), cfg_renorm_min=0.0, cfg_renorm_type=e["renorm"],
                timestep_shift=e["shift"], **gi,
                cfg_text_packed_position_ids=cfg_text["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_text["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_text["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=cfg_text["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=cfg_img_in["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=cfg_img_in["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=cfg_img_in["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=cfg_img_in["cfg_packed_key_value_indexes"])
        return self.decode(latent, h, w), think


def _done(dest: Path, a: argparse.Namespace) -> None:
    (dest / "_done").mkdir(exist_ok=True)
    tag = f"s{a.shard}of{a.nshards}" + (f"_limit{a.limit}" if a.limit is not None else "")
    (dest / "_done" / tag).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def run_und(model, dest: Path, todo: list, a: argparse.Namespace) -> None:
    """Text answers of BAGEL (with any checkpoint overlay) to the test canvases: dest/chat_s<k>of<n>.jsonl."""
    import torch
    from .baselines import norm
    out_f = dest / f"chat_s{a.shard}of{a.nshards}.jsonl"
    done = {json.loads(x)["item_id"] for x in out_f.read_text().splitlines() if x.strip()} if out_f.exists() else set()
    started = time.time()
    with out_f.open("a") as f:
        for k, (n, it) in enumerate(todo, start=1):
            item = it["item"]
            if item["item_id"] in done:
                continue
            torch.manual_seed(42 + n)
            out = model.inf(image=it["images"]()[0], text=UND_Q, understanding_output=True, max_think_token_n=64,
                            do_sample=False)
            reply = out["text"] or ""
            ok = norm(reply.strip().rstrip("."), item["casefold"]) == norm(item["expected"], item["casefold"])
            f.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                "expected": item["expected"], "reply": reply, "correct": ok}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"und {k}/{len(todo)} {time.time() - started:.0f}s", flush=True)
    _done(dest, a)


def run_main(a: argparse.Namespace) -> None:
    from . import bagel_generate  # noqa: F401  (puts the flash-attn shim and the BAGEL source on sys.path)
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    from data.data_utils import pil_img2rgb
    benches = a.bench.split(",")
    for b in benches:
        if b not in BENCHES:
            sys.exit(f"bad bench {b} (choose from {', '.join(BENCHES)})")
    a.out.mkdir(parents=True, exist_ok=True)
    model = None
    for bench in benches:
        plain = a.plain or bench in NO_THINK
        dest = a.dest / (bench + ("_plain" if a.plain and bench not in NO_THINK else ""))
        todo = [(n, it) for n, it in enumerate(items(bench, a)) if n % a.nshards == a.shard][:a.limit]
        dest.mkdir(parents=True, exist_ok=True)
        left = [(n, it) for n, it in todo if not (dest / f"{it['key']}.png").exists()]
        print(f"{bench} shard {a.shard}/{a.nshards}: {len(todo)} items, {len(left)} to do", flush=True)
        if left and model is None:
            model = Bagel(a.ckpt)
        if left:
            info = {"bench": bench, "ckpt": str(a.ckpt) if a.ckpt else None, "plain": plain, "limit": a.limit,
                    "overlay": model.overlay}
            (dest / f"_run_s{a.shard}of{a.nshards}.json").write_text(json.dumps(info, indent=1))
        started = time.time()
        if bench == "und":
            if model is None:
                model = Bagel(a.ckpt)
            run_und(model, dest, todo, a)
            continue
        manifest = dest / "gen" / "manifest.jsonl"
        for k, (n, it) in enumerate(left, start=1):
            png = dest / f"{it['key']}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            images = [pil_img2rgb(x) for x in it["images"]()]
            image, think = model.edit(images, it["prompt"], None, not plain, 42 + n, CFG_IMG[bench],
                                      do_sample=False, max_len=1000)
            if think is not None:
                (png.parent / "think.txt").write_text(think)
            image = image.convert("RGB")
            if image.size != (1024, 1024):
                (png.parent / "resized.txt").write_text(str(image.size))
                image = image.resize((1024, 1024))
            images[0].save(png.parent / "source.png")
            tmp = png.with_name(png.stem + ".tmp.png")
            image.save(tmp)
            os.replace(tmp, png)
            item = it["item"]
            with manifest.open("a") as f:
                f.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                    "seed": 42 + n, "model": "bagel"}) + "\n")
            if k % 10 == 0 or k == len(left):
                print(f"{bench} {k}/{len(left)} {time.time() - started:.0f}s (last {time.time() - t0:.1f}s)", flush=True)
        _done(dest, a)
    print("done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="comma list of " + ", ".join(BENCHES))
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, help="gen.safetensors / model.safetensors written by bagel_train.py")
    ap.add_argument("--plain", action="store_true", help="cards without the think (the Default condition)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only the first K items of the shard (smoke tests)")
    ap.add_argument("--per-level", type=int, default=30, help="test items per level (15 levels: 450 items)")
    ap.add_argument("--tasks", nargs="*", help="tasks (default all)")
    ap.add_argument("--out", type=Path, required=True)
    run_main(ap.parse_args())


if __name__ == "__main__":
    main()
