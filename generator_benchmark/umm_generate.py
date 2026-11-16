"""Unified multimodal models on the benchmark canvases (Table 1: Default, Render, Think).

modes:
  native  canvas + the fixed instruction, image output (Default)
  render  the correct answer is also written in the prompt: can the model draw a given answer? (Render)
  think   the model's own reasoning-before-generation mode (Think); the reasoning text is saved as think.txt
          InternVL-U: generation_mode="text_image"; SenseNova: think_mode of it2i_generate; Emu3.5: see Emu35Full
models: internvlu (InternVL-U 4B), sensenova (SenseNova-U1.5-8B-MoT), emu35 (Emu3.5-Image 34B: native, render),
        emu35full (Emu3.5 34B: think)
Each model runs in the environment of its upstream repository (UMM_SRC_DIR/<repo>, weights in UMM_MODEL_DIR).
usage: python -m generator_benchmark.umm_generate --model internvlu --mode native --set t1 --out DIR [--shard 0/2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import render, tasks
from .paths import model as model_path, src as src_path

KEY = [("qa", "L1", 60), ("qa", "L4", 60), ("arithmetic", "A1", 60), ("visual", "L6A", 60), ("copy_complete", "copy", 60)]
# Table 1 on one item set: the key set plus instruction I1 and context L5A (t1x: only the two added levels)
T1X = [("instruction", "I1", 60), ("context", "L5A", 60)]
T1 = KEY[:3] + T1X + KEY[3:]
UND_PROMPT = "Answer the question on the card. Answer with only the answer, in as few words as possible."


item_seed = tasks.item_seed


def select(name: str = "t1") -> list[dict]:
    """t1: Table 1 (7 levels x 60); key / t1x: its two parts; t1rest: the rest of each Table 1 level; closed60: 60 items
    of all 15 test levels."""
    from .rewrite import SETS, T1REST
    data = tasks.build_all()
    items = []
    for task, level, n, *start in {"key": KEY, "t1": T1, "t1x": T1X, "t1rest": T1REST}.get(name) or SETS["closed60"]:
        items += [i for i in data[task]["test"] if i["level"] == level][(start[0] if start else 0):n]
    return items


class InternVLU:
    def __init__(self):
        import sys
        import torch
        sys.path.insert(0, src_path("InternVL-U"))
        from internvlu import InternVLUPipeline
        self.torch = torch
        self.pipe = InternVLUPipeline.from_pretrained(model_path("InternVL-U"), torch_dtype=torch.bfloat16).to("cuda")
        self.tok = self.pipe.processor.tokenizer

    def image(self, image, prompt: str, seed: int, think: bool):
        with self.torch.no_grad():
            out = self.pipe(prompt=prompt, image=image, generation_mode="text_image" if think else "image",
                            height=1024, width=1024, generator=self.torch.Generator(device="cuda").manual_seed(seed))
        thought = self.tok.decode(out.generate_output[0], skip_special_tokens=True).strip() if think else None
        return out.images[0], thought


class SenseNova:
    """SenseNova-U1.5-8B-MoT (NEO-Unify, flow-matching image head on the shared Qwen3 MoT). Settings follow
    examples/editing/inference.py + examples/README.md: cfg_scale 4.0, img_cfg_scale 1.0, cfg_norm none,
    timestep_shift 3.0, cfg_interval (0, 1), 50 steps; think=True is the built-in think_mode of it2i_generate
    (<think>...</think> decoded before the image, no external API). The cards are fed at their native 1024x1024 and
    generated at 1024x1024 (the script default would upscale both to 2048x2048; 1024 is within the model's
    min/max_pixels and 4x cheaper)."""

    def __init__(self):
        import sys
        import torch
        sys.path.insert(0, src_path("SenseNova-U1") + "/src")
        import sensenova_u1
        from sensenova_u1.models.neo_unify.utils import load_image_native
        from sensenova_u1.utils import load_model_and_tokenizer
        sensenova_u1.set_attn_backend("auto")   # flash-attn if importable, else SDPA
        self.torch, self.load_image_native = torch, load_image_native
        self.model, self.tok = load_model_and_tokenizer(model_path("SenseNova-U1.5-8B-MoT"), dtype=torch.bfloat16,
                                                        device="cuda")

    def image(self, image, prompt: str, seed: int, think: bool):
        import numpy as np
        from PIL import Image
        with self.torch.inference_mode():
            out = self.model.it2i_generate(self.tok, prompt, [image.convert("RGB")], image_size=(1024, 1024),
                                           cfg_scale=4.0, img_cfg_scale=1.0, cfg_norm="none", timestep_shift=3.0,
                                           cfg_interval=(0.0, 1.0), num_steps=50, batch_size=1, think_mode=think,
                                           seed=seed)
        x, thought = (out[0], out[1]) if think else (out, None)
        x = ((x[:1].float() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round()
        if thought is not None:
            thought = thought.replace("<think>", "").replace("</think>", "").replace("<img>", "").strip()
        return Image.fromarray(x[0].astype(np.uint8)), thought


class Emu35:
    """Emu3.5-Image (34B autoregressive next-token model, IBQ vision tokens). X2I setting of
    configs/example_config_x2i.py (the preset gradio_demo_image.py uses for x2i): task template 'x2i',
    classifier_free_guidance 3.0, unconditional_type no_text, image_area 1048576 (64x64 tokens = 1024x1024),
    max_new_tokens 5120, differential sampling (text top_k 1024/top_p 0.9, image top_k 5120), and the generation path
    of inference.py (generate/multimodal_decode). The reference image is a single-element list as the README
    recommends for x2i. SDPA attention instead of flash_attention_2 (flash-attn not installed). No separate reasoning
    mode exists for Emu3.5-Image, so think is unsupported (see Emu35Full); any text the model emits before the image goes
    to the log."""

    def __init__(self, path: str | None = None):
        import sys
        import types
        import torch
        from transformers import AutoTokenizer
        sys.path.insert(0, src_path("Emu3.5"))
        from src.emu3p5 import Emu3Config, Emu3ForCausalLM
        from src.utils.generation_utils import generate, multimodal_decode
        from src.utils.input_utils import build_image
        from src.vision_tokenizer import build_vision_tokenizer
        self.torch, self.generate, self.decode, self.build_image = torch, generate, multimodal_decode, build_image
        path = path or model_path("Emu3.5-Image")
        self.model = Emu3ForCausalLM.from_pretrained(path, config=Emu3Config.from_pretrained(path, trust_remote_code=True),
                                                     torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                     attn_implementation="flash_attention_2").eval()  # as inference.py
        tdir = src_path("Emu3.5") + "/src/tokenizer_emu3_ibq"
        tok = AutoTokenizer.from_pretrained(tdir, special_tokens_file=f"{tdir}/emu3_vision_tokens.txt",
                                            trust_remote_code=True)
        special = dict(BOS="<|extra_203|>", EOS="<|extra_204|>", PAD="<|endoftext|>", EOL="<|extra_200|>",
                       EOF="<|extra_201|>", TMS="<|extra_202|>", IMG="<|image token|>", BOI="<|image start|>",
                       EOI="<|image end|>", BSS="<|extra_100|>", ESS="<|extra_101|>", BOG="<|extra_60|>",
                       EOG="<|extra_61|>", BOC="<|extra_50|>", EOC="<|extra_51|>")
        for k, v in special.items():
            setattr(tok, k.lower() + "_token", v)
        self.tok = tok
        self.vq = build_vision_tokenizer("ibq", model_path("Emu3.5-VisionTokenizer"), device="cuda:0")
        sampling = dict(use_cache=True, text_top_k=1024, text_top_p=0.9, text_temperature=1.0, image_top_k=5120,
                        image_top_p=1.0, image_temperature=1.0, top_k=131072, top_p=1.0, temperature=1.0,
                        num_beams_per_group=1, num_beam_groups=1, diversity_penalty=0.0, max_new_tokens=5120,
                        guidance_scale=1.0, use_differential_sampling=True, do_sample=True, num_beams=1)
        self.cfg = types.SimpleNamespace(
            streaming=False, unconditional_type="no_text", classifier_free_guidance=3.0, image_area=1048576,
            sampling_params=sampling, special_token_ids={k: tok.encode(v)[0] for k, v in special.items()})
        self.unc = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        self.tmpl = "<|extra_203|>You are a helpful assistant for x2i task. USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>"

    def _ids(self, s: str):
        ids = self.tok.encode(s, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        bos = self.cfg.special_token_ids["BOS"]
        if ids[0, 0] != bos:
            ids = self.torch.cat([self.torch.tensor([[bos]], device=ids.device, dtype=ids.dtype), ids], dim=1)
        return ids

    def image(self, image, prompt: str, seed: int, think: bool):
        if think:
            raise NotImplementedError("Emu3.5-Image has no reasoning-before-generation mode")
        import random
        random.seed(seed)
        self.torch.manual_seed(seed)
        img = self.build_image(image.convert("RGB"), self.cfg, self.tok, self.vq)
        ids = self._ids(self.tmpl.format(question=prompt).replace("<|IMAGE|>", img))
        unc = self.tok.encode(self.unc.replace("<|IMAGE|>", img), return_tensors="pt",
                              add_special_tokens=False).to(self.model.device)
        self.torch.cuda.empty_cache()
        out, texts = None, []
        for tokens in self.generate(self.cfg, self.model, self.tok, ids, unc, None, True):
            for kind, payload in self.decode(self.tok.decode(tokens, skip_special_tokens=False), self.tok, self.vq):
                if kind == "image" and out is None:
                    out = payload
                elif kind != "image":
                    texts.append(f"[{kind}] {payload}")
        if texts:
            print("emu35 text:", " | ".join(texts)[:2000], flush=True)
        if out is None:
            raise RuntimeError("Emu3.5 produced no complete image: " + " | ".join(texts)[:500])
        return out, None


class Emu35Full(Emu35):
    """The full Emu3.5 (Emu3.5-Image, the X2I variant, has no text-first mode). Its official text-capable mode is the
    visual-guidance task of
    configs/example_config_visual_guidance.py (task_type howto: interleaved text and images): its template with the card
    as input image, image_area 518400, CFG 3.0 on visual tokens only, text tokens sampled top_k 200 / top_p 0.8 /
    temperature 0.7 by the official logits processor (generation_utils.non_streaming_generate).
    text(): the same generation, stopped where the model starts its first image (eos = EOS or image start): the reply is
    the text it writes first (global / image CoT segments included).
    image(think=True): the thought is its own text answer (text() with the understanding prompt, stopped at the image
    start); the image is the first complete image of a full howto generation with the same seed. The thought is saved
    as think.txt together with any text that generation writes before its image; it is not inserted into the context
    of the image generation."""

    def __init__(self):
        import types
        super().__init__(path=model_path("Emu3.5"))
        sp = dict(self.cfg.sampling_params, text_top_k=200, text_top_p=0.8, text_temperature=0.7, image_top_k=10240,
                  max_new_tokens=8192)
        self.cfg = types.SimpleNamespace(**{**vars(self.cfg), "image_area": 518400, "sampling_params": sp})
        self.unc = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        self.tmpl = ("<|extra_203|>You are a helpful assistant for howto task. Please generate a response with interleaved "
                     "text and images. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>")

    def _run(self, image, prompt: str, seed: int, stop_at_image: bool):
        import random
        from src.utils.generation_utils import non_streaming_generate
        random.seed(seed)
        self.torch.manual_seed(seed)
        img = self.build_image(image.convert("RGB"), self.cfg, self.tok, self.vq)
        ids = self._ids(self.tmpl.format(question=prompt).replace("<|IMAGE|>", img))
        unc = self.tok.encode(self.unc.replace("<|IMAGE|>", img), return_tensors="pt", add_special_tokens=False).to(self.model.device)
        cfg = self.cfg
        if stop_at_image:
            sid = self.cfg.special_token_ids
            cfg = types_ns(cfg, sid)
        self.torch.cuda.empty_cache()
        out = non_streaming_generate(cfg, self.model, self.tok, ids, unc, None, True)
        return self.tok.decode(out, skip_special_tokens=False)

    def text(self, image, prompt: str, max_new_tokens=1024) -> str:
        raw = self._run(image, prompt, 0, True)
        parts = [p for k, p in self.decode(raw, self.tok, self.vq) if k != "image"]
        s = " ".join(parts)
        for t in ("<|image start|>", "<|extra_101|>", "<|extra_204|>", "<|endoftext|>"):
            s = s.replace(t, "")
        return s.strip()

    def image(self, image, prompt: str, seed: int, think: bool):
        # left to itself the howto generation opens its image at once, with no text before it; the thought is therefore
        # its own text answer, generated first (see the class docstring)
        thought = self.text(image, UND_PROMPT) if think else ""
        raw = self._run(image, prompt, seed, False)
        out, texts = None, ([thought] if thought else [])
        for kind, payload in self.decode(raw, self.tok, self.vq):
            if kind == "image" and out is None:
                out = payload
            elif kind != "image" and out is None:
                texts.append(payload.strip())
        if out is None:
            raise RuntimeError("Emu3.5 produced no complete image: " + " | ".join(texts)[:500])
        return out, (" ".join(texts).strip() if think else None)


def types_ns(cfg, sid):
    """a copy of the generation config whose end tokens also include the image start (text-only replies)"""
    import types
    sp = dict(cfg.sampling_params)
    c = types.SimpleNamespace(**vars(cfg))
    c.sampling_params = sp
    c.special_token_ids = dict(sid)
    c.special_token_ids["EOS"] = [sid["EOS"], sid["BOI"]]
    return c


MODELS = {"internvlu": InternVLU, "sensenova": SenseNova, "emu35": Emu35, "emu35full": Emu35Full}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--mode", choices=("native", "render", "think"), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="smoke runs: first N items only")
    ap.add_argument("--set", choices=("t1", "key", "t1x", "t1rest", "closed60"), default="t1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    items = select(a.set)[k::n]
    if a.limit:
        items = items[::max(1, len(items) // a.limit)][:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    model = MODELS[a.model]()
    started = time.time()
    gen = a.out / "gen"
    gen.mkdir(exist_ok=True)
    with (gen / "manifest.jsonl").open("a") as manifest:
        for m, item in enumerate(items, start=1):
            d = gen / item["item_id"]
            if (d / "output.png").exists():
                continue
            d.mkdir(exist_ok=True)
            source = render.render_input(item)
            source.save(d / "source.png")
            prompt = tasks.GENERIC_PROMPT
            if a.mode == "render":
                prompt += tasks.RENDER_INSTRUCTION.format(answer=item["target"])
            image, thought = model.image(source, prompt, item_seed(item["item_id"]), a.mode == "think")
            image = image.convert("RGB")
            if image.size != (1024, 1024):
                (d / "resized.txt").write_text(str(image.size))
                image = image.resize((1024, 1024))
            image.save(d / "output.png")
            if thought is not None:
                (d / "think.txt").write_text(thought)
            manifest.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                       "model": a.model, "mode": a.mode, "prompt": prompt},
                                      ensure_ascii=False) + "\n")
            manifest.flush()
            if m % 20 == 0:
                print(f"{m}/{len(items)} {time.time() - started:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
