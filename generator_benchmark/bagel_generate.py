"""BAGEL-7B-MoT on the benchmark canvases (Table 1: Default, Render, Think). Runs in BAGEL's own environment.

modes: native - canvas + the fixed instruction, edit without thinking (Default)
       render - the correct answer is also written in the prompt (Render)
       think  - BAGEL's own <think> planning text generated before the image (its official think mode; Think)
       und    - understanding only: BAGEL answers the canvas as a VLM (what its text module knows)
BAGEL_SRC: the BAGEL repository (default UMM_SRC_DIR/Bagel); BAGEL_WEIGHTS: BAGEL-7B-MoT (default UMM_MODEL_DIR/BAGEL-7B-MoT).
usage: python -m generator_benchmark.bagel_generate --mode think --out DIR [--per-level 100] [--tasks qa context ...] [--levels L1 A1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .paths import model as _model, src as _src

sys.path.insert(0, str(Path(__file__).resolve().parent / "shims"))   # flash_attn stand-in for inference
BAGEL_SRC = os.environ.get("BAGEL_SRC", _src("Bagel"))
BAGEL_W = os.environ.get("BAGEL_WEIGHTS", _model("BAGEL-7B-MoT"))
sys.path.insert(0, BAGEL_SRC)

from . import render, tasks  # noqa: E402
from .api_generate import LEVELS  # noqa: E402
from .baselines import norm  # noqa: E402
item_seed = tasks.item_seed

HYPER = dict(cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0], timestep_shift=3.0, num_timesteps=50,
             cfg_renorm_min=0.0, cfg_renorm_type="text_channel")  # BAGEL's recommended editing settings


def load():
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    llm_config = Qwen2Config.from_json_file(os.path.join(BAGEL_W, "llm_config.json"))
    llm_config.qk_norm, llm_config.tie_word_embeddings, llm_config.layer_module = True, False, "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(BAGEL_W, "vit_config.json"))
    vit_config.rope, vit_config.num_hidden_layers = False, vit_config.num_hidden_layers - 1
    vae_model, vae_config = load_ae(local_path=os.path.join(BAGEL_W, "ae.safetensors"))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
                         vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
                         latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(BAGEL_W)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(BAGEL_W, "ema.safetensors"),
                                         device_map={"": "cuda:0"}, dtype=torch.bfloat16, force_hooks=True).eval()
    # ae.safetensors is float32. BAGEL's notebook keeps the VAE on the CPU and hands it CPU pixels (the LLM parts are
    # moved by accelerate hooks); we run it on the GPU in float32 and move its inputs there.
    vae_model = vae_model.to("cuda:0", torch.float32).eval()
    for name in ("encode", "decode"):
        fn = getattr(vae_model, name)
        setattr(vae_model, name, lambda x, _fn=fn: _fn(x.to("cuda:0", torch.float32)))
    return InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer,
                                vae_transform=ImageTransform(1024, 512, 16), vit_transform=ImageTransform(980, 224, 14),
                                new_token_ids=new_token_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("native", "think", "und", "render"), required=True)
    ap.add_argument("--levels", nargs="*", help="only these levels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-level", type=int, default=100)
    ap.add_argument("--tasks", nargs="*", default=sorted(LEVELS))
    a = ap.parse_args()
    import torch
    data = tasks.build_all()
    items = []
    for task in a.tasks:
        for level in LEVELS[task]:
            if a.levels and level not in a.levels:
                continue
            items += [i for i in data[task]["test"] if i["level"] == level][:a.per_level]
    inferencer = load()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if a.mode == "und":
        with (a.out / "chat.jsonl").open("w") as f:
            for item in items:
                torch.manual_seed(item_seed(item["item_id"]))
                out = inferencer(image=render.render_input(item), text="Answer the question on the card. Answer with "
                                 "only the answer, in as few words as possible.", understanding_output=True,
                                 max_think_token_n=64, do_sample=False)
                reply = out["text"] or ""
                f.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                    "expected": item["expected"], "reply": reply,
                                    "correct": norm(reply.strip().rstrip("."), item["casefold"]) == norm(item["expected"], item["casefold"])},
                                   ensure_ascii=False) + "\n")
                f.flush()
        print("done", flush=True)
        return
    gen = a.out / "gen"
    gen.mkdir(exist_ok=True)
    with (gen / "manifest.jsonl").open("a") as manifest:
        for n, item in enumerate(items, start=1):
            d = gen / item["item_id"]
            if (d / "output.png").exists():
                continue
            d.mkdir(exist_ok=True)
            source = render.render_input(item)
            source.save(d / "source.png")
            torch.manual_seed(item_seed(item["item_id"]))
            prompt = tasks.GENERIC_PROMPT
            if a.mode == "render":
                prompt += tasks.RENDER_INSTRUCTION.format(answer=item["target"])
            out = inferencer(image=source, text=prompt, think=a.mode == "think", max_think_token_n=1000,
                             do_sample=False, **HYPER)
            image = out["image"].convert("RGB")
            if image.size != (1024, 1024):
                image = image.resize((1024, 1024))  # BAGEL keeps the aspect ratio; record the resize
                (d / "resized.txt").write_text(str(out["image"].size))
            image.save(d / "output.png")
            if out.get("text"):
                (d / "think.txt").write_text(out["text"])
            manifest.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "level": item["level"],
                                       "seed": item_seed(item["item_id"]), "model": f"bagel_{a.mode}"}) + "\n")
            manifest.flush()
            if n % 20 == 0:
                print(f"{n}/{len(items)} {time.time() - started:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
