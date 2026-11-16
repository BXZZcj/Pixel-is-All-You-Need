"""Video generators on the benchmark canvases. Image-to-video: the canvas is the first frame; does the video end with
the answer written in the answer box? The prompt is VIDEO_PROMPT, the fixed instruction made video-native (static
camera, write in the box). It never contains the answer.

models (square sizes so the canvas is not cropped; wide variants pad the canvas to the model's native 16:9 and crop the
frames back):
  wan22    Wan2.2-TI2V-5B, original repo API, max_area 960x960, 81 frames, 50 steps, cfg 5, shift 5
           (--variant f121: the official 121 frames; wide121: 121 frames at 1280x704)
  omni     HY-OmniWeaving (HunyuanVideo-1.5 DiT + Qwen2.5-VL MLLM encoder), i2v 480p bucket, 81 frames, 50 steps;
           --think: its own MLLM rewrites the prompt after looking at the canvas; --variant wide
  cosmos3  Cosmos3-Nano, diffusers Cosmos3OmniPipeline, 93 frames, 35 steps, cfg 6; --variant sq (960x960, pipeline
           prompt templates on) | sq_notpl (templates off) | sq_json (prompts from its JSON upsampler: templates off,
           official negative prompt) | wide / wide_notpl (1280x704)
  h3       MiniMax-H3 Base FL2VA (open weights), served by sglang (URL in $H3_URL); fl2va with the canvas as frame 0,
           768 short edge, 4 s, 50 steps, flow shift 12
  wan3     Wan 3.0 (closed API, OpenAI-style /v1/videos at $VIDEO_API_BASE_URL with $VIDEO_API_KEY), the canvas as the
           first frame, 5 s; --variant ext: Wan's own prompt extension (its rewritten prompt is saved as think.txt)
routes: native   the fixed prompt (Default)
        render   the correct answer is also written in the prompt; render_q: the same with the answer in double quotes
        rewrite  the prompt is a rewriter's output for this item (--rewrites jsonl from rewrite.py / cosmos_upsample.py /
                 delete_answer.py, --rewrite-field prompt (Rewrite) | prompt_noans (Delete)); the items are the ones in
                 that file
Per item: OUT/<item_id>/source.png, video.mp4, frames/f_XXXX.png (8 frames at 1/8..8/8 of the clip, mapped back to the
1024x1024 canvas layout), output.png (= last frame), think.txt, meta.json.
usage: python -m generator_benchmark.video_generate --model wan22 --route native --set t1 --out DIR [--shard 0/3]
       (omni: launch with python -m torch.distributed.run --nproc_per_node=1 -m generator_benchmark.video_generate ...)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import render, tasks
from .paths import model as model_path, src as src_path

SETS = {
    "main": [("copy_complete", "copy", 20), ("qa", "L1", 30), ("qa", "L4", 20), ("arithmetic", "A1", 20),
             ("context", "L5A", 20), ("visual", "L6A", 20), ("instruction", "I1", 20)],
    "render": [("qa", "L1", 8), ("qa", "L4", 8), ("arithmetic", "A1", 8), ("context", "L5A", 8), ("visual", "L6A", 8),
               ("instruction", "I1", 8)],
    "smoke": [("qa", "L1", 1)],
    # Table 1 on one item set (= rewrite.SETS["t1"], 7 x 60). (task, level, n, start): items [start:n] of a level in
    # frozen order; t1_rest_main / t1_rest_render are the t1 items that the main / render sets do not have
    "t1": [("qa", "L1", 60), ("qa", "L4", 60), ("arithmetic", "A1", 60), ("instruction", "I1", 60), ("context", "L5A", 60),
           ("visual", "L6A", 60), ("copy_complete", "copy", 60)],
    "t1_rest_main": [("qa", "L1", 60, 30), ("qa", "L4", 60, 20), ("arithmetic", "A1", 60, 20), ("instruction", "I1", 60, 20),
                     ("context", "L5A", 60, 20), ("visual", "L6A", 60, 20), ("copy_complete", "copy", 60, 20)],
    "t1_rest_render": [("qa", "L1", 60, 8), ("qa", "L4", 60, 8), ("arithmetic", "A1", 60, 8), ("instruction", "I1", 60, 8),
                       ("context", "L5A", 60, 8), ("visual", "L6A", 60, 8), ("copy_complete", "copy", 60, 0)],
    "t1rest": None, "t1rest_noans": None,   # the rest of every t1 level (rewrite.T1REST), set below
}
from .rewrite import T1REST, T1REST_NOANS   # noqa: E402
SETS["t1rest"], SETS["t1rest_noans"] = T1REST, T1REST_NOANS
N_SAMPLED = 8
VIDEO_PROMPT = ("Static camera. Preserve the question card and every printed character exactly. Complete the task shown "
                "on the card: in the empty answer box, write only the answer, one character at a time. Do not add any "
                "other text or decoration.")


item_seed = tasks.item_seed


def select(name: str) -> list[dict]:
    data = tasks.build_all()
    items = []
    for task, level, n, *start in SETS[name]:
        items += [i for i in data[task]["test"] if i["level"] == level][(start[0] if start else 0):n]
    return items


def to_card(frame: np.ndarray) -> Image.Image:
    """Undo the model's resize-to-cover + center-crop of the 1024x1024 card (identity up to scale for square video)."""
    img = Image.fromarray(frame)
    w, h = img.size
    scale = 1024 / max(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (1024, 1024), "white")
    canvas.paste(img, ((1024 - img.width) // 2, (1024 - img.height) // 2))
    return canvas


def pad_wide(image: Image.Image, width: int, height: int) -> Image.Image:
    """The square card, scaled to the full height, in the middle of a white width x height canvas."""
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image.resize((height, height), Image.LANCZOS), ((width - height) // 2, 0))
    return canvas


def crop_middle(video: np.ndarray) -> np.ndarray:
    """Inverse of pad_wide on (F, H, W, C) frames: keep the middle H x H square."""
    h, w = video.shape[1:3]
    return video[:, :, (w - h) // 2:(w + h) // 2]


class Wan22:
    fps = 24

    def __init__(self, variant=""):
        # f121: the official TI2V-5B clip length (generate.py default). wide121: the official setting in full - 121
        # frames at 1280x704, the only TI2V-5B training size (wan/configs SUPPORTED_SIZES); the square canvas
        # white-padded to 16:9 and the frames cropped back, as OmniWeaving's wide variant.
        self.frames = 121 if variant in ("f121", "wide121") else 81
        self.wide = variant == "wide121"
        sys.path.insert(0, src_path("Wan2.2"))
        import wan
        from wan.configs import WAN_CONFIGS
        self.cfg = WAN_CONFIGS["ti2v-5B"]
        self.model = wan.WanTI2V(config=self.cfg, checkpoint_dir=model_path("Wan2.2-TI2V-5B"), device_id=0, rank=0,
                                 t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False, convert_model_dtype=True)

    def __call__(self, prompt, source_path, seed, think):
        img = Image.open(source_path).convert("RGB")
        if self.wide:
            img = pad_wide(img, 1280, 704)
        video = self.model.generate(prompt, img=img, max_area=1280 * 704 if self.wide else 960 * 960,
                                    frame_num=self.frames, shift=self.cfg.sample_shift, sample_solver="unipc",
                                    sampling_steps=self.cfg.sample_steps, guide_scale=self.cfg.sample_guide_scale,
                                    seed=seed, offload_model=False)  # (C, N, H, W) in [-1, 1]
        video = ((video.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        return (crop_middle(video) if self.wide else video), None


class OmniWeaving:
    fps = 16  # generate.py: video_length <= 81 -> 16 fps

    def __init__(self, variant=""):
        self.wide = variant == "wide"
        src = src_path("OmniWeaving")
        sys.path[:0] = [p for p in (os.environ.get("OMNI_EXTRA_PATH"), src) if p]   # optional extra site dir
        os.chdir(src)
        spec = importlib.util.spec_from_file_location("omni_generate", f"{src}/generate.py")
        ow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ow)  # sets up the (1-process) parallel state, as generate.py does
        # take generate.py's own argument defaults by running its parser with our flags
        captured = {}
        run, destroy, argv = ow.generate_video, ow.dist.destroy_process_group, sys.argv
        ow.generate_video = lambda args: captured.setdefault("args", args)
        ow.dist.destroy_process_group = lambda: None
        sys.argv = ["generate.py", "--task", "i2v", "--model_path", model_path("HY-OmniWeaving"), "--prompt", "x",
                    "--image_path", "x.png", "--video_length", "81", "--num_inference_steps", "50",
                    "--sparse_attn", "false", "--use_sageattn", "false", "--enable_cache", "false",
                    "--offloading", "false", "--group_offloading", "false", "--overlap_group_offloading", "false"]
        try:
            ow.main()
        finally:
            ow.generate_video, ow.dist.destroy_process_group, sys.argv = run, destroy, argv
        args = self.args = captured["args"]
        import torch
        infer_state = ow.initialize_infer_state(args)
        self.pipe = ow.HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=args.model_path, transformer_dtype=torch.bfloat16,
            device=torch.device("cuda"), transformer_init_device=torch.device("cuda"),
            pipeline_config=args.pipeline_config)
        self.pipe.apply_infer_optimization(infer_state=infer_state, enable_offloading=False,
                                           enable_group_offloading=False, overlap_group_offloading=False)
        rewrite = self.pipe.activate_think_to_rewrite_prompt
        self.thought = None

        def record(*a, **k):
            self.thought = rewrite(*a, **k)
            return self.thought
        self.pipe.activate_think_to_rewrite_prompt = record

    def __call__(self, prompt, source_path, seed, think):
        self.thought = None
        a = self.args
        if self.wide:
            pad_wide(Image.open(source_path).convert("RGB"), 1820, 1024).save(source_path.with_name("source_wide.png"))
            source_path = source_path.with_name("source_wide.png")
        out = self.pipe(prompt=prompt, aspect_ratio=a.aspect_ratio, num_inference_steps=a.num_inference_steps,
                        video_length=a.video_length, negative_prompt=a.negative_prompt, seed=seed, think=think,
                        output_type="pt", only_give_text=False, deepstack=a.deepstack, setclip=a.setclip,
                        task_type="i2v", ref_image_paths=[str(source_path)])
        video = out.videos[0] if out.videos.ndim == 5 else out.videos  # (C, F, H, W) in [0, 1]
        video = (video.float().clamp(0, 1) * 255).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        return (crop_middle(video) if self.wide else video), self.thought


class Cosmos3:
    fps = 24

    def __init__(self, variant="sq"):
        self.wide, self.templates = variant.startswith("wide"), not variant.endswith("notpl")
        # sq_json: the model card usage for prompts from its own JSON prompt upsampler: resolution /
        # duration templates off, the official negative prompt (assets/negative_prompt.json, JSON-dumped as in the card)
        self.negative = ""
        if variant.endswith("json"):
            self.templates = False
            self.negative = json.dumps(json.load(open(model_path("Cosmos3-Nano") + "/assets/negative_prompt.json")))
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        self.torch = torch
        self.pipe = Cosmos3OmniPipeline.from_pretrained(model_path("Cosmos3-Nano"), torch_dtype=torch.bfloat16,
                                                        safety_checker=None, enable_safety_checker=False,
                                                        local_files_only=True)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=10.0)
        self.pipe.to("cuda")

    def __call__(self, prompt, source_path, seed, think):
        torch = self.torch
        image = Image.open(source_path).convert("RGB")
        height, width = (704, 1280) if self.wide else (960, 960)
        if self.wide:
            image = pad_wide(image, width, height)
        with torch.inference_mode():
            out = self.pipe(prompt=prompt, negative_prompt=self.negative, image=image, num_frames=93, height=height, width=width,
                            fps=self.fps, num_inference_steps=35, guidance_scale=6.0, enable_sound=False,
                            enable_safety_check=False, output_type="np", add_resolution_template=self.templates,
                            add_duration_template=self.templates,
                            generator=torch.Generator(device="cuda").manual_seed(seed))
        video = np.asarray(out.video)
        if video.ndim == 5:
            video = video[0]
        if video.dtype != np.uint8:
            video = (np.clip(video, 0, 1) * 255).round().astype(np.uint8)
        return (crop_middle(video) if self.wide else video), None


class H3:
    """Client of a running sglang MiniMax-H3 FL2VA server (URL in $H3_URL, e.g. http://localhost:30010)."""
    fps = 24

    def __init__(self, variant=""):
        import urllib.request
        self.request, self.url = urllib.request, os.environ["H3_URL"].rstrip("/")

    def _json(self, url, payload=None, timeout=120):
        data = json.dumps(payload).encode() if payload is not None else None
        req = self.request.Request(url, data=data, method="POST" if data else "GET",
                                   headers={"Content-Type": "application/json"})
        with self.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def __call__(self, prompt, source_path, seed, think):
        import imageio
        payload = {"model": "MiniMax/MiniMax-H3", "task": "fl2va", "prompt": prompt, "seconds": 4.0,
                   "conditions": [{"type": "image", "uri": Path(source_path).resolve().as_uri(), "role": "keyframe",
                                   "frame_index": 0}],
                   "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 4.0},
                   "num_outputs_per_prompt": 1, "num_inference_steps": 50, "flow_shift": 12.0, "audio_flow_shift": 3.0,
                   "seed": seed % 2**31}
        job = self._json(f"{self.url}/v1/videos", payload)["id"]
        deadline = time.time() + 3600
        while True:
            state = self._json(f"{self.url}/v1/videos/{job}").get("status")
            if state == "completed":
                break
            if state in ("failed", "cancelled") or time.time() > deadline:
                raise RuntimeError(f"H3 job {job}: {state}")
            time.sleep(5)
        path = Path(source_path).with_name("h3_raw.mp4")
        with self.request.urlopen(f"{self.url}/v1/videos/{job}/content", timeout=3600) as r, path.open("wb") as f:
            while chunk := r.read(8 << 20):
                f.write(chunk)
        return np.stack([np.asarray(f)[..., :3] for f in imageio.get_reader(path)]), None


class ApiVideo(H3):
    """A closed video model behind an OpenAI-style /v1/videos gateway ($VIDEO_API_BASE_URL, key $VIDEO_API_KEY)."""
    fps = 24

    def __init__(self, variant=""):
        import urllib.request
        self.request, self.url = urllib.request, os.environ["VIDEO_API_BASE_URL"].rstrip("/") + "/v1"
        self.auth = {"Authorization": "Bearer " + os.environ["VIDEO_API_KEY"]}

    def _json(self, url, payload=None, timeout=120):
        data = json.dumps(payload).encode() if payload is not None else None
        req = self.request.Request(url, data=data, method="POST" if data else "GET",
                                   headers={"Content-Type": "application/json", **self.auth})
        for attempt in range(5):
            try:
                with self.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except Exception:
                if attempt == 4 or data:  # never resubmit a paid generation
                    raise
                time.sleep(10)


class Wan3(ApiVideo):
    """Wan 3.0 (closed) through an OpenAI-style /v1/videos gateway, the canvas as the first frame (data URL), 5 s.
    variant: 'ext' turns on Wan's own prompt extension
    (prompt_extend=true; its rewritten prompt, when the API returns it, is saved as think.txt), 'prime' uses
    wan3.0-video-prime. The task record returned by the API is kept next to the video (wan3_task.json)."""
    fps = 24

    def __init__(self, variant=""):
        super().__init__(variant)
        self.model = "wan3.0-video-prime" if "prime" in variant else "wan3.0-video"
        self.extend = "ext" in variant

    def __call__(self, prompt, source_path, seed, think):
        import base64
        import imageio
        image = "data:image/png;base64," + base64.b64encode(Path(source_path).read_bytes()).decode()
        payload = {"model": self.model, "prompt": prompt, "image": image, "seconds": "5", "duration": 5,
                   "metadata": {"prompt_extend": self.extend, "seed": seed % 2**31, "watermark": False}}
        job = self._json(f"{self.url}/videos", payload)["id"]
        deadline = time.time() + 3600
        while True:
            time.sleep(10)
            info = self._json(f"{self.url}/videos/{job}")
            if info.get("status") == "completed":
                break
            if info.get("status") in ("failed", "cancelled") or time.time() > deadline:
                raise RuntimeError(f"Wan3 job {job}: {str(info)[:600]}")
        Path(source_path).with_name("wan3_task.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        path = Path(source_path).with_name("wan3_raw.mp4")
        req = self.request.Request(f"{self.url}/videos/{job}/content", headers=self.auth)
        with self.request.urlopen(req, timeout=600) as r, path.open("wb") as f:
            while chunk := r.read(8 << 20):
                f.write(chunk)
        extended = None
        for key in ("actual_prompt", "extended_prompt", "revised_prompt"):
            stack = [info]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    if isinstance(x.get(key), str):
                        extended = x[key]
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
        return np.stack([np.asarray(f)[..., :3] for f in imageio.get_reader(path)]), extended


MODELS = {"wan22": Wan22, "omni": OmniWeaving, "cosmos3": Cosmos3, "h3": H3, "wan3": Wan3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--route", choices=("native", "render", "render_q", "rewrite"), default="native")
    ap.add_argument("--rewrites", type=Path, nargs="*", default=[], help="rewrite rows (rewrite.py format) for --route rewrite")
    ap.add_argument("--rewrite-field", default="prompt", choices=("prompt", "prompt_noans"))
    ap.add_argument("--set", choices=sorted(SETS), default="t1")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ids", default="", help="comma-separated item ids: keep only these (same items across routes)")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--variant", default="", help="model-specific setting (see the module docstring)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    k, n = map(int, a.shard.split("/"))
    items = select(a.set)
    rewritten = {}
    if a.route == "rewrite":
        for f in a.rewrites:
            for line in f.read_text().splitlines():
                row = json.loads(line)
                if row.get(a.rewrite_field):
                    rewritten[row["item_id"]] = row[a.rewrite_field]
        index = {i["item_id"]: i for d in tasks.build_all().values() for i in d["test"]}
        items = [index[i] for i in rewritten]
    if a.ids:
        keep = set(a.ids.split(","))
        items = [i for i in items if i["item_id"] in keep]
    # items round-robin over (task, level), each level in its frozen order, so a stopped or partial run covers all
    # levels evenly instead of finishing one first
    groups = {}
    for i in items:
        groups.setdefault((i["task"], i["level"]), []).append(i)
    items = [g[r] for r in range(max((len(g) for g in groups.values()), default=0)) for g in groups.values() if r < len(g)]
    items = items[k::n]
    todo = [i for i in items if not (out / i["item_id"] / "output.png").exists()]
    print(f"{len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        return
    import imageio
    model = MODELS[a.model](a.variant) if a.variant else MODELS[a.model]()
    with (out / "manifest.jsonl").open("a") as manifest:
        for m, item in enumerate(todo, start=1):
            d = out / item["item_id"]
            (d / "frames").mkdir(parents=True, exist_ok=True)
            prompt = rewritten[item["item_id"]] if a.route == "rewrite" else VIDEO_PROMPT
            if a.route in ("render", "render_q"):   # render_q: quoted, as some generators need (LongCat, BAGEL)
                q = '"' if a.route == "render_q" else ""
                prompt += tasks.RENDER_INSTRUCTION.format(answer=f'{q}{item["target"]}{q}')
            render.render_input(item).save(d / "source.png")
            started = time.time()
            video, thought = model(prompt, d / "source.png", item_seed(item["item_id"]), a.think)
            seconds = time.time() - started
            imageio.mimwrite(d / "video.mp4", list(video), fps=model.fps, macro_block_size=1)
            count = len(video)
            picks = [round((j + 1) * (count - 1) / N_SAMPLED) for j in range(N_SAMPLED)]
            for j in picks:
                to_card(video[j]).save(d / "frames" / f"f_{j:04d}.png")
            if thought is not None:
                (d / "think.txt").write_text(str(thought))
            meta = {"item_id": item["item_id"], "task": item["task"], "level": item["level"], "model": a.model,
                    "route": a.route, "rewrite_field": a.rewrite_field if a.route == "rewrite" else None,
                    "think": a.think, "variant": a.variant, "prompt": prompt, "seed": item_seed(item["item_id"]),
                    "frames": count, "size": list(video.shape[1:3]), "sampled": picks, "seconds": round(seconds, 1)}
            (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            to_card(video[-1]).save(d / "output.png")
            manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"{m}/{len(todo)} {item['item_id']} {seconds:.0f}s frames={count} size={video.shape[1:3]}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
