"""Fine-tune BAGEL on the benchmark canvases: does training the generation module make it answer on the canvas?

Arms (same canvases, same replay data, same number of updates):
  pixel   [canvas][instruction]                                  -> answered canvas (the answer must come from pixels)
  render  [canvas][instruction + 'Write this answer in the answer box: "S"'] -> the canvas with S written in
          (S is another training answer or a random string: the arm teaches writing a given answer, not solving)
By default only the generation expert is trained (--freeze_und True: und-token states are detached in every layer, so
the understanding expert, ViT path and text embeddings get zero gradient; AdamW has weight_decay 0, so they do not
move). --train-und also trains the understanding expert (both experts; ViT and connector stay frozen).
The pixel arm uses only training canvases whose greedy BAGEL think states the answer (bagel_cards.py think), i.e. items
the text module can answer, and from whose think every sentence stating the answer can be removed.

Training is BAGEL's own train/pretrain_unified_navit.py, run with these patches: no wandb, no EMA copy, gen-only data
without CE tokens is allowed with visual_und, the connector frozen, a dummy ViT pass for batches without ViT tokens
(keeps the FSDP collectives of all ranks in step), an optional start checkpoint (--init) and a slim checkpoint (bf16
weights only; --save gen keeps just the generation-side tensors). Replay: base BAGEL's own outputs (bagel_replay.py),
mixed 50 / 25 / 25 with the canvases (canvases / replay t2i / replay edit).

usage: python -m generator_benchmark.bagel_train prepare --arm pixel|render --replay REPLAY_DIR --thinks 'DIR/s*/think.jsonl' --out DIR
       python -m generator_benchmark.bagel_train datatest --out DIR [--batches 4]     (CPU: pack a few batches, dump one)
       torchrun --nproc_per_node 4 -m generator_benchmark.bagel_train train --out DIR --steps 2400 --save-every 800
           [--lr 2e-5] [--train-und] [--init CKPT] [--save gen|full]
       python -m generator_benchmark.bagel_train check --ckpt DIR/ckpt/0002400/gen.safetensors  (which tensors moved)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import types
from pathlib import Path

from .paths import model as model_path, src as src_path

REPO = Path(__file__).resolve().parent
BAGEL_SRC = os.environ.get("BAGEL_SRC", src_path("Bagel"))
BAGEL_W = os.environ.get("BAGEL_WEIGHTS", model_path("BAGEL-7B-MoT"))
ARMS = ("pixel", "render")
TEXT_DROP = 0.1   # prompt (+ think) dropped together = BAGEL's cfg_text context [system][card]
MIX_CARDS = 0.5   # the canvases' share; the replay splits the rest between t2i and edit
VAE_T = {"image_stride": 16, "max_image_size": 1024, "min_image_size": 512}
CARD_VIT_T = {"image_stride": 14, "max_image_size": 980, "min_image_size": 224}   # = bagel_generate inference
GEN_KEYS = ("moe_gen", "time_embedder.", "vae2llm.", "llm2vae.")   # generation-side tensors (--save gen)
SAVE_MODE = ["gen"]   # set by train_main from --save


def bagel_path() -> None:
    for p in (BAGEL_SRC, str(REPO / "shims")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # BAGEL's data/video_utils imports decord at module level; no video data is used here, so a stub is enough.
    import importlib.machinery
    import importlib.util
    if "decord" in sys.modules or importlib.util.find_spec("decord") is not None:
        return
    import transformers  # noqa: F401  (transformers records decord as unavailable before the stub exists)

    class VideoReader:
        def __init__(self, *_, **__):
            raise RuntimeError("decord is not installed (no video data is used)")
    stub = types.ModuleType("decord")
    stub.__spec__ = importlib.machinery.ModuleSpec("decord", None)
    stub.VideoReader, stub.video_reader = VideoReader, types.SimpleNamespace(VideoReader=VideoReader)
    sys.modules["decord"] = stub


def is_gen_key(name: str) -> bool:
    return any(k in name for k in GEN_KEYS)


# ----------------------------------------------------------------------------------------------------------- prepare
def prepare_main(a: argparse.Namespace) -> None:
    from .bagel_cards import drop_answer, states, train_items
    if a.arm == "render":   # every write canvas of the training mix; the string to draw is random (no thinks needed)
        rows = [{"item_id": it["item_id"], "task": it["task"], "level": it["level"]}
                for it in train_items() if it["mode"] == "write"]
        return _write_prepare(a, rows, {"render_items": len(rows)})
    if not a.thinks:
        sys.exit("the pixel arm needs --thinks (the think.jsonl shards of bagel_cards.py think)")
    files = sorted(glob.glob(a.thinks))
    if len(files) < a.min_shards:
        sys.exit(f"only {len(files)} think shards (need {a.min_shards})")
    rows, seen = [], set()
    stats = {"shards": len(files), "thinks": 0, "states_answer": 0, "noans_clean": 0}
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["item_id"] in seen:
                continue
            seen.add(r["item_id"])
            stats["thinks"] += 1
            if not r["states_answer"]:
                continue
            stats["states_answer"] += 1
            if states(drop_answer(r["think"], r["expected"]), r["expected"]):
                continue   # the answer cannot be cleanly removed from this think: the item is left out
            stats["noans_clean"] += 1
            rows.append({"item_id": r["item_id"], "task": r["task"], "level": r["level"]})
    _write_prepare(a, rows, stats)


def _write_prepare(a: argparse.Namespace, rows: list[dict], stats: dict) -> None:
    rows.sort(key=lambda r: r["item_id"])
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "cards.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = Path(a.replay) / "replay.jsonl"
    if not (Path(a.replay) / "_done" / "complete").exists():
        sys.exit(f"replay {a.replay} not packed (no _done/complete; run bagel_replay.py pack)")
    kinds = {}
    for line in manifest.read_text().splitlines():
        r = json.loads(line)
        kinds[f"{r['kind']}_{r['mode']}"] = kinds.get(f"{r['kind']}_{r['mode']}", 0) + 1
    stats["replay"] = {"dir": str(a.replay), "samples": kinds}
    yaml = {"cards": {"dataset_names": ["cards"], "image_transform_args": VAE_T, "vit_image_transform_args": CARD_VIT_T,
                      "is_mandatory": False, "weight": MIX_CARDS, "manifest": str(a.out / "cards.jsonl"), "arm": a.arm,
                      "text_drop": TEXT_DROP},
            "replay_t2i": {"dataset_names": ["replay"], "image_transform_args": VAE_T, "is_mandatory": False,
                           "weight": (1 - MIX_CARDS) / 2, "manifest": str(manifest), "kind": "t2i",
                           "text_drop": TEXT_DROP},
            "replay_edit": {"dataset_names": ["replay"], "image_transform_args": VAE_T,
                            "vit_image_transform_args": CARD_VIT_T, "is_mandatory": False,
                            "weight": (1 - MIX_CARDS) / 2, "manifest": str(manifest), "kind": "edit",
                            "text_drop": TEXT_DROP}}
    (a.out / "data.yaml").write_text(json.dumps(yaml, indent=1))   # JSON is valid YAML
    stats["arm"] = a.arm
    stats["by_task"] = {}
    for r in rows:
        stats["by_task"][r["task"]] = stats["by_task"].get(r["task"], 0) + 1
    (a.out / "prepare.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


# ----------------------------------------------------------------------------------------------------------- dataset
def register(out: Path) -> None:
    """Add the canvas and replay datasets to BAGEL's dataset registry."""
    bagel_path()
    from data import dataset_info
    from data.interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset
    from data.data_utils import pil_img2rgb
    from inferencer import GEN_THINK_SYSTEM_PROMPT

    from . import render, tasks
    from .bagel_cards import train_items

    class CardEditDataset(InterleavedBaseIterableDataset):
        def __init__(self, dataset_name, transform, tokenizer, vit_transform, manifest, arm, text_drop=TEXT_DROP,
                     local_rank=0, world_size=1, num_workers=8, data_status=None, **_):
            super().__init__(dataset_name, local_rank, world_size, num_workers)
            assert arm in ARMS, arm
            self.transform, self.vit_transform, self.tokenizer = transform, vit_transform, tokenizer
            self.arm, self.text_drop = arm, text_drop
            items = {it["item_id"]: it for it in train_items()}
            # render arm: the string to draw is another training answer (half) or random letters and digits (half)
            self.pool = sorted({it["target"] for it in items.values() if it["mode"] == "write" and it["target"]})
            self.rows = {}
            for line in Path(manifest).read_text().splitlines():
                r = json.loads(line)
                self.rows[r["item_id"]] = (items[r["item_id"]], r)
            self.data_paths = sorted(self.rows)
            self.set_epoch()

        def build(self, item: dict, row: dict) -> dict:
            if self.arm == "render":   # [canvas][prompt + 'Write this answer in the answer box: "S"'] -> canvas with S
                import string
                if random.random() < 0.5:
                    s = random.choice(self.pool)
                else:
                    s = "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(3, 8)))
                data = self._init_data()
                data = self._add_image(data, pil_img2rgb(render.render_input(item)), need_loss=False, need_vae=True,
                                       need_vit=True)
                if random.random() >= self.text_drop:
                    prompt = tasks.GENERIC_PROMPT + tasks.RENDER_INSTRUCTION.format(answer=f'"{s}"')
                    data = self._add_text(data, prompt, need_loss=False, enable_cfg=False)
                return self._add_image(data, pil_img2rgb(render.render_target({**item, "target": s})), need_loss=True,
                                       need_vae=False, need_vit=False)
            data = self._init_data()   # pixel: [canvas][prompt] -> answered canvas
            data = self._add_image(data, pil_img2rgb(render.render_input(item)), need_loss=False, need_vae=True,
                                   need_vit=True)
            if random.random() >= self.text_drop:
                data = self._add_text(data, tasks.GENERIC_PROMPT, need_loss=False, enable_cfg=False)
            return self._add_image(data, pil_img2rgb(render.render_target(item)), need_loss=True, need_vae=False,
                                   need_vit=False)

        def __iter__(self):
            ids, worker_id = self.get_data_paths_per_worker()
            epoch = 0
            while True:
                order = list(ids)
                random.Random(f"{self.local_rank}-{worker_id}-{epoch}").shuffle(order)
                for n, iid in enumerate(order):
                    data = self.build(*self.rows[iid])
                    data["data_indexes"] = {"data_indexes": [epoch, n], "worker_id": worker_id,
                                            "dataset_name": self.dataset_name}
                    yield data
                epoch += 1
                print(f"{self.dataset_name} repeat {epoch} in rank-{self.local_rank} worker-{worker_id}", flush=True)

    class ReplayDataset(InterleavedBaseIterableDataset):
        """Replay rows (bagel_replay.py), laid out as BAGEL's own datasets / inference paths:
        t2i plain  [caption] -> image                      (= T2IIterableDataset; caption dropped by the packer's
                                                              text_cond_dropout, enable_cfg)
        edit plain [image vae+vit][instruction] -> image   (= UnifiedEditIterableDataset)
        think rows [system][image vae+vit, edit only][prompt][think] -> image, prompt + think dropped together with
                   text_drop (the cards' layout; cfg_text branch [system][image])"""

        def __init__(self, dataset_name, transform, tokenizer, manifest, kind, vit_transform=None,
                     text_drop=TEXT_DROP, local_rank=0, world_size=1, num_workers=8, data_status=None, **_):
            super().__init__(dataset_name, local_rank, world_size, num_workers)
            self.transform, self.vit_transform, self.tokenizer = transform, vit_transform, tokenizer
            self.text_drop, self.root = text_drop, Path(manifest).parent
            self.rows = {}
            for line in Path(manifest).read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == kind:
                    self.rows[r["id"]] = r
            self.data_paths = sorted(self.rows)
            self.set_epoch()

        def build(self, r: dict) -> dict:
            from PIL import Image
            think = r["think"] if r["mode"] == "think" else None
            data = self._init_data()
            if think is not None:
                data = self._add_text(data, GEN_THINK_SYSTEM_PROMPT, need_loss=False, enable_cfg=False)
            if r["kind"] == "edit":
                data = self._add_image(data, pil_img2rgb(Image.open(self.root / r["input"])), need_loss=False,
                                       need_vae=True, need_vit=True)
            if think is None:
                data = self._add_text(data, r["prompt"], need_loss=False)
            elif random.random() >= self.text_drop:
                data = self._add_text(data, r["prompt"], need_loss=False, enable_cfg=False)
                data = self._add_text(data, think, need_loss=False, enable_cfg=False)
            target = self.root / (r["output"] if r["kind"] == "edit" else r["image"])
            return self._add_image(data, pil_img2rgb(Image.open(target)), need_loss=True, need_vae=False,
                                   need_vit=False)

        def __iter__(self):
            ids, worker_id = self.get_data_paths_per_worker()
            epoch = 0
            while True:
                order = list(ids)
                random.Random(f"{self.dataset_name}-{self.local_rank}-{worker_id}-{epoch}").shuffle(order)
                for n, rid in enumerate(order):
                    data = self.build(self.rows[rid])
                    data["data_indexes"] = {"data_indexes": [epoch, n], "worker_id": worker_id,
                                            "dataset_name": self.dataset_name}
                    yield data
                epoch += 1
                print(f"{self.dataset_name} repeat {epoch} in rank-{self.local_rank} worker-{worker_id}", flush=True)

    dataset_info.DATASET_REGISTRY["cards"] = CardEditDataset
    dataset_info.DATASET_INFO["cards"] = {"cards": {"data_dir": str(out)}}
    for group in ("replay_t2i", "replay_edit"):
        dataset_info.DATASET_REGISTRY[group] = ReplayDataset
        dataset_info.DATASET_INFO[group] = {"replay": {"data_dir": str(out)}}   # unused: rows come from the manifest


def datatest_main(a: argparse.Namespace) -> None:
    """Pack a few batches on the CPU exactly as training would and dump the first card sample."""
    register(a.out)
    import torch
    import yaml
    from data.data_utils import add_special_tokens
    from data.dataset_base import DataConfig, PackedDataset
    from modeling.qwen2 import Qwen2Tokenizer
    tok = Qwen2Tokenizer.from_pretrained(BAGEL_W)
    tok, new_ids, _ = add_special_tokens(tok)
    cfg = DataConfig(grouped_datasets=yaml.safe_load((a.out / "data.yaml").read_text()))
    cfg.vit_patch_size, cfg.max_num_patch_per_side = 14, 70
    cfg.vae_image_downsample, cfg.max_latent_size = 16, 64
    cfg.text_cond_dropout_prob, cfg.vae_cond_dropout_prob, cfg.vit_cond_dropout_prob = 0.1, 0.3, 0.3
    ds = PackedDataset(cfg, tokenizer=tok, special_tokens=new_ids, local_rank=0, world_size=1, num_workers=1,
                       expected_num_tokens=a.expected_tokens, max_num_tokens=a.max_tokens,
                       max_num_tokens_per_sample=a.max_sample_tokens, use_flex=True)
    ds.set_epoch(42)
    groups, dumped = {}, set()
    for b, data in zip(range(a.batches), iter(ds)):
        names = [d["dataset_name"] for d in data["batch_data_indexes"]]
        for n in names:
            groups[n] = groups.get(n, 0) + 1
        print(f"batch {b}: tokens {data['sequence_length']} samples {len(names)} {names} "
              f"mse {len(data.get('mse_loss_indexes', []))} ce {len(data.get('ce_loss_indexes', []))} "
              f"images {tuple(data['padded_images'].shape) if 'padded_images' in data else None}", flush=True)
        text = tok.decode(data["packed_text_ids"].tolist())
        if b == 0:
            (a.out / "datatest_text.txt").write_text(text)
            print(text[:3000], flush=True)
        for n in dict.fromkeys(names):   # the first batch with each dataset (layout check of every group)
            if n not in dumped:
                dumped.add(n)
                (a.out / f"datatest_text_{n}_b{b}.txt").write_text(text)
    print("groups", groups, flush=True)


# ------------------------------------------------------------------------------------------------------------- train
PATCHES = [
    # gen-only data has no CE tokens; BAGEL asserts that only for visual_und=False, but freeze_und needs the ViT path
    ("            assert not training_args.visual_und\n", "            pass\n"),
    # no EMA copy: short fine-tunes are evaluated on the raw weights (saves 56 GB of host RAM per rank and a GPU copy)
    ("    ema_model = deepcopy(model)\n", "    ema_model = None\n"),
    ("    ema_model = fsdp_ema_setup(ema_model, fsdp_config)\n", "    ema_model = None\n"),
    ("    ema_model.eval()\n", "    pass\n"),
    # --init: start from an earlier fine-tuned checkpoint (gen.safetensors or model.safetensors) over BAGEL's weights
    ("    fsdp_model = fsdp_wrapper(model, fsdp_config)\n",
     "    if os.environ.get('BAGEL_FT_INIT'):\n"
     "        from safetensors.torch import load_file as _ft_load\n"
     "        _sd, _own = _ft_load(os.environ['BAGEL_FT_INIT']), model.state_dict()\n"
     "        _bad = [k for k in _sd if k not in _own or _own[k].shape != _sd[k].shape]\n"
     "        assert not _bad, f'--init keys not in the model: {_bad[:5]}'\n"
     "        model.load_state_dict(_sd, strict=False)\n"
     "        if dist.get_rank() == 0:\n"
     "            print(f\"init {os.environ['BAGEL_FT_INIT']}: {len(_sd)} tensors\", flush=True)\n"
     "        del _sd, _own\n"
     "    fsdp_model = fsdp_wrapper(model, fsdp_config)\n"),
    ("            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)\n", "            pass\n"),
    # the connector only feeds und (ViT) tokens, whose states freeze_und detaches, so its gradient is zero anyway;
    # frozen, no rank runs a backward through the ViT path (see patch_vit_dummy). With --train-und (freeze_und False)
    # the ViT and connector stay frozen too (only the LLM experts learn), else a rank whose batch has no ViT tokens
    # skips their backward and the FSDP reduce-scatters mismatch (a hang at the first step)
    ("    # Setup FSDP and load pretrained model:\n",
     "    if training_args.visual_und:\n"
     "        model.connector.requires_grad_(False)\n"
     "        model.vit_model.requires_grad_(False)\n"
     "    # Setup FSDP and load pretrained model:\n"),
]


def patch_vit_dummy() -> None:
    """A batch without ViT tokens (t2i samples only, or every ViT condition dropped) makes BAGEL skip its ViT block on
    that rank only, so the other ranks' all-gathers for the FSDP-sharded ViT/connector pair with the wrong collectives
    (first smoke: a crash; it can also hang). Such a batch runs the same modules, in the same order, on a 2x2-patch
    dummy without grad (the ViT and connector are frozen), then the official forward without the ViT block."""
    import torch
    from modeling.bagel.bagel import Bagel
    orig = Bagel.forward
    if getattr(orig, "dummy_vit_patch", False):
        return

    def forward(self, *args, **kw):
        if not self.config.visual_und or kw.get("vit_token_seqlens") is not None:
            return orig(self, *args, **kw)
        dev, side, p = kw["packed_text_ids"].device, self.vit_max_num_patch_per_side, self.vit_patch_size
        pos = torch.tensor([0, 1, side, side + 1], device=dev)
        with torch.no_grad():
            x = self.vit_model(packed_pixel_values=torch.zeros(4, 3 * p * p, dtype=torch.bfloat16, device=dev),
                               packed_flattened_position_ids=pos,
                               cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device=dev), max_seqlen=4)
            self.connector(x)
            self.vit_pos_embed(pos)
        self.config.visual_und = False
        try:
            return orig(self, *args, **kw)
        finally:
            self.config.visual_und = True
    forward.dummy_vit_patch = True
    Bagel.forward = forward


def save_slim(ckpt_dir, train_steps, model, ema_model, optimizer, scheduler, data_status, logger, fsdp_config):
    """bf16 weights only (no optimizer/EMA); --save gen keeps only the generation-side tensors."""
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
    path = os.path.join(ckpt_dir, f"{train_steps:07d}")
    os.makedirs(path, exist_ok=True)
    logger.info(f"Saving slim checkpoint to {path}.")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
        state = model.state_dict()
    if dist.get_rank() == 0:
        mode = SAVE_MODE[0]
        keep = {k: v.to(torch.bfloat16).contiguous() for k, v in state.items() if mode == "full" or is_gen_key(k)}
        name = "model.safetensors" if mode == "full" else "gen.safetensors"
        save_file(keep, os.path.join(path, name + ".tmp"))
        os.replace(os.path.join(path, name + ".tmp"), os.path.join(path, name))
        logger.info(f"saved {len(keep)}/{len(state)} tensors ({mode})")
    del state
    dist.barrier()


def train_main(a: argparse.Namespace) -> None:
    register(a.out)
    stub = types.ModuleType("wandb")
    noop = lambda *x, **k: None  # noqa: E731
    stub.init = stub.log = stub.finish = stub.Settings = noop
    stub.config = types.SimpleNamespace(update=noop)
    sys.modules["wandb"] = stub
    path = Path(BAGEL_SRC) / "train" / "pretrain_unified_navit.py"
    src = path.read_text()
    for old, new in PATCHES:
        assert src.count(old) == 1, f"patch target not unique: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("bagel_pretrain")   # registered: its @dataclass argument classes look up their module
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    ns = mod.__dict__
    exec(compile(src, str(path), "exec"), ns)
    ns["FSDPCheckpoint"].fsdp_save_ckpt = staticmethod(save_slim)
    patch_vit_dummy()
    ck = a.out / "ckpt"
    # --train-und lets the image loss train the und expert too
    freeze_und = "False" if a.train_und else "True"
    lr = a.lr
    SAVE_MODE[0] = a.save
    if a.init:
        os.environ["BAGEL_FT_INIT"] = str(a.init)
    (a.out / "train_env.json").write_text(json.dumps({"freeze_und": freeze_und, "lr": lr,
                                                      "init": str(a.init or ""), "save": a.save}))
    sys.argv = [sys.argv[0],
                "--dataset_config_file", str(a.out / "data.yaml"),
                "--model_path", BAGEL_W, "--layer_module", "Qwen2MoTDecoderLayer", "--max_latent_size", "64",
                "--resume_from", BAGEL_W, "--finetune_from_hf", "True", "--resume_model_only", "True",
                "--finetune_from_ema", "True", "--copy_init_moe", "False",
                "--visual_gen", "True", "--visual_und", "True", "--freeze_und", freeze_und, "--freeze_vit", "True",
                "--use_flex", "True", "--num_workers", "1",
                "--sharding_strategy", "HYBRID_SHARD", "--num_shard", str(a.world), "--num_replicate", "1",
                "--lr", str(lr), "--lr_scheduler", "constant", "--warmup_steps", str(a.warmup),
                "--total_steps", str(a.steps), "--save_every", str(a.save_every), "--log_every", str(a.log_every),
                "--expected_num_tokens", str(a.expected_tokens), "--max_num_tokens", str(a.max_tokens),
                "--max_num_tokens_per_sample", str(a.max_sample_tokens), "--global_seed", str(a.seed),
                "--data_seed", str(a.seed), "--results_dir", str(a.out / "results"), "--checkpoint_dir", str(ck),
                "--wandb_offline", "True"]
    ns["main"]()


# ------------------------------------------------------------------------------------------------------------- check
def check_main(a: argparse.Namespace) -> None:
    """Compare a saved checkpoint with BAGEL's ema.safetensors: which tensors moved, and are they all gen-side?"""
    import torch
    from safetensors import safe_open
    moved, same, missing = [], 0, []
    with safe_open(str(a.ckpt), "pt") as ck, safe_open(os.path.join(BAGEL_W, "ema.safetensors"), "pt") as base:
        names = set(base.keys())
        for k in ck.keys():
            if k not in names:
                missing.append(k)
                continue
            x, y = ck.get_tensor(k), base.get_tensor(k).to(torch.bfloat16)
            if x.shape != y.shape or not torch.equal(x, y):
                moved.append((k, (x.float() - y.float()).abs().max().item() if x.shape == y.shape else -1))
            else:
                same += 1
    bad = [k for k, _ in moved if not is_gen_key(k)]
    print(json.dumps({"tensors": same + len(moved), "unchanged": same, "moved": len(moved),
                      "moved_not_gen": bad[:50], "not_in_base": missing[:50],
                      "moved_sample": moved[:10]}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--min-shards", type=int, default=16)
    p.add_argument("--replay", required=True, help="replay dir (bagel_replay.py pack output)")
    p.add_argument("--thinks", help="glob of the think.jsonl shards written by bagel_cards.py think (pixel arm)")
    p.add_argument("--out", type=Path, required=True)
    for name in ("datatest", "train"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--expected-tokens", type=int, default=24576)
        s.add_argument("--max-tokens", type=int, default=28672)
        s.add_argument("--max-sample-tokens", type=int, default=16384)
        if name == "datatest":
            s.add_argument("--batches", type=int, default=4)
        else:
            s.add_argument("--steps", type=int, required=True)
            s.add_argument("--save-every", type=int, default=500)
            s.add_argument("--warmup", type=int, default=100)
            s.add_argument("--lr", type=float, default=2e-5)
            s.add_argument("--log-every", type=int, default=10)
            s.add_argument("--seed", type=int, default=4396)
            # not --nproc: torchrun would read it as an abbreviation of its own --nproc-per-node
            s.add_argument("--world", type=int, default=int(os.environ.get("WORLD_SIZE", "4")))
            s.add_argument("--train-und", action="store_true", help="also train the understanding expert")
            s.add_argument("--init", type=Path, help="start from this fine-tuned checkpoint over BAGEL's weights")
            s.add_argument("--save", choices=("gen", "full"), default="gen",
                           help="gen: generation-side tensors only; full: all weights (needed with --train-und)")
    c = sub.add_parser("check")
    c.add_argument("--ckpt", type=Path, required=True)
    a = ap.parse_args()
    {"prepare": prepare_main, "datatest": datatest_main, "train": train_main, "check": check_main}[a.cmd](a)


if __name__ == "__main__":
    main()
