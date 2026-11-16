"""One adapter per open image editing model. Each mirrors the model's own diffusers pipeline and upstream training recipe.

Conventions: z = normalised clean latent in the model's own (unpacked) layout; `condition` builds everything derived
from prompt + input card; `predict` returns the velocity for the target latent in the same layout as z.

Routing of the input card (read from the pipeline sources):
  flux2_klein_base, flux3_klein_base : text encoder sees only the prompt; the card reaches the DiT as VAE latents.
  qwen_image_21, longcat_image_edit  : the VLM text encoder also reads the card; the DiT also gets VAE latents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .paths import model as model_path

SIZE = 1024
DTYPE = torch.bfloat16


def _to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Adapter:
    name: str
    path: str
    lora_suffixes: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")
    encoder_sees_image = False
    steps: int = 50

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.pipe: Any = None

    @property
    def dit(self):
        return self.pipe.transformer

    @property
    def encoder(self):
        return self.pipe.text_encoder

    def load(self) -> "Adapter":
        raise NotImplementedError

    def freeze(self) -> None:
        for module in (self.dit, self.encoder, self.pipe.vae):
            module.requires_grad_(False)

    def latent(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    def condition(self, prompt: str, image: Image.Image, *, encoder_image: Image.Image | None = None) -> dict[str, Any]:
        """encoder_image: for encoders that read the card, a different image for the encoder only (ablations)."""
        raise NotImplementedError

    def predict(self, zt: torch.Tensor, timestep: torch.Tensor, z_cond: torch.Tensor, cond: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, prompt: str, image: Image.Image, seed: int, steps: int | None = None,
                 encoder_image: Image.Image | None = None) -> Image.Image:
        """encoder_image: a different card for the text encoder only (encoders that read the card; ablations)."""
        raise NotImplementedError

    def sample_time(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform over the training schedule. Returns (timestep in [0,1000], sigma)."""
        sched = self.train_scheduler
        index = torch.randint(0, sched.config.num_train_timesteps, (1,), generator=generator)
        return sched.timesteps[index].to(self.device), sched.sigmas[index].to(self.device)


class Flux2Klein(Adapter):
    name, path = "flux2_klein_base", model_path("FLUX.2-klein-base-4B")
    lora_suffixes = ("to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj", "attn.to_out")
    steps = 50
    guidance = 4.0

    def load(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(self.path, torch_dtype=DTYPE).to(self.device)
        self.train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.freeze()
        return self

    @torch.no_grad()
    def latent(self, image):
        tensor = self.pipe.image_processor.preprocess(image, height=SIZE, width=SIZE).to(self.device, DTYPE)
        return self.pipe._encode_vae_image(tensor, generator=None)

    def condition(self, prompt, image, *, encoder_image=None):
        if encoder_image is not None:
            raise ValueError("this encoder does not read the card")
        embeds, text_ids = self.pipe.encode_prompt(prompt=prompt, device=self.device)
        return {"embeds": embeds, "text_ids": text_ids}

    def predict(self, zt, timestep, z_cond, cond):
        pipe = self.pipe
        ids = pipe._prepare_latent_ids(zt).to(self.device)
        cond_ids = pipe._prepare_image_ids([z_cond[:1]]).to(self.device)
        packed = pipe._pack_latents(zt)
        out = self.dit(hidden_states=torch.cat([packed, pipe._pack_latents(z_cond)], dim=1), timestep=timestep / 1000,
                       guidance=None, encoder_hidden_states=cond["embeds"], txt_ids=cond["text_ids"],
                       img_ids=torch.cat([ids, cond_ids], dim=1), return_dict=False)[0]
        return pipe._unpack_latents_with_ids(out[:, :packed.shape[1]], ids)

    @torch.no_grad()
    def generate(self, prompt, image, seed, steps=None, encoder_image=None):
        if encoder_image is not None:
            raise ValueError("this encoder does not read the card")
        generator = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(prompt=prompt, image=image, height=SIZE, width=SIZE, num_inference_steps=steps or self.steps,
                         guidance_scale=self.guidance, generator=generator).images[0]


class Flux3Klein(Adapter):
    name, path = "flux3_klein_base", model_path("flux-3-image-klein-base")
    lora_suffixes = ("to_q", "to_k", "to_v", "to_out.0", "ff.linear_in", "ff.linear_out")
    steps = 50
    FLOW_SHIFT = 6.93
    OUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
    ROPE_SCALE = 0.5
    PAD_MULTIPLE = 80

    def load(self):
        import diffusers
        from diffusers import FlowMatchEulerDiscreteScheduler
        blocks = diffusers.ModularPipeline.from_pretrained(self.path).blocks
        blocks.sub_blocks.pop("prompt_upsampler")
        self.pipe = blocks.init_pipeline(self.path)
        self.pipe.load_components(dtype={"vae": torch.float32, "default": DTYPE})
        self.pipe.to(self.device)
        self.train_scheduler = FlowMatchEulerDiscreteScheduler(shift=self.FLOW_SHIFT, use_dynamic_shifting=False)
        vae = self.pipe.vae
        self.bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(self.device, torch.float32)
        self.bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(self.device, torch.float32)
        self.freeze()
        return self

    @staticmethod
    def _pack(z):
        b, c, h, w = z.shape
        return z.reshape(b, c, h * w).permute(0, 2, 1)

    def _ids(self, h, w, l_value):
        ids = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.tensor([l_value])).float()
        ids[:, 1:3] *= self.ROPE_SCALE
        return ids.to(self.device)

    @torch.no_grad()
    def latent(self, image):
        from torchvision.transforms.functional import pil_to_tensor
        tensor = (pil_to_tensor(image.convert("RGB")).float() / 127.5 - 1)[None].to(self.device, torch.float32)
        z = self.pipe.vae.encode(tensor).latent_dist.mode()
        b, c, h, w = z.shape
        z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        return ((z - self.bn_mean) / self.bn_std).to(DTYPE)

    def condition(self, prompt, image, *, encoder_image=None):
        import math
        if encoder_image is not None:
            raise ValueError("this encoder does not read the card")
        tok = self.pipe.tokenizer
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        real = tok.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)["input_ids"].shape[1]
        length = min(math.ceil(real / self.PAD_MULTIPLE) * self.PAD_MULTIPLE, 8192)
        enc = tok.tokenizer(text, return_tensors="pt", padding="max_length", padding_side="right", truncation=True,
                            max_length=length)
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        positions = torch.arange(length, device=self.device)[None].masked_fill(mask == 0, 1)
        out = self.encoder(input_ids=ids, attention_mask=mask, position_ids=positions, output_hidden_states=True,
                           use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in self.OUT_LAYERS], dim=1).to(DTYPE)
        b, n, s, d = stacked.shape
        text_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(s)).float().to(self.device)
        return {"embeds": stacked.permute(0, 2, 1, 3).reshape(b, s, n * d), "text_ids": text_ids}

    def predict(self, zt, timestep, z_cond, cond):
        _, _, h, w = zt.shape
        out = self.dit(hidden_states=self._pack(zt), img_ids=self._ids(h, w, 0), timestep=timestep / 1000,
                       image_cond_hidden_states=self._pack(z_cond), image_cond_ids=self._ids(h, w, -1),
                       encoder_hidden_states=cond["embeds"], txt_ids=cond["text_ids"], return_dict=False)[0]
        b, _, c = out.shape
        return out.permute(0, 2, 1).reshape(b, c, h, w)

    @torch.no_grad()
    def generate(self, prompt, image, seed, steps=None, encoder_image=None):
        if encoder_image is not None:
            raise ValueError("this encoder does not read the card")
        generator = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(prompt=prompt, image=image, height=SIZE, width=SIZE, generator=generator,
                        num_inference_steps=steps or self.steps, output="images")
        return list(out)[0]


class QwenImage21(Adapter):
    name, path = "qwen_image_21", model_path("Qwen-Image-2.1")
    encoder_sees_image = True
    steps = 40

    def load(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
        self.pipe = QwenImage21Pipeline.from_pretrained(self.path, torch_dtype=DTYPE).to(self.device)
        self.train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.freeze()
        return self

    @torch.no_grad()
    def latent(self, image):
        tensor = self.pipe.image_processor.preprocess(image.convert("RGBA"), width=SIZE, height=SIZE).unsqueeze(2)
        return self.pipe._encode_vae_image(tensor.to(self.device, DTYPE), None)

    def condition(self, prompt, image, *, encoder_image=None):
        seen = (encoder_image or image).convert("RGBA")
        seen = self.pipe.image_processor.resize(seen, width=SIZE, height=SIZE)
        embeds, mask, pad_mask = self.pipe.encode_prompt(image=[seen], prompt=prompt, device=self.device,
                                                         num_images_per_prompt=1)
        return {"embeds": embeds, "mask": mask, "pad_mask": pad_mask}

    def predict(self, zt, timestep, z_cond, cond):
        pipe = self.pipe
        b, c, _, h, w = zt.shape
        packed = pipe._pack_latents(zt, b, c, h, w)
        packed_cond = pipe._pack_latents(z_cond, b, c, h, w)
        img_mask = torch.cat([cond["pad_mask"], cond["pad_mask"].new_ones(b, (h * w) // 4)], dim=1)
        out = self.dit(hidden_states=torch.cat([packed_cond, packed], dim=1), encoder_hidden_states=cond["embeds"],
                       encoder_hidden_states_mask=cond["mask"], timestep=timestep / 1000,
                       img_shapes=[[(1, h, w), (1, h, w)]] * b, img_mask=img_mask, return_dict=False)[0]
        return pipe._unpack_latents(out[:, -packed.shape[1]:], SIZE, SIZE, pipe.vae_scale_factor)

    @torch.no_grad()
    def generate(self, prompt, image, seed, steps=None, encoder_image=None):
        generator = torch.Generator(self.device).manual_seed(seed)
        original = self.pipe._get_qwen_prompt_embeds
        if encoder_image is not None:
            seen = self.pipe.image_processor.resize(encoder_image.convert("RGBA"), width=SIZE, height=SIZE)
            self.pipe._get_qwen_prompt_embeds = lambda p, image=None, device=None: original(p, [seen], device)
        try:
            out = self.pipe(prompt=prompt, image=image, height=SIZE, width=SIZE, num_inference_steps=steps or self.steps,
                            generator=generator).images[0]
        finally:
            self.pipe._get_qwen_prompt_embeds = original
        return out.convert("RGB")


class LongCatEdit(Adapter):
    name, path = "longcat_image_edit", model_path("LongCat-Image-Edit")
    encoder_sees_image = True
    steps = 50
    guidance = 4.5
    PACKED_SIDE = 64  # 1024 / 8 / 2

    def load(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, LongCatImageEditPipeline
        # cuDNN's scaled-dot-product attention breaks in backward for this model ("mha_graph.execute" / NaN loss).
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.pipe = LongCatImageEditPipeline.from_pretrained(self.path, torch_dtype=DTYPE).to(self.device)
        self.train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.freeze()
        return self

    @torch.no_grad()
    def latent(self, image):
        tensor = self.pipe.image_processor.preprocess(image, SIZE, SIZE).to(self.device, DTYPE)
        return self.pipe._encode_vae_image(tensor, generator=None)

    def condition(self, prompt, image, *, encoder_image=None):
        seen = self.pipe.image_processor.resize(encoder_image or image, SIZE // 2, SIZE // 2)
        embeds, text_ids = self.pipe.encode_prompt(prompt=[prompt], image=seen)
        return {"embeds": embeds, "text_ids": text_ids}

    def predict(self, zt, timestep, z_cond, cond):
        from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids
        pipe = self.pipe
        b, c, h, w = zt.shape
        packed = pipe._pack_latents(zt, b, c, h, w)
        packed_cond = pipe._pack_latents(z_cond, b, c, h, w)
        seq = cond["embeds"].shape[1]
        ids = prepare_pos_ids(modality_id=1, type="image", start=(seq, seq), height=h // 2, width=w // 2).to(self.device)
        cond_ids = prepare_pos_ids(modality_id=2, type="image", start=(seq, seq), height=h // 2, width=w // 2).to(self.device, dtype=torch.float64)
        out = self.dit(hidden_states=torch.cat([packed, packed_cond], dim=1), timestep=timestep / 1000, guidance=None,
                       encoder_hidden_states=cond["embeds"], txt_ids=cond["text_ids"],
                       img_ids=torch.cat([ids, cond_ids], dim=0), return_dict=False)[0]
        return pipe._unpack_latents(out[:, :packed.shape[1]], SIZE, SIZE, pipe.vae_scale_factor)

    @torch.no_grad()
    def generate(self, prompt, image, seed, steps=None, encoder_image=None):
        generator = torch.Generator(self.device).manual_seed(seed)
        original = self.pipe._encode_prompt
        if encoder_image is not None:
            seen = self.pipe.image_processor.resize(encoder_image, SIZE // 2, SIZE // 2)
            self.pipe._encode_prompt = lambda p, image: original(p, seen)
        try:
            return self.pipe(image=image, prompt=prompt, negative_prompt=" ", guidance_scale=self.guidance,
                             num_inference_steps=steps or self.steps, generator=generator).images[0]
        finally:
            self.pipe._encode_prompt = original


ADAPTERS = {cls.name: cls for cls in (Flux2Klein, Flux3Klein, QwenImage21, LongCatEdit)}
