"""Minimal LoRA for nn.Linear layers (self-contained so the injection, scaling and saving are fully under test).

Base weights stay frozen. LoRA weights are float32 whatever the base dtype: inputs are cast up and the update cast
back. Up weights start at zero, so an untrained LoRA leaves the model output unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file, save_file
from torch import nn

META_KEY = "e5x"   # safetensors metadata key of a saved LoRA (the benchmark's internal name, as in its item ids)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.multiplier = alpha / rank
        self.scale = 1.0

    @property
    def weight(self):  # some blocks read .weight.dtype from their projections
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.scale == 0.0:
            return out
        delta = self.up(self.down(x.to(self.down.weight.dtype))) * (self.multiplier * self.scale)
        return out + delta.to(out.dtype)


def default_matcher(suffixes: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda name: any(name == s or name.endswith("." + s) for s in suffixes)


def inject(root: nn.Module, match: Callable[[str], bool], rank: int, alpha: float) -> list[str]:
    """Replace every matching nn.Linear under root. Returns the wrapped module names (sorted)."""
    names = [n for n, m in root.named_modules() if isinstance(m, nn.Linear) and match(n)]
    if not names:
        raise ValueError("LoRA matcher selected no linear layers")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha))
    return sorted(names)


def lora_modules(root: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in root.named_modules() if isinstance(m, LoRALinear)]


def trainable_parameters(root: nn.Module) -> list[nn.Parameter]:
    params = []
    for _, module in lora_modules(root):
        params += [module.down.weight, module.up.weight]
    return params


def set_scale(root: nn.Module, scale: float) -> None:
    for _, module in lora_modules(root):
        module.scale = scale


def freeze_base(root: nn.Module) -> None:
    for p in root.parameters():
        p.requires_grad_(False)
    for p in trainable_parameters(root):
        p.requires_grad_(True)


def save(roots: dict[str, nn.Module], path: Path, meta: dict) -> None:
    tensors = {}
    for prefix, root in roots.items():
        for name, module in lora_modules(root):
            tensors[f"{prefix}/{name}/down"] = module.down.weight.detach().cpu().contiguous()
            tensors[f"{prefix}/{name}/up"] = module.up.weight.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={META_KEY: json.dumps(meta)})


def load(roots: dict[str, nn.Module], path: Path, rank: int, alpha: float) -> dict:
    """Inject exactly the layers saved in `path` and load their weights (prefixes absent from the file are untouched)."""
    tensors = load_file(str(path))
    by_prefix: dict[str, set[str]] = {}
    for key in tensors:
        prefix, name, _ = key.split("/")
        by_prefix.setdefault(prefix, set()).add(name)
    for prefix, names in by_prefix.items():
        root = roots[prefix]
        inject(root, lambda n, s=names: n in s, rank, alpha)
        device = next(root.parameters()).device
        for name, module in lora_modules(root):
            module.down.weight.data = tensors[f"{prefix}/{name}/down"].to(device)
            module.up.weight.data = tensors[f"{prefix}/{name}/up"].to(device)
    return {k: len(v) for k, v in by_prefix.items()}
