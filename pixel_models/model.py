"""The pixel-space flow-matching transformer of Sec. 4.2.

A plain ViT on 8x8 grey patches in the style of JiT: no VAE, no tokenizer, no text encoder. Its input is the
channel-wise concatenation of the conditioning canvas (the problem canvas, or the current canvas in frame-by-frame
reasoning) and the noisy target canvas; the time step enters through adaptive layer normalization. The network predicts
the clean canvas (x-prediction) and is trained with the velocity loss:

    x_t = t * y + (1 - t) * e,   t = sigmoid(n), n ~ N(0, 1),   loss = || (y_hat - x_t) / (1 - t) - (y - e) ||^2

Sizes (width, depth, heads) are those of Tab. 6. S, the canvas side in pixels, is set by the trainers before build().
"""

from __future__ import annotations

import math
from pathlib import Path

S = 192                # canvas side in pixels (192; 256 for Word Problems), set by the caller before build()
PATCH = 8
ASSETS = Path(__file__).resolve().parents[1] / "assets"
SIZES = {"XS": (256, 6, 4), "S": (384, 12, 6), "M": (512, 12, 8), "B": (768, 12, 12), "L": (1024, 24, 16)}
CKPT_ACT = False       # activation checkpointing of the blocks while training (--ckpt-act)


def build(size: str, ch: int = 1):
    """the flow ViT of one size for S x S canvases with ch channels"""
    import torch
    import torch.nn as nn
    d, depth, heads = SIZES[size]
    P = PATCH
    N = (S // P) ** 2

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
            self.ada = nn.Linear(d, 6 * d)
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

        def forward(self, x, c):
            s1, b1, g1, s2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, -1)
            h = self.n1(x) * (1 + s1) + b1
            x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
            h = self.n2(x) * (1 + s2) + b2
            return x + g2 * self.mlp(h)

    class PixFlow(nn.Module):
        """input [condition, x_t] as 8x8 patches, output the clean canvas"""

        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(2 * ch * P * P, d)
            self.pos = nn.Parameter(torch.zeros(1, N, d)); nn.init.normal_(self.pos, std=0.02)
            self.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(d, elementwise_affine=False)
            self.ada = nn.Linear(d, 2 * d); nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
            self.out = nn.Linear(d, ch * P * P); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        @staticmethod
        def patch(x):  # (B, C, S, S) -> (B, N, C*P*P)
            B, C = x.shape[:2]
            return x.reshape(B, C, S // P, P, S // P, P).permute(0, 2, 4, 1, 3, 5).reshape(B, N, C * P * P)

        @staticmethod
        def unpatch(t):  # (B, N, ch*P*P) -> (B, ch, S, S)
            B = t.shape[0]
            return t.reshape(B, S // P, S // P, ch, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, ch, S, S)

        def forward(self, cond, xt, t):
            half = 128
            f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
            e = torch.cat([torch.sin(t[:, None] * 1000 * f), torch.cos(t[:, None] * 1000 * f)], -1)
            c = self.temb(e)
            h = self.embed(self.patch(torch.cat([cond, xt], 1))) + self.pos
            for blk in self.blocks:
                if CKPT_ACT and self.training:
                    import torch.utils.checkpoint as tuc
                    h = tuc.checkpoint(blk, h, c, use_reentrant=False)
                else:
                    h = blk(h, c)
            s, b = self.ada(c).unsqueeze(1).chunk(2, -1)
            return self.unpatch(self.out(self.norm(h) * (1 + s) + b))

    return PixFlow()
