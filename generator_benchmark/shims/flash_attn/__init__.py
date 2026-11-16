"""Minimal stand-in for flash_attn.flash_attn_varlen_func, used by BAGEL when flash-attn is not installed.

This computes the same thing with
torch scaled_dot_product_attention, one packed sequence at a time. Causal masking follows flash-attn's convention
for seqlen_q != seqlen_k (the query block is aligned to the end of the key block).
"""

import torch
import torch.nn.functional as F


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
                           softmax_scale=None, causal=False, **_):
    out = torch.empty_like(q)
    heads, kv_heads = q.shape[1], k.shape[1]
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs = q[cq[i]:cq[i + 1]].transpose(0, 1)
        ks = k[ck[i]:ck[i + 1]].transpose(0, 1)
        vs = v[ck[i]:ck[i + 1]].transpose(0, 1)
        if kv_heads != heads:
            ks = ks.repeat_interleave(heads // kv_heads, dim=0)
            vs = vs.repeat_interleave(heads // kv_heads, dim=0)
        mask = None
        if causal:
            lq, lk = qs.shape[1], ks.shape[1]
            mask = torch.arange(lk, device=q.device)[None, :] <= (torch.arange(lq, device=q.device)[:, None] + (lk - lq))
        o = F.scaled_dot_product_attention(qs[None], ks[None], vs[None], attn_mask=mask, scale=softmax_scale)
        out[cq[i]:cq[i + 1]] = o[0].transpose(0, 1)
    return out
