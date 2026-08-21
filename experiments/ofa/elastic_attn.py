"""Width-elastic attention for YOLO26 (P5) — unfreezes C2PSA and C3k2-attn.

WHY THIS MATTERS
----------------
Until now the 32 convs inside C2PSA (L10) and C3k2-attn (L22) were frozen at
full width, and they are the WIDEST layers in the network. That floored w=0.5
at 5.05 M params — nearly 2x yolo26n's 2.6 M — so "reach n-territory" was
unreachable by construction. Attention is the only remaining source of
low-width savings.

WHY num_heads STAYS FIXED AND head_dim SHRINKS
----------------------------------------------
`Attention.forward` reshapes qkv as (B, num_heads, 2*key_dim + head_dim, N),
so its channels are laid out HEAD-MAJOR: per head, [q(key_dim) | k(key_dim) |
v(head_dim)]. Two ways to shrink it:

* Drop whole heads (num_heads scales). Every retained head stays intact, but
  granularity is one head = head_dim channels. With c=256 and head_dim=64 the
  only attainable widths are {1.0, 0.75, 0.5, 0.25} — far too coarse, and
  w=0.875 would need 3.5 heads.
* Shrink key_dim and head_dim, num_heads fixed (CHOSEN). Because
  `dim == num_heads * head_dim` exactly, the block's active channel count
  `num_heads * round(head_dim*w)` matches the head-structured plan at EVERY
  width, so the `x + attn(x)` residual always lines up. Granularity is
  num_heads channels instead of head_dim.

For this to hold, the tensors flowing into an attention block must be
HEAD-STRUCTURED: plan `(head_dim,) * num_heads`, not a single `(c,)` group.
A single group would give `round(c*w)`, which differs from
`num_heads*round(head_dim*w)` at some widths (w=0.9: 230 vs 232) and would
break the residual. plan_builder builds these plans.

WHAT THE PATCHED FORWARD MUST RECOMPUTE
---------------------------------------
`num_heads`, `head_dim`, `key_dim` and `scale` are cached ints/floats set in
__init__. The elastic forward derives the ACTIVE key_dim/head_dim from the
current width using the SAME rounding as ChannelPlan (so the tensor it receives
always matches), and recomputes `scale = key_dim**-0.5` — forgetting the scale
would silently mis-temper the softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2PSA, Attention

from channel_plan import _OUT_PLAN, _WIDTH_ATTR, _round_group

__all__ = ["install_elastic_attention", "active_head_dims"]


def active_head_dims(attn: Attention) -> tuple[int, int, int]:
    """(num_heads, active key_dim, active head_dim) for the current width.

    Rounding MUST match ChannelPlan._round_group, or the reshape below will
    disagree with the number of channels qkv actually produced.
    """
    w = getattr(attn.qkv, _WIDTH_ATTR, 1.0) if hasattr(attn.qkv, _OUT_PLAN) else 1.0
    if w >= 1.0:
        return attn.num_heads, attn.key_dim, attn.head_dim
    return (
        attn.num_heads,
        _round_group(attn.key_dim, w),
        _round_group(attn.head_dim, w),
    )


def _elastic_attention_forward(self: Attention, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    N = H * W
    nh, kd, hd = active_head_dims(self)

    qkv = self.qkv(x)
    expected = nh * (2 * kd + hd)
    if qkv.shape[1] != expected:
        raise RuntimeError(
            f"Attention: qkv emitted {qkv.shape[1]} channels but the active "
            f"head geometry needs {expected} (num_heads={nh}, key_dim={kd}, "
            f"head_dim={hd}). The qkv out_plan must be (kd, kd, hd) repeated "
            f"num_heads times, head-major."
        )
    q, k, v = qkv.view(B, nh, 2 * kd + hd, N).split([kd, kd, hd], dim=2)

    scale = kd ** -0.5  # recomputed: self.scale is for the FULL key_dim
    attn = (q * scale).transpose(-2, -1) @ k
    attn = attn.softmax(dim=-1)

    vc = nh * hd
    if C != vc:
        raise RuntimeError(
            f"Attention: input has {C} channels but v has {vc}. The block's "
            f"input plan must be head-structured (head_dim,)*num_heads so the "
            f"residual stays aligned."
        )
    out = (v @ attn.transpose(-2, -1)).view(B, vc, H, W) + self.pe(
        v.reshape(B, vc, H, W)
    )
    return self.proj(out)


def _elastic_c2psa_forward(self: C2PSA, x: torch.Tensor) -> torch.Tensor:
    """Stock uses `.split((self.c, self.c), 1)`; `self.c` is frozen at init.

    `chunk(2, 1)` is width-adaptive and splits in exactly the same place,
    because cv1's out_plan is two identical head-structured halves.
    """
    a, b = self.cv1(x).chunk(2, dim=1)
    b = self.m(b)
    return self.cv2(torch.cat((a, b), 1))


_INSTALLED = False


def install_elastic_attention() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    Attention.forward = _elastic_attention_forward
    C2PSA.forward = _elastic_c2psa_forward
    _INSTALLED = True
