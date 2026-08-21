"""Importance-based channel sorting for the width-elastic supernet (P3).

WHY
---
Gate A measured that arbitrary first-k channel selection injects error at
*every* layer (3.2% at the very first conv from dropping 4 of 32 channels) and
compounds to ~89% relative MSE by L10. Slicing correctly is not enough if the
channels being kept are arbitrary. Sorting makes "the first k" mean "the k most
important", which is the lever that failure mode calls for.

THE INVARIANT THAT MAKES THIS SAFE
----------------------------------
Permuting a tensor's channels is a pure RELABELLING: if a producer's output
channels are permuted by pi, and every consumer's input columns are permuted by
the same pi, the network computes the same function. So sorting must leave the
w=1.0 output unchanged. That is the test for every module below.

(Unchanged, not bit-identical: permuting input columns changes the order of the
summation inside the convolution, and float addition is not associative, so
differences of ~1e-7 relative are expected and fine. The slicing tests in P1
could demand exactly 0.0 because nothing was reordered there.)

FOUR CONSTRAINTS, EACH LEARNED FROM THE STRUCTURE
-------------------------------------------------
1. **Within-group only.** A permutation may reorder channels *inside* a plan
   group but never across groups — otherwise group-prefix selection stops
   meaning "top-k of each semantic branch" and the chunk/cat boundaries break.
2. **Frozen groups keep identity.** A frozen block's interior is unplanned and
   still expects the original channel order (attention head layout in
   particular), so an adapter's output must NOT be permuted.
3. **Residual ties are forced, not sorted.** In `x + cv2(cv1(x))`, cv2's output
   ordering must equal the block input's ordering. Sorting it independently is
   exactly what broke the earlier `channel_sort.py` attempt.
4. **Repeated segments share one permutation.** SPPF concatenates n+1 pooled
   copies of the same tensor, so all n+1 column segments take cv1's permutation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import (
    SPPF,
    Attention,
    Bottleneck,
    C2f,
    C2PSA,
    C3k,
    C3k2,
)
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.modules.head import Detect

from channel_plan import _IN_PLAN, _OUT_PLAN, ChannelPlan

__all__ = ["sort_model", "importance", "SortError"]


class SortError(RuntimeError):
    pass


# --- importance -------------------------------------------------------------


def importance(conv: Conv) -> torch.Tensor:
    """Per-output-channel importance.

    Uses the effective post-fusion scale |gamma| / sqrt(running_var + eps),
    which is what actually multiplies each channel once BN is folded into the
    conv — a better ranking than |gamma| alone, because a large gamma paired
    with a large running variance contributes no more than a small one paired
    with a small variance.
    """
    bn = conv.bn
    if isinstance(bn, nn.BatchNorm2d):
        return (bn.weight.detach().abs()
                / (bn.running_var.detach() + bn.eps).sqrt())
    # no BN (already fused, or act-only): fall back to weight magnitude
    return conv.conv.weight.detach().abs().flatten(1).sum(1)


# --- permutation helpers ----------------------------------------------------


def _identity(n: int, device=None) -> torch.Tensor:
    return torch.arange(n, device=device)


def _perm_by_importance(conv: Conv, plan: ChannelPlan) -> torch.Tensor:
    """Global permutation of `conv`'s outputs, sorted WITHIN each group.

    Elastic groups are ordered by descending importance; frozen groups keep
    their original order (constraint 2).
    """
    score = importance(conv)
    parts = []
    for off, g, el in zip(plan.offsets(), plan.groups, plan.elastic):
        idx = torch.arange(off, off + g, device=score.device)
        if el:
            order = torch.argsort(score[off:off + g], descending=True)
            parts.append(idx[order])
        else:
            parts.append(idx)
    return torch.cat(parts)


def _split_local(perm: torch.Tensor, plan: ChannelPlan) -> list[torch.Tensor]:
    """Split a global permutation into per-group LOCAL permutations."""
    out = []
    for off, g in zip(plan.offsets(), plan.groups):
        out.append(perm[off:off + g] - off)
    return out


def _permute_out(conv: Conv, perm: torch.Tensor) -> None:
    """Reorder output channels: new position i takes old channel perm[i]."""
    c = conv.conv
    c.weight.data = c.weight.data[perm].contiguous()
    if c.bias is not None:
        c.bias.data = c.bias.data[perm].contiguous()
    bn = conv.bn
    if isinstance(bn, nn.BatchNorm2d):
        bn.weight.data = bn.weight.data[perm].contiguous()
        bn.bias.data = bn.bias.data[perm].contiguous()
        bn.running_mean.data = bn.running_mean.data[perm].contiguous()
        bn.running_var.data = bn.running_var.data[perm].contiguous()


def _permute_in(conv: Conv, perm: torch.Tensor) -> None:
    """Reorder input columns to match a producer that was permuted by `perm`."""
    c = conv.conv
    if c.groups != 1:
        return  # depth-wise: no cross-channel mixing; handled via _permute_out
    c.weight.data = c.weight.data[:, perm].contiguous()


def _sort_out(conv: Conv) -> torch.Tensor:
    """Sort a conv's outputs by importance and return the permutation."""
    plan: ChannelPlan = getattr(conv, _OUT_PLAN)
    perm = _perm_by_importance(conv, plan)
    _permute_out(conv, perm)
    return perm


# --- per-module sorting -----------------------------------------------------


def sort_conv(conv: Conv, in_perm: torch.Tensor) -> torch.Tensor:
    c = conv.conv
    if c.groups == 1:
        _permute_in(conv, in_perm)
        return _sort_out(conv)
    # depth-wise: output channel i depends only on input channel i, so the
    # tied axis simply inherits the incoming order.
    _permute_out(conv, in_perm)
    return in_perm


def sort_bottleneck(blk: Bottleneck, in_perm: torch.Tensor) -> torch.Tensor:
    _permute_in(blk.cv1, in_perm)
    hid = _sort_out(blk.cv1)
    _permute_in(blk.cv2, hid)
    if blk.add:
        # constraint 3: `x + cv2(...)` requires cv2's output ordering to equal
        # the block input's ordering, so this is FORCED, not sorted.
        _permute_out(blk.cv2, in_perm)
        return in_perm
    return _sort_out(blk.cv2)


def sort_submodule(sub: nn.Module, in_perm: torch.Tensor) -> torch.Tensor:
    if isinstance(sub, C3k):
        return sort_c3k(sub, in_perm)
    if isinstance(sub, Bottleneck):
        return sort_bottleneck(sub, in_perm)
    raise SortError(f"unhandled submodule {type(sub).__name__}")


def sort_c3k(blk: C3k, in_perm: torch.Tensor) -> torch.Tensor:
    """cv3(cat((m(cv1(x)), cv2(x)), 1)) — two branches over the same input."""
    c_ = blk.cv1.conv.out_channels
    _permute_in(blk.cv1, in_perm)
    p_m = _sort_out(blk.cv1)
    cur = _split_local(p_m, getattr(blk.cv1, _OUT_PLAN))[0]
    for sub in blk.m:
        cur = sort_submodule(sub, cur)

    _permute_in(blk.cv2, in_perm)
    p_skip = _split_local(_sort_out(blk.cv2), getattr(blk.cv2, _OUT_PLAN))[0]

    # cv3's columns are laid out [m-branch | cv2-branch], matching cat order
    _permute_in(blk.cv3, torch.cat([cur, p_skip + c_]))
    return _sort_out(blk.cv3)


def sort_c2f(blk: C2f, in_perm: torch.Tensor) -> torch.Tensor:
    """cv2(cat(chunk(cv1(x)) + [m(y) ...])) — (2+n) segments of c."""
    c = blk.c
    _permute_in(blk.cv1, in_perm)
    p_cv1 = _sort_out(blk.cv1)
    pi_a, pi_b = _split_local(p_cv1, getattr(blk.cv1, _OUT_PLAN))

    segments = [pi_a, pi_b]
    cur = pi_b
    for sub in blk.m:
        cur = sort_submodule(sub, cur)
        segments.append(cur)

    cols = torch.cat([seg + i * c for i, seg in enumerate(segments)])
    if cols.numel() != blk.cv2.conv.in_channels:
        raise SortError(
            f"cv2 columns {cols.numel()} != in_channels "
            f"{blk.cv2.conv.in_channels}"
        )
    _permute_in(blk.cv2, cols)
    return _sort_out(blk.cv2)


def sort_sppf(blk: SPPF, in_perm: torch.Tensor) -> torch.Tensor:
    c_ = blk.cv1.conv.out_channels
    _permute_in(blk.cv1, in_perm)
    ph = _split_local(_sort_out(blk.cv1), getattr(blk.cv1, _OUT_PLAN))[0]

    # constraint 4: every pooled copy carries cv1's permutation
    n_rep = int(getattr(blk, "n", 3)) + 1
    _permute_in(blk.cv2, torch.cat([ph + i * c_ for i in range(n_rep)]))

    if getattr(blk, "add", False):
        _permute_out(blk.cv2, in_perm)  # residual tie
        return in_perm
    return _sort_out(blk.cv2)


def sort_adapter_chain(mod: nn.Module, in_perm: torch.Tensor) -> torch.Tensor:
    """Frozen-block boundary: consume the permuted input, emit identity.

    Mirrors plan_builder.plan_adapter_chain. The adapter's OUTPUT must stay in
    the original order (constraint 2) because everything downstream of it is
    unplanned and was never permuted.
    """
    cur = in_perm
    for conv in (m for m in mod.modules() if isinstance(m, Conv)):
        c = conv.conv
        if c.groups == 1:
            _permute_in(conv, cur)
            return _identity(c.out_channels, cur.device)
        _permute_out(conv, cur)  # depth-wise pass-through
    raise SortError(f"{type(mod).__name__}: no adapter conv found")


def sort_detect(det: Detect, source_perms: list[torch.Tensor]) -> None:
    for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branches = getattr(det, attr, None)
        if branches is None:
            continue
        for seq, perm in zip(branches, source_perms):
            sort_adapter_chain(seq, perm)


def _has_attention(mod: nn.Module) -> bool:
    return any(isinstance(s, Attention) for s in mod.modules())


# --- whole graph ------------------------------------------------------------


@torch.no_grad()
def sort_model(model: nn.Module, verbose: bool = False) -> list:
    """Sort every planned layer's channels by importance, in forward order.

    Must run AFTER plan_model and BEFORE any BN recalibration (stored per-width
    stats would otherwise refer to the pre-permutation channel order).
    """
    top = model.model
    perms: list = [None] * len(top)
    dev = next(model.parameters()).device
    rgb = _identity(3, dev)

    def srcs_of(i: int, f):
        return [(i - 1 if s == -1 else s) for s in (f if isinstance(f, list) else [f])]

    for i, L in enumerate(top):
        f = getattr(L, "f", -1)
        srcs = srcs_of(i, f)
        in_perms = [rgb if i == 0 else perms[s] for s in srcs]

        if isinstance(L, Detect):
            sort_detect(L, in_perms)
            perms[i] = None
        elif isinstance(L, Concat):
            offs, acc = [], 0
            for s, p in zip(srcs, in_perms):
                offs.append(p + acc)
                acc += p.numel()
            perms[i] = torch.cat(offs)
        elif isinstance(L, nn.Upsample):
            perms[i] = in_perms[0]
        elif isinstance(L, C2PSA) or _has_attention(L):
            perms[i] = sort_adapter_chain(L, in_perms[0])
        elif isinstance(L, SPPF):
            perms[i] = sort_sppf(L, in_perms[0])
        elif isinstance(L, (C2f, C3k2)):
            perms[i] = sort_c2f(L, in_perms[0])
        elif isinstance(L, Conv):
            perms[i] = sort_conv(L, in_perms[0])
        else:
            raise SortError(f"layer {i}: unhandled {type(L).__name__}")

        if verbose and perms[i] is not None:
            moved = int((perms[i] != _identity(perms[i].numel(), dev)).sum())
            print(f"  L{i:2d} {type(L).__name__:10s} "
                  f"n={perms[i].numel():4d} moved={moved}")
    return perms
