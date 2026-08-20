"""Build a genuinely-narrow twin of a planned module, for the oracle.

The twin is produced by deep-copying the wide module and replacing every
*planned* Conv with a real, smaller `nn.Conv2d` + `nn.BatchNorm2d` holding the
gathered `[out_sel, in_sel]` sub-tensor. Plans are then stripped, so the twin
runs the **stock ultralytics forward on genuinely small tensors**.

Why build it this way instead of via the module's constructor:
  * type-agnostic — one implementation covers Conv, C2f/C3k2, C3k, SPPF, …
  * no need to reverse-engineer constructor args (`e`, `c3k`, `shortcut`, …)
    or worry about `int()` vs `round()` mismatches in derived channel counts
  * the reference executes a DIFFERENT code path (stock ultralytics) from the
    thing under test (the elastic sliced path), which is what makes the
    comparison meaningful rather than circular

Caveat handled: modules that cache an integer channel count used by their
forward (`C2PSA.split(self.c, self.c)`) would need that attribute rescaled.
C2f/C3k2/C3k/SPPF forwards use only `chunk(2,1)` / `cat`, which are
layout-driven, so nothing to patch for M1–M4. `assert_no_stale_channel_attr`
guards against that assumption silently breaking on a new module type.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from channel_plan import _IN_PLAN, _OUT_PLAN, _WIDTH_ATTR, ChannelPlan  # noqa: E402

__all__ = ["build_narrow_twin", "planned_convs", "assert_no_stale_channel_attr"]


def planned_convs(mod: nn.Module):
    """Every Conv carrying a plan, in module-traversal order."""
    return [m for m in mod.modules() if isinstance(m, Conv) and hasattr(m, _OUT_PLAN)]


def _narrow_conv(src_conv: Conv, w: float) -> None:
    """Replace src_conv.conv / .bn in place with gathered narrow versions."""
    in_plan: ChannelPlan = getattr(src_conv, _IN_PLAN)
    out_plan: ChannelPlan = getattr(src_conv, _OUT_PLAN)
    c: nn.Conv2d = src_conv.conv

    if c.groups == 1:
        in_sel = in_plan.select(w)
        out_sel = out_plan.select(w)
        new = nn.Conv2d(
            in_sel.numel(), out_sel.numel(), c.kernel_size, c.stride,
            c.padding, c.dilation, groups=1, bias=c.bias is not None,
        )
        new.weight.data.copy_(c.weight.data[out_sel][:, in_sel])
    else:
        # depth-wise: one group per surviving channel
        sel = in_plan.select(w)
        out_sel = sel
        new = nn.Conv2d(
            sel.numel(), sel.numel(), c.kernel_size, c.stride, c.padding,
            c.dilation, groups=sel.numel(), bias=c.bias is not None,
        )
        new.weight.data.copy_(c.weight.data[sel])

    if c.bias is not None:
        new.bias.data.copy_(c.bias.data[out_sel])
    src_conv.conv = new

    bn = src_conv.bn
    if isinstance(bn, nn.BatchNorm2d):
        nb = nn.BatchNorm2d(out_sel.numel(), eps=bn.eps, momentum=bn.momentum)
        nb.weight.data.copy_(bn.weight.data[out_sel])
        nb.bias.data.copy_(bn.bias.data[out_sel])
        nb.running_mean.data.copy_(bn.running_mean.data[out_sel])
        nb.running_var.data.copy_(bn.running_var.data[out_sel])
        nb.eval()
        src_conv.bn = nb

    # Strip plans so the twin takes the stock (unplanned) forward path.
    for attr in (_IN_PLAN, _OUT_PLAN, _WIDTH_ATTR):
        if hasattr(src_conv, attr):
            delattr(src_conv, attr)


_CHANNEL_ATTRS = ("c", "c1", "c2", "c_", "cv", "num_heads", "key_dim", "head_dim")


def assert_no_stale_channel_attr(mod: nn.Module, w: float) -> None:
    """Fail loudly if a module caches a channel count its forward might use.

    C2f/C3k2 cache `.c` but their forward uses `chunk(2,1)`, which is derived
    from the tensor, so `.c` is inert there. Attention modules DO use their
    cached dims — this is the tripwire for when we get to M7/M8.
    """
    from ultralytics.nn.modules.block import Attention

    for m in mod.modules():
        if isinstance(m, Attention):
            raise AssertionError(
                "Attention module in the narrow twin: its cached num_heads / "
                "key_dim / head_dim ARE used by forward, so a channel-sliced "
                "twin is invalid until M7/M8 rescales them."
            )


def build_narrow_twin(wide: nn.Module, w: float) -> nn.Module:
    """Deep-copy `wide` and shrink every planned Conv to its width-`w` slice."""
    assert_no_stale_channel_attr(wide, w)
    twin = copy.deepcopy(wide)
    for conv in planned_convs(twin):
        _narrow_conv(conv, w)
    twin.eval()
    return twin
