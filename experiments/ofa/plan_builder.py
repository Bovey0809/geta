"""Assign ChannelPlans to YOLO26 modules, one module type at a time.

Each `plan_*` function takes the block's INPUT plan, attaches plans to every
Conv it owns, and returns the block's OUTPUT plan. Composing them over
`model.model` gives the whole graph (later phases).

Module coverage (see OFA_SUPERNET_PLAN.md Part 3):
  M1  plain Conv                              DONE
  M2  C3k2 / C2f whose .m are Bottlenecks     DONE   (yolo26s L2, L4)
  M3  C3k2 whose .m are C3k (nested)          TODO   (L6, L8, L13, L16, L19)
  M4  SPPF (repeated concat + residual)       TODO   (L9)
  M5  inter-layer Concat                      TODO   (L12, L15, L18, L21)
  M6  Detect (per-scale inputs, fixed out)    TODO   (L23)
  M7/M8 attention blocks stay unplanned       (frozen at width 1.0 by design)

An unplanned Conv keeps width 1.0 and runs the stock forward, so partial
coverage is always safe to run — that is what makes the module-by-module
sequence possible.
"""

from __future__ import annotations

import torch.nn as nn
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k, C3k2
from ultralytics.nn.modules.conv import Conv

from channel_plan import ChannelPlan, set_plans

__all__ = [
    "plan_conv",
    "plan_bottleneck",
    "plan_c2f",
    "plan_block",
    "PlanError",
]


class PlanError(NotImplementedError):
    """Raised for a module type this phase does not yet handle."""


# --- M1 ---------------------------------------------------------------------


def plan_conv(conv: Conv, in_plan: ChannelPlan) -> ChannelPlan:
    """Plain Conv: output is one free group, sortable later."""
    c = conv.conv
    if c.groups == 1:
        out_plan = ChannelPlan((c.out_channels,))
    else:
        # depth-wise: channels tied to the input layout
        if c.groups != c.in_channels or c.in_channels != c.out_channels:
            raise PlanError(
                f"grouped (non-depth-wise) conv unsupported: groups={c.groups} "
                f"in={c.in_channels} out={c.out_channels}"
            )
        out_plan = in_plan
    set_plans(conv, in_plan, out_plan)
    return out_plan


# --- M2 ---------------------------------------------------------------------


def plan_bottleneck(blk: Bottleneck, in_plan: ChannelPlan) -> ChannelPlan:
    """`y = x + cv2(cv1(x))` when add else `cv2(cv1(x))`.

    The residual forces cv2's OUTPUT layout to equal the block's INPUT layout:
    both operands of the `+` must select the same channels. cv1's hidden layout
    is unconstrained (one free group).
    """
    hidden = ChannelPlan((blk.cv1.conv.out_channels,))
    set_plans(blk.cv1, in_plan, hidden)

    if blk.add:
        if blk.cv2.conv.out_channels != in_plan.total:
            raise PlanError(
                f"residual bottleneck out {blk.cv2.conv.out_channels} != "
                f"in {in_plan.total}"
            )
        out_plan = in_plan  # equal groups => identical selection
    else:
        out_plan = ChannelPlan((blk.cv2.conv.out_channels,))
    set_plans(blk.cv2, hidden, out_plan)
    return out_plan


def plan_c2f(blk: C2f, in_plan: ChannelPlan) -> ChannelPlan:
    """C2f / C3k2:

        y = list(cv1(x).chunk(2, 1))          # cv1 emits 2c; boundary at c
        y.extend(m(y[-1]) for m in self.m)
        return cv2(cat(y, 1))                 # (2 + n) segments

    Giving cv1 the two-group plan `(c, c)` is what makes `chunk(2, 1)` split at
    the right index at every width — no change to the forward code.
    """
    c = blk.c
    if blk.cv1.conv.out_channels != 2 * c:
        raise PlanError(
            f"expected cv1 out == 2*c ({2 * c}), got {blk.cv1.conv.out_channels}"
        )
    set_plans(blk.cv1, in_plan, ChannelPlan((c, c)))

    branch = ChannelPlan((c,))
    segments = [branch, branch]  # y[0], y[1] after the chunk

    cur = branch
    for sub in blk.m:
        cur = plan_submodule(sub, cur)
        segments.append(cur)

    cv2_in = ChannelPlan.cat(segments)
    if blk.cv2.conv.in_channels != cv2_in.total:
        raise PlanError(
            f"cv2 in_channels {blk.cv2.conv.in_channels} != composed segments "
            f"{cv2_in.total} (groups={cv2_in.groups})"
        )
    out_plan = ChannelPlan((blk.cv2.conv.out_channels,))
    set_plans(blk.cv2, cv2_in, out_plan)
    return out_plan


def plan_submodule(sub: nn.Module, in_plan: ChannelPlan) -> ChannelPlan:
    """Dispatch for the contents of a C2f/C3k2 `.m` list."""
    if isinstance(sub, Bottleneck):
        return plan_bottleneck(sub, in_plan)
    if isinstance(sub, C3k):
        raise PlanError("C3k not planned yet — that is M3")
    if isinstance(sub, nn.Sequential):
        raise PlanError(
            "Sequential(.m) implies an attention block (PSABlock) — M7/M8, "
            "left frozen at width 1.0 by design"
        )
    raise PlanError(f"unhandled submodule type {type(sub).__name__}")


# --- dispatcher -------------------------------------------------------------


def plan_block(block: nn.Module, in_plan: ChannelPlan) -> ChannelPlan:
    """Plan one top-level layer. Raises PlanError for not-yet-covered types."""
    if isinstance(block, Conv):
        return plan_conv(block, in_plan)
    if isinstance(block, (C2f, C3k2)):
        return plan_c2f(block, in_plan)
    raise PlanError(f"unhandled block type {type(block).__name__}")
