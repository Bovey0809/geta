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
from ultralytics.nn.modules.block import (
    Attention,
    Bottleneck,
    C2f,
    C2PSA,
    C3k,
    C3k2,
    SPPF,
)
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.modules.head import Detect

from channel_plan import ChannelPlan, set_plans

__all__ = [
    "plan_conv",
    "plan_bottleneck",
    "plan_c2f",
    "plan_c3k",
    "plan_sppf",
    "plan_adapter_chain",
    "plan_frozen_block",
    "plan_detect",
    "plan_model",
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
    if isinstance(sub, C3k):  # before Bottleneck: C3k is not a Bottleneck, but be explicit
        return plan_c3k(sub, in_plan)
    if isinstance(sub, Bottleneck):
        return plan_bottleneck(sub, in_plan)
    if isinstance(sub, nn.Sequential):
        raise PlanError(
            "Sequential(.m) implies an attention block (PSABlock) — M7/M8, "
            "left frozen at width 1.0 by design"
        )
    raise PlanError(f"unhandled submodule type {type(sub).__name__}")


# --- M3 ---------------------------------------------------------------------


def plan_c3k(blk: C3k, in_plan: ChannelPlan) -> ChannelPlan:
    """C3 / C3k:  `cv3(cat((m(cv1(x)), cv2(x)), 1))`

    Two parallel branches read the SAME input; cv3 consumes their concat, so
    cv3's in_plan is two segments — m's output first, then cv2's — matching the
    `cat` argument order. The inner Bottlenecks have e=1.0 (hidden == c_) and
    add=True, so each ties its output back to c_.
    """
    c_ = blk.cv1.conv.out_channels
    hidden = ChannelPlan((c_,))

    set_plans(blk.cv1, in_plan, hidden)
    cur = hidden
    for sub in blk.m:  # nn.Sequential of Bottlenecks
        if not isinstance(sub, Bottleneck):
            raise PlanError(f"unexpected {type(sub).__name__} inside C3k.m")
        cur = plan_bottleneck(sub, cur)

    set_plans(blk.cv2, in_plan, hidden)

    cv3_in = ChannelPlan.cat([cur, hidden])  # (m-branch, cv2-branch)
    if blk.cv3.conv.in_channels != cv3_in.total:
        raise PlanError(
            f"cv3 in_channels {blk.cv3.conv.in_channels} != {cv3_in.total}"
        )
    out_plan = ChannelPlan((blk.cv3.conv.out_channels,))
    set_plans(blk.cv3, cv3_in, out_plan)
    return out_plan


# --- M4 ---------------------------------------------------------------------


def plan_sppf(blk: SPPF, in_plan: ChannelPlan) -> ChannelPlan:
    """SPPF:

        y = [cv1(x)];  y.extend(m(y[-1]) for _ in range(n))
        y = cv2(cat(y, 1));  return y + x if add else y

    MaxPool2d is per-channel, so all n+1 tensors share cv1's layout — cv2's
    input is that layout REPEATED n+1 times. When `add` is set the trailing
    residual forces cv2's output layout to equal the block's input layout,
    which is the tightest constraint in the network.
    """
    hidden = ChannelPlan((blk.cv1.conv.out_channels,))
    set_plans(blk.cv1, in_plan, hidden)

    n_rep = int(getattr(blk, "n", 3)) + 1
    cv2_in = ChannelPlan.repeat(hidden, n_rep)
    if blk.cv2.conv.in_channels != cv2_in.total:
        raise PlanError(
            f"cv2 in_channels {blk.cv2.conv.in_channels} != {n_rep} repeats of "
            f"{hidden.total} = {cv2_in.total}"
        )

    if getattr(blk, "add", False):
        if blk.cv2.conv.out_channels != in_plan.total:
            raise PlanError(
                f"SPPF residual needs cv2 out {blk.cv2.conv.out_channels} == "
                f"block in {in_plan.total}"
            )
        out_plan = in_plan  # identical layout => `y + x` stays aligned
    else:
        out_plan = ChannelPlan((blk.cv2.conv.out_channels,))
    set_plans(blk.cv2, cv2_in, out_plan)
    return out_plan


# --- frozen-block boundaries (M5/M6 support) --------------------------------


def _convs_in_order(mod: nn.Module) -> list[Conv]:
    return [m for m in mod.modules() if isinstance(m, Conv)]


def _block_out_channels(block: nn.Module) -> int:
    cv2 = getattr(block, "cv2", None)
    if isinstance(cv2, Conv):
        return cv2.conv.out_channels
    convs = _convs_in_order(block)
    if not convs:
        raise PlanError(f"cannot determine out channels of {type(block).__name__}")
    return convs[-1].conv.out_channels


def plan_adapter_chain(mod: nn.Module, in_plan: ChannelPlan) -> None:
    """Let a FROZEN sub-network accept a sliced input.

    A frozen block's convs expect full `in_channels`, so a sliced tensor would
    be a shape error. We therefore plan only the leading convs:

      * depth-wise convs tie out to in, so they pass the sliced layout through
      * the first `groups == 1` conv is the ADAPTER: sliced input columns,
        FULL output — restoring full width for everything downstream
      * everything after the adapter stays unplanned (full width, stock path)

    Detect's `cv3` branches begin with a depth-wise conv, which is exactly why
    the pass-through case is needed rather than just adapting the first conv.
    """
    cur = in_plan
    for conv in _convs_in_order(mod):
        c = conv.conv
        if c.groups == 1:
            set_plans(conv, cur, ChannelPlan.frozen(c.out_channels))
            return
        if c.groups == c.in_channels == c.out_channels:
            set_plans(conv, cur, cur)  # depth-wise pass-through
            continue
        raise PlanError(
            f"cannot adapt grouped conv groups={c.groups} in={c.in_channels} "
            f"out={c.out_channels}"
        )
    raise PlanError(
        f"{type(mod).__name__}: no groups==1 conv to act as boundary adapter"
    )


def plan_frozen_block(block: nn.Module, in_plan: ChannelPlan) -> ChannelPlan:
    """Frozen block (attention): adapt its input, emit a frozen output plan."""
    plan_adapter_chain(block, in_plan)
    return ChannelPlan.frozen(_block_out_channels(block))


def plan_detect(det: Detect, source_plans: list[ChannelPlan]) -> None:
    """Detect: adapt each per-scale branch input; outputs are fixed (nc, reg*4).

    Detect has up to four parallel branch families (`cv2`, `cv3`, and the
    end-to-end `one2one_*` twins), each a ModuleList with one entry per scale,
    and each entry reads a DIFFERENT neck tensor — so there is no single input
    plan for the head.
    """
    for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branches = getattr(det, attr, None)
        if branches is None:
            continue
        if len(branches) != len(source_plans):
            raise PlanError(
                f"Detect.{attr} has {len(branches)} branches but got "
                f"{len(source_plans)} source plans"
            )
        for seq, src_plan in zip(branches, source_plans):
            plan_adapter_chain(seq, src_plan)


def _has_attention(mod: nn.Module) -> bool:
    return any(isinstance(s, Attention) for s in mod.modules())


# --- M5/M6: whole-graph walk ------------------------------------------------


def plan_model(model: nn.Module, verbose: bool = False) -> list:
    """Walk `model.model` in forward order and plan every layer.

    Returns a list of per-layer output plans (None for the terminal head).
    Attention-bearing layers are left frozen at width 1.0, with a boundary
    adapter so their neighbours can still shrink.
    """
    top = model.model
    plans: list = [None] * len(top)
    rgb = ChannelPlan((3,), (False,))  # the image never shrinks

    def src_indices(i: int, f) -> list[int]:
        return [(i - 1 if s == -1 else s) for s in (f if isinstance(f, list) else [f])]

    for i, L in enumerate(top):
        f = getattr(L, "f", -1)
        srcs = src_indices(i, f)
        in_plans = [rgb if i == 0 and s < 0 else plans[s] for s in srcs]
        if any(p is None for p in in_plans) and not isinstance(L, Detect):
            if i == 0:
                in_plans = [rgb]
            else:
                raise PlanError(f"layer {i} ({type(L).__name__}): unresolved source plan")

        if isinstance(L, Detect):
            plan_detect(L, in_plans)
            plans[i] = None
        elif isinstance(L, Concat):
            plans[i] = ChannelPlan.cat(in_plans)
        elif isinstance(L, nn.Upsample):
            plans[i] = in_plans[0]
        elif isinstance(L, (C2PSA,)) or _has_attention(L):
            plans[i] = plan_frozen_block(L, in_plans[0])
        elif isinstance(L, SPPF):
            plans[i] = plan_sppf(L, in_plans[0])
        elif isinstance(L, (C2f, C3k2)):
            plans[i] = plan_c2f(L, in_plans[0])
        elif isinstance(L, Conv):
            plans[i] = plan_conv(L, in_plans[0])
        else:
            raise PlanError(f"layer {i}: unhandled block type {type(L).__name__}")

        if verbose:
            p = plans[i]
            desc = "terminal" if p is None else (
                f"groups={p.groups} elastic={p.elastic}"
            )
            print(f"  L{i:2d} {type(L).__name__:10s} f={str(f):12s} {desc}")
    return plans
