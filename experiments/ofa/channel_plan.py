"""ChannelPlan — group-structured channel slicing for width-elastic YOLO26.

WHY THIS EXISTS
---------------
The previous implementation (`width_elastic.py`) sliced every Conv's output as
`weight[:round(C*w)]` — a single contiguous "take the first k" range. That is
wrong for YOLO26, because several modules impose *internal* channel structure
that a contiguous range silently violates:

  * `C2f.forward` / `C3k2` does `cv1(x).chunk(2, 1)` where cv1 emits 2c channels
    and the SEMANTIC boundary between the two branches sits at index c.
    Taking `[0, round(2c*w))` and letting `chunk` split *that* misaligns the
    boundary — at w=0.5 the residual branch is fed entirely from the *first*
    semantic half. (Proven in tests/verify_chunk_bug.py.)
  * `cv2` of the same block consumes `cat([y0, y1, ...])` — (2+n) segments of c
    each. A contiguous input-column range mixes one segment into another.
  * `SPPF` concatenates 1+n *repeated* copies of cv1's output.
  * Residual adds (`Bottleneck`, `SPPF` with add=True, `PSABlock`) require the
    output channel selection to be IDENTICAL to the input selection.

THE FIX
-------
Describe every tensor's channel layout explicitly as an ordered list of
semantic groups, and slice *within each group*:

    ChannelPlan(groups=(c, c)).select(0.75)
        -> [0..round(c*.75)) ++ [c .. c+round(c*.75))

Then `chunk(2, 1)` splits at exactly the right place by construction, with no
change to the module's forward code. Residual ties are expressed by giving both
tensors plans with equal `groups` (selection is a pure function of the groups,
so equal groups => identical indices).

INVARIANT ENFORCED AT RUNTIME
-----------------------------
A Conv's `in_plan` must have the same `groups` as its producer's `out_plan`, so
`len(in_sel)` always equals the incoming tensor's channel count. We assert that
on every forward. That assertion alone would have caught the chunk bug on the
first narrow forward instead of hiding behind an end-to-end 0.0 mAP.

NOTE ON BN BUFFERS
------------------
Group-structured selection uses advanced indexing, which returns a COPY (not a
view). This removes the second historical bug — narrow train-mode passes can no
longer write half-width statistics into the shared `running_mean`/`running_var`
(see tests/verify_bn_corruption.py). It also means running stats do NOT
accumulate through a sliced forward, so per-width BN must be handled explicitly
(plan phase P4), not incidentally. `elastic_bn_is_safe()` documents the state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = [
    "ChannelPlan",
    "set_plans",
    "set_width",
    "get_width",
    "install_elastic_conv",
]


def _round_group(n: int, w: float) -> int:
    """Active size of one group at width w. Never drops a group to zero."""
    return max(1, int(round(n * w)))


@dataclass(frozen=True)
class ChannelPlan:
    """An ordered list of semantic channel groups making up one tensor.

    `groups` are FULL (width=1.0) sizes. Selection keeps the first
    `round(g*w)` channels of each group, concatenated in group order.

    Equality is by value, which is exactly what residual ties need: two tensors
    that must stay aligned simply carry plans with equal `groups`.
    """

    groups: tuple[int, ...]

    def __post_init__(self):
        assert len(self.groups) >= 1, "a plan needs at least one group"
        assert all(isinstance(g, int) and g >= 1 for g in self.groups), (
            f"groups must be positive ints, got {self.groups}"
        )

    # -- sizes ---------------------------------------------------------------

    @property
    def total(self) -> int:
        """Full channel count at width 1.0."""
        return sum(self.groups)

    def offsets(self) -> tuple[int, ...]:
        """Start index of each group in the full tensor."""
        out, acc = [], 0
        for g in self.groups:
            out.append(acc)
            acc += g
        return tuple(out)

    def active_groups(self, w: float) -> tuple[int, ...]:
        """Per-group active sizes at width w."""
        if w >= 1.0:
            return self.groups
        return tuple(_round_group(g, w) for g in self.groups)

    def active(self, w: float) -> int:
        """Total active channel count at width w."""
        return sum(self.active_groups(w))

    # -- selection -----------------------------------------------------------

    def select(self, w: float, device=None) -> torch.Tensor:
        """Indices into the FULL tensor that survive at width w.

        Contiguous when there is a single group; otherwise a gather of the
        per-group prefixes. Returned as a LongTensor so it can index weights.
        """
        if w >= 1.0:
            return torch.arange(self.total, device=device)
        parts = [
            torch.arange(off, off + k, device=device)
            for off, k in zip(self.offsets(), self.active_groups(w))
        ]
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def is_contiguous_at(self, w: float) -> bool:
        """True iff select(w) happens to be a plain prefix range.

        Only true for single-group plans (or w>=1.0). Useful in tests to show
        that the old contiguous-slicing code was correct *only* in this case.
        """
        return len(self.groups) == 1 or w >= 1.0

    # -- composition ---------------------------------------------------------

    def __add__(self, other: "ChannelPlan") -> "ChannelPlan":
        """Concatenation: the plan of `cat([a, b], dim=1)`."""
        return ChannelPlan(self.groups + other.groups)

    @staticmethod
    def cat(plans) -> "ChannelPlan":
        groups: tuple[int, ...] = ()
        for p in plans:
            groups = groups + p.groups
        return ChannelPlan(groups)

    @staticmethod
    def repeat(plan: "ChannelPlan", times: int) -> "ChannelPlan":
        """Plan of `cat([t]*times)` where t has layout `plan` (SPPF pattern)."""
        return ChannelPlan(plan.groups * times)

    @staticmethod
    def uniform(total: int, n_groups: int = 1) -> "ChannelPlan":
        """`n_groups` equal groups summing to `total` (e.g. cv1's (c, c))."""
        assert total % n_groups == 0, f"{total} not divisible into {n_groups} groups"
        g = total // n_groups
        return ChannelPlan((g,) * n_groups)


# ---------------------------------------------------------------------------
# Elastic Conv forward driven by plans
# ---------------------------------------------------------------------------

_WIDTH_ATTR = "_ofa_width"
_IN_PLAN = "_ofa_in_plan"
_OUT_PLAN = "_ofa_out_plan"


def set_plans(conv: Conv, in_plan: ChannelPlan, out_plan: ChannelPlan) -> None:
    """Attach input/output channel layouts to a Conv."""
    assert isinstance(conv, Conv), f"expected ultralytics Conv, got {type(conv)}"
    c = conv.conv
    if c.groups == 1:
        assert in_plan.total == c.in_channels, (
            f"in_plan total {in_plan.total} != conv.in_channels {c.in_channels}"
        )
    else:
        # depth-wise: channels are tied, so both plans describe the same axis
        assert in_plan.groups == out_plan.groups, (
            f"depth-wise conv needs in_plan == out_plan, got {in_plan} vs {out_plan}"
        )
    assert out_plan.total == c.out_channels, (
        f"out_plan total {out_plan.total} != conv.out_channels {c.out_channels}"
    )
    setattr(conv, _IN_PLAN, in_plan)
    setattr(conv, _OUT_PLAN, out_plan)


def get_width(mod: nn.Module) -> float:
    return getattr(mod, _WIDTH_ATTR, 1.0)


def set_width(model: nn.Module, w: float, only: nn.Module | None = None) -> int:
    """Set the active width on every planned Conv (or just those under `only`).

    Convs without plans are left at width 1.0 — that is how we keep attention
    blocks and the Detect head fixed while the rest shrinks.
    """
    assert 0.0 < w <= 1.0, f"width must be in (0, 1], got {w}"
    scope = only if only is not None else model
    n = 0
    for m in scope.modules():
        if isinstance(m, Conv) and hasattr(m, _OUT_PLAN):
            setattr(m, _WIDTH_ATTR, float(w))
            n += 1
    return n


def _sel(plan: ChannelPlan, w: float, device) -> torch.Tensor:
    return plan.select(w, device=device)


def _elastic_forward(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """Conv.forward honouring ChannelPlan-based width slicing."""
    out_plan: ChannelPlan | None = getattr(self, _OUT_PLAN, None)
    w = getattr(self, _WIDTH_ATTR, 1.0)
    conv: nn.Conv2d = self.conv

    if out_plan is None or w >= 1.0:
        # unplanned or full width -> stock path, bit-identical by construction
        return self.act(self.bn(conv(x)))

    in_plan: ChannelPlan = getattr(self, _IN_PLAN)
    dev = conv.weight.device
    out_sel = _sel(out_plan, w, dev)

    if conv.groups == 1:
        in_sel = _sel(in_plan, w, dev)
        # The invariant that makes structural bugs loud instead of silent.
        assert x.shape[1] == in_sel.numel(), (
            f"{type(self).__name__}: incoming channels {x.shape[1]} != in_plan "
            f"selection {in_sel.numel()} at w={w}. The producer's out_plan and "
            f"this conv's in_plan disagree — a channel-structure bug."
        )
        weight = conv.weight[out_sel][:, in_sel]
        groups = 1
    elif conv.groups == conv.in_channels == conv.out_channels:
        # depth-wise: one group per channel; out is structurally tied to in
        out_sel = _sel(in_plan, w, dev)
        assert x.shape[1] == out_sel.numel(), (
            f"depth-wise {type(self).__name__}: incoming {x.shape[1]} != "
            f"selection {out_sel.numel()} at w={w}"
        )
        weight = conv.weight[out_sel]
        groups = out_sel.numel()
    else:
        raise NotImplementedError(
            f"grouped conv not supported: groups={conv.groups} "
            f"in={conv.in_channels} out={conv.out_channels}"
        )

    bias = conv.bias[out_sel] if conv.bias is not None else None
    y = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, groups)

    bn = self.bn
    if isinstance(bn, nn.BatchNorm2d):
        y = F.batch_norm(
            y,
            bn.running_mean[out_sel] if bn.running_mean is not None else None,
            bn.running_var[out_sel] if bn.running_var is not None else None,
            bn.weight[out_sel],
            bn.bias[out_sel],
            training=bn.training,
            momentum=bn.momentum if bn.momentum is not None else 0.1,
            eps=bn.eps,
        )
    else:
        y = bn(y)
    return self.act(y)


def _elastic_forward_fuse(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """Conv.forward_fuse (post-`model.fuse()`) honouring ChannelPlan slicing.

    After fusion BN is folded into conv.weight/bias and `self.bn` is gone, so we
    only slice the fused parameters. Patching this too is essential: `fuse()`
    rebinds each instance's `.forward` to `.forward_fuse`, and `model.val()`
    fuses — so patching only `forward` silently bypasses elasticity at eval.
    """
    out_plan: ChannelPlan | None = getattr(self, _OUT_PLAN, None)
    w = getattr(self, _WIDTH_ATTR, 1.0)
    conv: nn.Conv2d = self.conv

    if out_plan is None or w >= 1.0:
        return self.act(conv(x))

    in_plan: ChannelPlan = getattr(self, _IN_PLAN)
    dev = conv.weight.device
    out_sel = _sel(out_plan, w, dev)

    if conv.groups == 1:
        in_sel = _sel(in_plan, w, dev)
        assert x.shape[1] == in_sel.numel(), (
            f"{type(self).__name__} (fused): incoming {x.shape[1]} != in_plan "
            f"selection {in_sel.numel()} at w={w}"
        )
        weight = conv.weight[out_sel][:, in_sel]
        groups = 1
    elif conv.groups == conv.in_channels == conv.out_channels:
        out_sel = _sel(in_plan, w, dev)
        weight = conv.weight[out_sel]
        groups = out_sel.numel()
    else:
        raise NotImplementedError(f"grouped conv not supported: groups={conv.groups}")

    bias = conv.bias[out_sel] if conv.bias is not None else None
    return self.act(F.conv2d(x, weight, bias, conv.stride, conv.padding,
                             conv.dilation, groups))


_INSTALLED = False


def install_elastic_conv() -> None:
    """Patch Conv.forward and Conv.forward_fuse once, process-wide."""
    global _INSTALLED
    if _INSTALLED:
        return
    Conv.forward = _elastic_forward
    Conv.forward_fuse = _elastic_forward_fuse
    _INSTALLED = True


def elastic_bn_is_safe() -> str:
    """Explain the BN situation (see module docstring)."""
    return (
        "Group-structured selection returns copies, so narrow train-mode passes "
        "cannot corrupt shared BN running stats (the old contiguous-view bug). "
        "Consequently running stats also do NOT accumulate through a sliced "
        "forward: per-width BN buffers are required before training (plan P4)."
    )
