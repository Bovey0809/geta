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

import types
from contextlib import contextmanager
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
    "recal_mode",
    "recalibrate",
    "clear_bn_stats",
    "bn_stats_coverage",
    "disable_fuse",
    "count_active_params",
]


def _round_group(n: int, w: float) -> int:
    """Active size of one group at width w. Never drops a group to zero."""
    return max(1, int(round(n * w)))


@dataclass(frozen=True)
class ChannelPlan:
    """An ordered list of semantic channel groups making up one tensor.

    `groups` are FULL (width=1.0) sizes. Selection keeps the first
    `round(g*w)` channels of each ELASTIC group, and all channels of each
    FROZEN group, concatenated in group order.

    Equality is by value, which is exactly what residual ties need: two tensors
    that must stay aligned simply carry plans with equal groups+flags.

    Frozen groups exist because attention blocks (C2PSA, C3k2-attn) and the
    Detect head cannot be sliced yet, so the tensors they emit stay full width
    even while their neighbours shrink. Without this, a frozen block's output
    would be sliced by its consumer and the two would disagree.
    """

    groups: tuple[int, ...]
    elastic: tuple[bool, ...] | None = None

    def __post_init__(self):
        assert len(self.groups) >= 1, "a plan needs at least one group"
        assert all(isinstance(g, int) and g >= 1 for g in self.groups), (
            f"groups must be positive ints, got {self.groups}"
        )
        if self.elastic is None:
            object.__setattr__(self, "elastic", (True,) * len(self.groups))
        assert len(self.elastic) == len(self.groups), (
            f"elastic flags {self.elastic} do not match groups {self.groups}"
        )

    # -- constructors --------------------------------------------------------

    @staticmethod
    def frozen(total: int) -> "ChannelPlan":
        """A single group that never shrinks (frozen-block output)."""
        return ChannelPlan((total,), (False,))

    @property
    def any_elastic(self) -> bool:
        return any(self.elastic)

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
        """Per-group active sizes at width w (frozen groups keep full size)."""
        if w >= 1.0:
            return self.groups
        return tuple(
            g if not el else _round_group(g, w)
            for g, el in zip(self.groups, self.elastic)
        )

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
        return ChannelPlan(self.groups + other.groups, self.elastic + other.elastic)

    @staticmethod
    def cat(plans) -> "ChannelPlan":
        groups: tuple[int, ...] = ()
        flags: tuple[bool, ...] = ()
        for p in plans:
            groups = groups + p.groups
            flags = flags + p.elastic
        return ChannelPlan(groups, flags)

    @staticmethod
    def repeat(plan: "ChannelPlan", times: int) -> "ChannelPlan":
        """Plan of `cat([t]*times)` where t has layout `plan` (SPPF pattern)."""
        return ChannelPlan(plan.groups * times, plan.elastic * times)

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
_STATS = "_ofa_bn_stats"  # dict[width_key] -> [mean, var, n_updates]

# Recalibration state. While active, planned Convs normalise with BATCH
# statistics and accumulate per-width running stats into their own store.
_RECAL = {"active": False, "momentum": None}  # None => cumulative average


def _wkey(w: float) -> float:
    return round(float(w), 6)


def _has_bn_stats(mod: nn.Module, w: float) -> bool:
    store = getattr(mod, _STATS, None)
    return store is not None and _wkey(w) in store


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


def _elastic_bn(
    owner: Conv, bn: nn.BatchNorm2d, y: torch.Tensor, out_sel: torch.Tensor, w: float
) -> torch.Tensor:
    """BatchNorm for a sliced activation, with PER-WIDTH running statistics.

    Why per-width stats are mandatory rather than a refinement: a sub-network's
    activation distribution is genuinely different from the full network's, so
    reusing the full-width `running_mean`/`running_var` mis-normalises every
    narrow forward. The previous implementation additionally *corrupted* the
    shared buffer, because `bn.running_mean[:k]` is a view that
    `F.batch_norm(training=True)` writes through. Group-structured selection
    returns copies, so that corruption is gone — but it also means stats can no
    longer accumulate implicitly. Hence an explicit store, keyed by width.
    """
    weight, bias = bn.weight[out_sel], bn.bias[out_sel]
    key = _wkey(w)

    if _RECAL["active"]:
        # Accumulate E[x] and E[x^2] rather than an average of per-batch
        # variances. Averaging batch variances UNDER-estimates the population
        # variance, because it discards the between-batch variation of the
        # mean: Var(x) = E[Var_batch] + Var(E_batch). Accumulating the second
        # moment is exact for any batch size.
        with torch.no_grad():
            bmean = y.mean(dim=(0, 2, 3))
            bsq = y.pow(2).mean(dim=(0, 2, 3))
        store = getattr(owner, _STATS, None)
        if store is None:
            store = {}
            setattr(owner, _STATS, store)
        entry = store.get(key)
        if entry is None:
            store[key] = [bmean.clone(), bsq.clone(), 1]
        elif _RECAL["momentum"] is None:
            # Cumulative average over all recal batches (the default).
            # A one-shot recal sees a FIXED set of batches, so there is no
            # reason to exponentially forget the earlier ones: an EMA with a
            # small momentum over few batches is dominated by whichever
            # batches came first. Measured cost of getting this wrong: recal
            # at w=1.0 returned 0.4509 instead of the 0.4715 baseline.
            entry[2] += 1
            n = entry[2]
            entry[0].add_((bmean - entry[0]) / n)
            entry[1].add_((bsq - entry[1]) / n)
        else:
            mom = _RECAL["momentum"]
            entry[0].mul_(1.0 - mom).add_(bmean, alpha=mom)
            entry[1].mul_(1.0 - mom).add_(bsq, alpha=mom)
            entry[2] += 1
        # normalise with batch stats, as BN train mode would
        return F.batch_norm(y, None, None, weight, bias, training=True,
                            momentum=0.0, eps=bn.eps)

    store = getattr(owner, _STATS, None)
    if store is not None and key in store:
        acc_mean, acc_sq, _ = store[key]
        rm = acc_mean
        rv = (acc_sq - acc_mean.pow(2)).clamp_min_(0.0)
    else:
        rm = bn.running_mean[out_sel] if bn.running_mean is not None else None
        rv = bn.running_var[out_sel] if bn.running_var is not None else None
    return F.batch_norm(y, rm, rv, weight, bias, training=False,
                        momentum=0.1, eps=bn.eps)


def _use_stock_path(mod: Conv, out_plan, in_plan, w: float) -> bool:
    """True when nothing needs slicing and no per-width stats apply.

    Note the recal/stats conditions: at w=1.0 we normally short-circuit to the
    stock op (that is what guarantees bit-identity), but during a
    recalibration pass — and afterwards, once w=1.0 stats exist — we must go
    through the sliced path so the recalibrated statistics are actually used.
    Otherwise the "recal at w=1.0 must reproduce the baseline" sanity gate
    would pass trivially without testing anything.
    """
    if _RECAL["active"] or _has_bn_stats(mod, w):
        return False
    if w >= 1.0:
        return True
    return not (in_plan.any_elastic or out_plan.any_elastic)


def _elastic_forward(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """Conv.forward honouring ChannelPlan-based width slicing."""
    out_plan: ChannelPlan | None = getattr(self, _OUT_PLAN, None)
    w = getattr(self, _WIDTH_ATTR, 1.0)
    conv: nn.Conv2d = self.conv

    if out_plan is None:
        return self.act(self.bn(conv(x)))

    in_plan: ChannelPlan = getattr(self, _IN_PLAN)
    if _use_stock_path(self, out_plan, in_plan, w):
        # bit-identical to the stock op by construction
        return self.act(self.bn(conv(x)))
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
        y = _elastic_bn(self, bn, y, out_sel, w)
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
    if not (in_plan.any_elastic or out_plan.any_elastic):
        return self.act(conv(x))
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


@contextmanager
def recal_mode(momentum: float | None = None):
    """Inside this block, planned Convs accumulate per-width BN statistics."""
    prev = dict(_RECAL)
    _RECAL["active"] = True
    _RECAL["momentum"] = None if momentum is None else float(momentum)
    try:
        yield
    finally:
        _RECAL.update(prev)


def clear_bn_stats(model: nn.Module, w: float | None = None) -> int:
    """Drop stored per-width stats (all widths, or just one). Returns count."""
    n = 0
    for m in model.modules():
        store = getattr(m, _STATS, None)
        if not store:
            continue
        if w is None:
            n += len(store)
            store.clear()
        elif _wkey(w) in store:
            del store[_wkey(w)]
            n += 1
    return n


def bn_stats_coverage(model: nn.Module, w: float) -> tuple[int, int]:
    """(convs with stats for this width, convs that need them)."""
    have = need = 0
    for m in model.modules():
        if not (isinstance(m, Conv) and hasattr(m, _OUT_PLAN)):
            continue
        if not isinstance(m.bn, nn.BatchNorm2d):
            continue
        need += 1
        if _has_bn_stats(m, w):
            have += 1
    return have, need


@torch.no_grad()
def recalibrate(
    model: nn.Module,
    batches,
    w: float,
    momentum: float | None = None,
    device=None,
) -> int:
    """Refit BN statistics for width `w` by forwarding `batches` (no grads).

    `batches` is any iterable of image tensors already normalised the way the
    validator normalises them. Returns the number of batches consumed.
    """
    set_width(model, w)
    clear_bn_stats(model, w)
    model.eval()  # dropout etc. off; BN behaviour is driven by recal_mode
    n = 0
    with recal_mode(momentum):
        for imgs in batches:
            if device is not None:
                imgs = imgs.to(device, non_blocking=True)
            model(imgs)
            n += 1
    return n


def disable_fuse(model: nn.Module) -> None:
    """Make `model.fuse()` a no-op.

    Ultralytics' validator fuses Conv+BN before evaluating. Fusion folds the
    ORIGINAL full-width running stats into the conv weights and deletes `bn`,
    which would silently discard every recalibrated per-width statistic and
    leave us measuring the wrong thing. Cheaper and safer to skip fusion for
    elastic evaluation than to try to fuse per width.
    """
    model.fuse = types.MethodType(lambda self, verbose=True: self, model)
    model.is_fused = types.MethodType(lambda self, thresh=10: False, model)


def count_active_params(model: nn.Module, w: float) -> int:
    """Parameter count of the sub-network actually executed at width w.

    Planned Convs contribute only their sliced weight/BN parameters; every
    other parameter (attention internals, Detect heads, unplanned convs) is
    counted in full, which is honest about the frozen blocks still costing
    their full size at every width.
    """
    counted: set[int] = set()
    total = 0
    for m in model.modules():
        if not (isinstance(m, Conv) and hasattr(m, _OUT_PLAN)):
            continue
        in_plan: ChannelPlan = getattr(m, _IN_PLAN)
        out_plan: ChannelPlan = getattr(m, _OUT_PLAN)
        c = m.conv
        if c.groups == 1:
            out_k, in_k = out_plan.active(w), in_plan.active(w)
        else:
            out_k = in_k = in_plan.active(w)
        kh, kw = c.kernel_size
        total += out_k * (in_k if c.groups == 1 else 1) * kh * kw
        counted.add(id(c.weight))
        if c.bias is not None:
            total += out_k
            counted.add(id(c.bias))
        if isinstance(m.bn, nn.BatchNorm2d):
            total += 2 * out_k  # weight + bias (running stats are buffers)
            counted.add(id(m.bn.weight))
            counted.add(id(m.bn.bias))
    for p in model.parameters():
        if id(p) not in counted:
            total += p.numel()
    return total


def elastic_bn_is_safe() -> str:
    """Explain the BN situation (see module docstring)."""
    return (
        "Group-structured selection returns copies, so narrow train-mode passes "
        "cannot corrupt shared BN running stats (the old contiguous-view bug). "
        "Consequently running stats also do NOT accumulate through a sliced "
        "forward: per-width BN buffers are required before training (plan P4)."
    )
