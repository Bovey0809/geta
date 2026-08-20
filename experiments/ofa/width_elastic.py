"""Width-elastic OFA scaffold for YOLO26 (target: n<->s pair).

Motivation: yolo26 n and s share depth=0.50; only width differs (n=0.25, s=0.50).
So a supernet spanning n<->s must be WIDTH-elastic, not depth-elastic. Prior
depth-elastic attempt on yolo26l failed (d=1 -> 0.0 mAP) for a reason that does
not apply to n/s at all: at depth=0.50 there's only one inner bottleneck per
C3k2, so there's nothing to remove along the depth axis.

Mechanism (single global scalar `_active_width` in (0, 1] on each Conv):
  * Conv.forward reads its own `_active_width` (defaults to 1.0).
  * At w=1.0 the forward is bit-identical to the unpatched module.
  * At w<1.0 the conv weight is sliced `[:out_k, :in_k]` on out-channels
    (`out_k = round(out_full * w)`) and on in-channels to match the actual
    incoming `x.shape[1]`. BN weight/bias/running_mean/running_var are
    sliced `[:out_k]`. All slices are views on the fully-trained parameters,
    so gradients flow back to the shared weights (this is the OFA property).

The Concat/chunk operations in C2f/C3k2 are shape-preserving under global
width slicing: 2*self.c -> chunk(2, 1) still splits evenly, and residual adds
still line up because both sides slice by the same fraction.

The Detect head has fixed output dims (nc + reg*4) and must NOT slice its
output channels. Apply `set_width(...)` in a way that leaves every Conv inside
`Detect` at `_active_width=1.0`; the first Conv inside Detect still handles
sliced neck input correctly because in-slicing follows `x.shape[1]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Attention, C2PSA
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.modules.head import Detect  # base class covers DetectEnd2End too


# ---- Elastic forward for Conv (Ultralytics-style: conv + bn + act, no bias on conv) ----


def _sliced_in_weight(self: Conv, out_k: int, x_channels: int) -> torch.Tensor:
    """Build the input-column slice of self.conv.weight.

    Standard case: contiguous `weight[:out_k, :x_channels]`.
    Concat-fed case: `_concat_ref` on self points to a Concat that recorded
    per-source runtime channel counts during its own forward. We slice
    `weight[:out_k, off_i : off_i + runtime_i]` per source and concatenate
    along dim=1 so the columns align with the trained per-source layout.
    """
    conv = self.conv
    concat_ref = getattr(self, "_concat_ref", None)
    if concat_ref is None:
        return conv.weight[:out_k, :x_channels]

    runtime_sizes = concat_ref._last_source_sizes
    offsets = self._source_offsets
    full_sizes = self._source_full_sizes
    assert sum(runtime_sizes) == x_channels, (
        f"Concat runtime size mismatch: sum(runtime)={sum(runtime_sizes)} vs x={x_channels}"
    )
    cols = []
    for off, full, runtime in zip(offsets, full_sizes, runtime_sizes):
        assert runtime <= full, f"source runtime {runtime} > full {full}"
        cols.append(conv.weight[:out_k, off : off + runtime])
    return torch.cat(cols, dim=1)


def _elastic_conv_forward(self: Conv, x: torch.Tensor) -> torch.Tensor:
    w = getattr(self, "_active_width", 1.0)
    conv: nn.Conv2d = self.conv
    bn: nn.BatchNorm2d = self.bn
    out_full = conv.out_channels
    in_k = x.shape[1]  # actual runtime input channels
    out_k = out_full if w >= 1.0 else max(1, int(round(out_full * w)))

    # Fast path: no slicing whatsoever (both dims match trained weights) AND
    # no Concat re-alignment needed.
    concat_ref = getattr(self, "_concat_ref", None)
    if (
        in_k == conv.in_channels
        and out_k == out_full
        and conv.groups == 1
        and concat_ref is None
    ):
        return self.act(bn(conv(x)))

    if conv.groups == 1:
        weight = _sliced_in_weight(self, out_k, in_k)
        groups = 1
    elif conv.groups == conv.in_channels and conv.groups == out_full:
        # Depth-wise: weight shape [C, 1, kH, kW]; groups = channels. For DW,
        # out is structurally determined by in (they must match), so ignore
        # the requested out_k and use in_k as both.
        out_k = in_k
        weight = conv.weight[:out_k]
        groups = out_k
    else:
        raise NotImplementedError(
            f"Unsupported conv groups={conv.groups} in={conv.in_channels} out={conv.out_channels}"
        )

    y = F.conv2d(x, weight, None, conv.stride, conv.padding, conv.dilation, groups)
    y = F.batch_norm(
        y,
        bn.running_mean[:out_k] if bn.running_mean is not None else None,
        bn.running_var[:out_k] if bn.running_var is not None else None,
        bn.weight[:out_k],
        bn.bias[:out_k],
        training=bn.training,
        momentum=bn.momentum,
        eps=bn.eps,
    )
    return self.act(y)


def _elastic_conv_forward_fuse(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """Elastic version of Conv.forward_fuse (used after model.fuse()).

    After fusion, BN scale/shift are folded into conv.weight/conv.bias, and
    `self.bn` is nn.Identity. Slicing the fused weight/bias by first-k gives
    the correctly-fused first-k channels; no BN work needed.
    """
    w = getattr(self, "_active_width", 1.0)
    conv: nn.Conv2d = self.conv
    out_full = conv.out_channels
    in_k = x.shape[1]
    out_k = out_full if w >= 1.0 else max(1, int(round(out_full * w)))

    concat_ref = getattr(self, "_concat_ref", None)
    if (
        in_k == conv.in_channels
        and out_k == out_full
        and conv.groups == 1
        and concat_ref is None
    ):
        return self.act(conv(x))

    if conv.groups == 1:
        weight = _sliced_in_weight(self, out_k, in_k)
        bias = conv.bias[:out_k] if conv.bias is not None else None
        groups = 1
    elif conv.groups == conv.in_channels and conv.groups == out_full:
        out_k = in_k
        weight = conv.weight[:out_k]
        bias = conv.bias[:out_k] if conv.bias is not None else None
        groups = out_k
    else:
        raise NotImplementedError(
            f"Unsupported conv groups={conv.groups} in={conv.in_channels} out={conv.out_channels}"
        )

    y = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, groups)
    return self.act(y)


# Apply once at import time. Class-level patch — every Conv instance uses it,
# including instances where .forward has been swapped to .forward_fuse.
Conv.forward = _elastic_conv_forward
Conv.forward_fuse = _elastic_conv_forward_fuse


# ---- Elastic forward for C2PSA ----
# Stock: `a, b = self.cv1(x).split((self.c, self.c), dim=1)` — `self.c` is
# frozen at init and stops matching cv1's runtime output under width-slicing.
# chunk(2, 1) is width-adaptive (both halves are equal by construction).


def _elastic_c2psa_forward(self: C2PSA, x: torch.Tensor) -> torch.Tensor:
    a, b = self.cv1(x).chunk(2, dim=1)
    b = self.m(b)
    return self.cv2(torch.cat((a, b), 1))


C2PSA.forward = _elastic_c2psa_forward


# ---- Concat.forward records runtime per-source channel counts ----
# The downstream Conv reads these to slice its weight columns per-source
# instead of contiguously (which mixes source-A columns into source-B territory
# whenever both sources are elastic — e.g. yolo26s model.16/19).


def _elastic_concat_forward(self: Concat, x_list):
    self._last_source_sizes = tuple(t.shape[1] for t in x_list)
    return torch.cat(x_list, self.d)


Concat.forward = _elastic_concat_forward


# ---- Helpers ----


def _iter_convs(model: nn.Module):
    for m in model.modules():
        if isinstance(m, Conv):
            yield m


def _module_contains_attention(m: nn.Module) -> bool:
    for sub in m.modules():
        if isinstance(sub, Attention):
            return True
    return False


def _frozen_conv_ids(model: nn.Module):
    """Convs whose output width must stay 1.0.

    - Inside Detect: fixed output dims (nc + reg*4).
    - Inside any TOP-LEVEL model[i] block that contains an Attention module:
      hardcoded num_heads/key_dim/head_dim at init don't survive width slicing.
      Catches both C2PSA (model.10) and C3k2(attn=True) (model.22).
      Same class of exclusion GETA needed for these blocks.
    """
    out = set()
    for m in model.modules():
        if isinstance(m, Detect):
            for sub in m.modules():
                if isinstance(sub, Conv):
                    out.add(id(sub))
    # `model.model` is the top-level Sequential from the yaml. Freeze whole
    # top-level entries that carry any Attention descendant.
    top = getattr(model, "model", None)
    if top is not None:
        for layer in top:
            if _module_contains_attention(layer):
                for sub in layer.modules():
                    if isinstance(sub, Conv):
                        out.add(id(sub))
    return out


def _get_layer_out_channels(layer: nn.Module):
    """Best-effort output channel count of a top-level model layer.

    Priority:
      1. `cv2` (the output conv in C2f/C3k2/C2PSA) -> its Conv2d.out_channels.
      2. A direct Conv layer's own out_channels.
      3. Last Conv found by module walk (fallback for exotic blocks).
    Never uses `self.c` (that's the hidden-channels count, not the block output).
    """
    cv2 = getattr(layer, "cv2", None)
    if isinstance(cv2, Conv):
        return cv2.conv.out_channels
    if isinstance(layer, Conv):
        return layer.conv.out_channels
    if isinstance(layer, nn.Upsample):
        return None  # pass-through; caller will inherit from source
    last_conv = None
    for m in layer.modules():
        if isinstance(m, Conv):
            last_conv = m
    if last_conv is not None:
        return last_conv.conv.out_channels
    return None


def _find_first_conv(layer: nn.Module):
    """First Conv encountered in a module (traversal order)."""
    for m in layer.modules():
        if isinstance(m, Conv):
            return m
    return None


def prepare_concat_alignment(model: nn.Module) -> int:
    """For each Concat in model.model, attach Concat-source metadata to the
    consumer layer's first internal Conv so its input-channel weight slicing
    can respect per-source boundaries.

    Idempotent: safe to call multiple times.
    Returns the number of consumer Convs successfully annotated.
    """
    top = getattr(model, "model", None)
    if top is None:
        return 0

    # First pass: resolve each layer's OUTPUT channel count. For Upsample and
    # other pass-through layers, inherit from the upstream layer they consume.
    layer_out_c = [None] * len(top)
    for i, layer in enumerate(top):
        c = _get_layer_out_channels(layer)
        if c is not None:
            layer_out_c[i] = c
            continue
        # Pass-through: use its input source's output.
        f = getattr(layer, "f", -1)
        if isinstance(f, int):
            src = i - 1 if f == -1 else f
            layer_out_c[i] = layer_out_c[src]
        else:
            # multi-input (Concat) — handled below when we hit the Concat itself
            layer_out_c[i] = None
    # For Concat layers, output = sum of source channels.
    for i, layer in enumerate(top):
        if isinstance(layer, Concat):
            f = layer.f
            srcs = [i - 1 if s == -1 else s for s in f]
            sizes = [layer_out_c[s] for s in srcs]
            if any(s is None for s in sizes):
                continue
            layer_out_c[i] = sum(sizes)

    n_annotated = 0
    for i, layer in enumerate(top):
        if not isinstance(layer, Concat):
            continue
        f = layer.f
        srcs = [i - 1 if s == -1 else s for s in f]
        source_full_sizes = [layer_out_c[s] for s in srcs]
        if any(s is None for s in source_full_sizes):
            continue
        offsets = [0]
        for sz in source_full_sizes[:-1]:
            offsets.append(offsets[-1] + sz)
        # Consumer: the next layer in the Sequential that takes this Concat
        # as its (only) input. In yolo26 that's always i+1 with .f == -1.
        if i + 1 >= len(top):
            continue
        consumer = top[i + 1]
        first_conv = _find_first_conv(consumer)
        if first_conv is None:
            continue
        first_conv._concat_ref = layer
        first_conv._source_offsets = tuple(offsets)
        first_conv._source_full_sizes = tuple(source_full_sizes)
        n_annotated += 1
    model._concat_alignment_ready = True
    return n_annotated


def set_width(model: nn.Module, w: float) -> int:
    """Set global width fraction; frozen convs (Detect, C2PSA) stay at 1.0.

    Also ensures Concat-alignment metadata is prepared (idempotent).
    """
    assert 0.0 < w <= 1.0, f"width must be in (0, 1], got {w}"
    if not getattr(model, "_concat_alignment_ready", False):
        prepare_concat_alignment(model)
    frozen = _frozen_conv_ids(model)
    n_elastic = 0
    for c in _iter_convs(model):
        if id(c) in frozen:
            c._active_width = 1.0
        else:
            c._active_width = float(w)
            n_elastic += 1
    return n_elastic


def count_active_params(model: nn.Module) -> int:
    """Effective params under the current per-Conv _active_width settings.

    Includes only Conv.conv weights and Conv.bn params (the pieces that slice).
    Other parameters (Detect internals, etc.) are counted at full size. This is
    a sanity metric, not a rigorous FLOP/param accounting.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, Conv):
            w = getattr(m, "_active_width", 1.0)
            out_full = m.conv.out_channels
            out_k = max(1, int(round(out_full * w)))
            # weight [out, in/groups, kH, kW]
            in_full = m.conv.in_channels // m.conv.groups
            # in-channel slicing depends on runtime x.shape[1] — approximate
            # by assuming the previous layer also slices by w (uniform case).
            in_k = max(1, int(round(in_full * w))) if m.conv.groups == 1 else 1
            kH, kW = m.conv.kernel_size
            total += out_k * in_k * kH * kW
            total += 2 * out_k  # bn weight + bias
        else:
            for p in getattr(m, "_parameters", {}).values():
                if p is not None:
                    total += p.numel()
    return total


if __name__ == "__main__":
    # Smoke: load yolo26s, sanity-check that w=1.0 gives full-conv output and
    # w=0.5 halves the effective conv output-channel count on a stem forward.
    from ultralytics import YOLO

    y = YOLO("yolo26s.pt")
    m = y.model.eval()
    print(f"convs: {sum(1 for _ in _iter_convs(m))}")
    print(f"detect-internal convs: {len(_detect_conv_ids(m))}")

    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        set_width(m, 1.0)
        y1 = m(x)
        set_width(m, 0.5)
        y05 = m(x)
    o1 = y1[0] if isinstance(y1, (list, tuple)) else y1
    o05 = y05[0] if isinstance(y05, (list, tuple)) else y05
    print(f"w=1.0 out={tuple(o1.shape)}")
    print(f"w=0.5 out={tuple(o05.shape)}  (should match at Detect output)")
    print("ELASTIC_WIDTH_SMOKE_DONE")
