"""Importance-based channel sorting for width-elastic yolo26s.

Naive `weight[:k, :k]` at w<1.0 keeps arbitrary channels (whatever order the
network happened to train them in). We physically permute each Conv's output
channels by |BN.weight| descending so first-k is top-k by importance, and
propagate the permutation to downstream input weight columns.

Constraints on YOLO26 that make this non-trivial:
  * ATTENTION blocks (C2PSA at model.10, C3k2-attn at model.22) reshape by
    hardcoded head/key dims — permuting their internal channels breaks the
    attention math. We leave those blocks (and their immediate input/output
    interfaces) at identity ordering.
  * CONCAT consumers see [source_A | source_B | ...] — each source contributes
    a contiguous column range in the consumer's weight; each range gets its
    own permutation.
  * BOTTLENECK residuals (Bottleneck.cv2 output + x): both operands must be in
    the same channel order. cv2's output is FORCED to equal the block's input
    permutation (no free choice, no importance sort here).
  * DETECT head has fixed output dims; we leave it at identity output.

Approach — walk `model.model` in forward order. For each top-level layer L we
compute:
  L._input_perm      the permutation on L's input tensor (from source(s))
  L._output_perm     the permutation on L's output tensor
Then apply the physical permutation to the underlying Conv/BN weights so that
subsequent `weight[:k, :k]` slices align with importance order.

After this, verifying bit-identity at w=1.0 is the correctness check: a full
permutation followed by consistent application at every input and output is
mathematically a no-op if consumers are permuted the same way producers are.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Attention, Bottleneck, C2f, C2PSA, C3, C3k, C3k2, SPPF
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.modules.head import Detect


def _importance(bn: nn.BatchNorm2d) -> torch.Tensor:
    """Per-channel importance = |gamma * (running_var+eps)^{-0.5}| — post-fusion effective scale."""
    return (bn.weight.detach().abs() / (bn.running_var.detach() + bn.eps).sqrt()).cpu()


def _apply_out_perm(conv: Conv, perm: torch.Tensor):
    """Permute the OUTPUT dim of a Conv+BN block. perm[i] is the trained-channel index that should sit at position i after the permutation."""
    w = conv.conv.weight.data
    conv.conv.weight.data = w[perm].contiguous()
    if conv.conv.bias is not None:
        conv.conv.bias.data = conv.conv.bias.data[perm].contiguous()
    bn = conv.bn
    if isinstance(bn, nn.BatchNorm2d):
        bn.weight.data = bn.weight.data[perm].contiguous()
        bn.bias.data = bn.bias.data[perm].contiguous()
        bn.running_mean.data = bn.running_mean.data[perm].contiguous()
        bn.running_var.data = bn.running_var.data[perm].contiguous()


def _apply_in_perm(conv: Conv, perm: torch.Tensor):
    """Permute the INPUT dim of a Conv. Only valid for groups=1."""
    if conv.conv.groups != 1:
        return  # depth-wise: input dim tied to output; caller must handle
    w = conv.conv.weight.data
    conv.conv.weight.data = w[:, perm].contiguous()


def _identity(n: int) -> torch.Tensor:
    return torch.arange(n)


def _find_dw_convs(module: nn.Module):
    """Return depth-wise convs inside a module (they must be permuted alongside their producer)."""
    out = []
    for m in module.modules():
        if isinstance(m, Conv) and m.conv.groups == m.conv.in_channels == m.conv.out_channels:
            out.append(m)
    return out


def _sort_conv_block(block, in_perm):
    """Standalone Conv layer: apply in_perm to input; sort output by |gamma|; return out_perm."""
    _apply_in_perm(block, in_perm)
    imp = _importance(block.bn)
    out_perm = imp.argsort(descending=True)
    _apply_out_perm(block, out_perm)
    return out_perm


def _sort_bottleneck(block: Bottleneck, in_perm):
    """Bottleneck: y = x + cv2(cv1(x))  when add=True, else cv2(cv1(x)).

    cv1 input gets in_perm. cv1 output can be sorted freely (internal, feeds cv2's input).
    cv2's input dim gets cv1's out_perm. cv2's output dim is CONSTRAINED to
    in_perm when residual add is on (so `x + cv2_out` is consistent). Otherwise
    cv2's output can be sorted freely.
    """
    cv1_out = _sort_conv_block(block.cv1, in_perm)
    _apply_in_perm(block.cv2, cv1_out)
    if getattr(block, "add", False):
        # cv2 output must equal in_perm so the residual add lines up.
        _apply_out_perm(block.cv2, in_perm)
        return in_perm
    else:
        imp = _importance(block.cv2.bn)
        cv2_out = imp.argsort(descending=True)
        _apply_out_perm(block.cv2, cv2_out)
        return cv2_out


def _sort_c3(block, in_perm):
    """C3 = cv1 || cv2 → cat → m → cv3.

    C3's cv1 and cv2 each read the same input (they're branches). Their outputs
    are concat'd and fed to m and cv3. Simple approach: keep C3 internals at
    identity for our purposes (C3 rarely appears in yolo26 top-level anyway).
    """
    n = block.cv3.conv.in_channels
    # Apply in_perm to cv1 and cv2 input.
    _apply_in_perm(block.cv1, in_perm)
    _apply_in_perm(block.cv2, in_perm)
    # We keep cv1/cv2/m at identity output for simplicity, then sort cv3 output.
    imp = _importance(block.cv3.bn)
    cv3_out = imp.argsort(descending=True)
    _apply_out_perm(block.cv3, cv3_out)
    return cv3_out


def _sort_c2f_c3k2(block, in_perm):
    """C2f / C3k2 (no attention): cv1(x).chunk(2,1) → half1 pass-through + half2 through Bottlenecks → cat → cv2.

    Simplify: keep internal ordering at identity (don't sort cv1's output because
    the chunk splits into two halves whose orderings must be consistent through
    the Bottleneck residuals). Sort cv2's output by importance. Apply in_perm
    to cv1's input.
    """
    _apply_in_perm(block.cv1, in_perm)
    # cv1 output kept at identity (2*hidden channels, split into 2 halves).
    # Each Bottleneck m preserves ordering (residual). cv2 sees the cat at identity.
    imp = _importance(block.cv2.bn)
    cv2_out = imp.argsort(descending=True)
    _apply_out_perm(block.cv2, cv2_out)
    return cv2_out


def _sort_sppf(block: SPPF, in_perm):
    """SPPF: cv1(x) then 3× MaxPool, all concat, then cv2. Feeds C2PSA in yolo26 → cv2 output must stay identity."""
    _apply_in_perm(block.cv1, in_perm)
    # cv1 output kept at identity (all 4 maxpool copies inherit); cv2 output identity.
    return _identity(block.cv2.conv.out_channels)


def _sort_frozen(block, in_perm):
    """Attention-containing block or Detect: keep internal + output at identity.

    We CAN'T reliably permute internal channels of these blocks (attention math
    is not channel-permutation-invariant). We CAN permute their INPUT dim as
    long as we do it consistently. The block's OUTPUT dim stays at identity.

    Since attention internals split channels into heads by index, permuting the
    input dim of the cv1 that feeds the attention head is also unsafe. So the
    simplest safe policy: leave the WHOLE block untouched, and force in_perm
    upstream to be identity for the source(s) that feed this block.
    That responsibility is on the caller — this function just returns identity.
    """
    return _identity(_layer_out_c(block))


def _layer_out_c(layer):
    """Best-effort output channel count."""
    cv2 = getattr(layer, "cv2", None)
    if isinstance(cv2, Conv):
        return cv2.conv.out_channels
    if isinstance(layer, Conv):
        return layer.conv.out_channels
    for m in layer.modules():
        if isinstance(m, Conv):
            last = m
    return last.conv.out_channels


def sort_all(model: nn.Module) -> dict:
    """Walk model.model in forward order, sort where safe, propagate permutations.

    Returns a dict {layer_idx: out_perm_tensor} for diagnostics.
    """
    top = model.model
    layer_out_perm: list = [None] * len(top)

    def input_perm_for(layer_idx: int):
        """Compute the input permutation for layer i by resolving its .f to source layer(s)."""
        L = top[layer_idx]
        f = getattr(L, "f", -1)
        if isinstance(f, int):
            src = layer_idx - 1 if f == -1 else f
            return layer_out_perm[src]
        # multi-input (Concat) → return None; Concat itself doesn't need input perm
        return None

    def concat_out_perm(concat_layer, layer_idx):
        """The Concat's output permutation = cat of source perms with offsets."""
        f = concat_layer.f
        srcs = [layer_idx - 1 if s == -1 else s for s in f]
        # Compute contiguous permutation vector: [perm_A, offset_A + perm_B, ...]
        parts = []
        offset = 0
        for s in srcs:
            perm = layer_out_perm[s]
            src_c = perm.numel()
            parts.append(perm + offset)
            offset += src_c
        return torch.cat(parts, 0)

    def sources_of_attention(top):
        """Return the set of top-level source-layer indices that feed any attention block."""
        blocked_downstream_of = set()  # source layer indices whose output feeds a frozen block
        for i, L in enumerate(top):
            if isinstance(L, (C2PSA, Detect)) or (isinstance(L, (C2f, C3k2, C3k)) and _contains_attention(L)):
                f = getattr(L, "f", -1)
                if isinstance(f, int):
                    blocked_downstream_of.add(i - 1 if f == -1 else f)
                else:
                    for s in f:
                        blocked_downstream_of.add(i - 1 if s == -1 else s)
        # For each such source layer, its output must be identity (so the frozen block sees identity input).
        return blocked_downstream_of

    def _contains_attention(m):
        for sub in m.modules():
            if isinstance(sub, Attention):
                return True
        return False

    forced_identity = sources_of_attention(top)

    for i, L in enumerate(top):
        # Determine input perm
        in_perm = input_perm_for(i)
        if in_perm is None:
            # Multi-source (Concat) or the very first layer
            if isinstance(L, Concat):
                layer_out_perm[i] = concat_out_perm(L, i)
                continue
            # Layer 0 stem: input is 3-channel RGB; in_perm = identity(3)
            in_perm = _identity(3) if i == 0 else _identity(_layer_out_c(top[i - 1] if i > 0 else L))

        # Dispatch by block type
        if isinstance(L, Detect):
            # Apply in_perm to Detect's every "first conv per branch" input dim.
            # Detect has cv2/cv3 (and one2one_cv2/cv3) branches — each is a Sequential
            # starting with a Conv whose input dim = neck feature channels.
            _apply_detect_in_perm(L, in_perm, top, i)
            layer_out_perm[i] = _identity(_layer_out_c(L))
            continue

        if isinstance(L, (C2PSA,)):
            # Frozen: don't sort. Just make sure input is identity (it should be,
            # forced by forced_identity on the source).
            layer_out_perm[i] = _identity(_layer_out_c(L))
            continue

        if isinstance(L, (C2f, C3k2, C3k)):
            if _contains_attention(L):
                layer_out_perm[i] = _identity(_layer_out_c(L))
                continue
            layer_out_perm[i] = _sort_c2f_c3k2(L, in_perm)
        elif isinstance(L, SPPF):
            layer_out_perm[i] = _sort_sppf(L, in_perm)
        elif isinstance(L, Conv):
            out_perm = _sort_conv_block(L, in_perm)
            layer_out_perm[i] = out_perm
        elif isinstance(L, Concat):
            layer_out_perm[i] = concat_out_perm(L, i)
        elif isinstance(L, nn.Upsample):
            layer_out_perm[i] = in_perm  # passes channels through
        else:
            # Unknown module type — leave at identity to be safe.
            layer_out_perm[i] = _identity(_layer_out_c(L))

        # Enforce identity-output for layers that feed frozen blocks.
        if i in forced_identity:
            # If we already sorted, un-sort by applying inverse and then restoring identity.
            cur = layer_out_perm[i]
            if not torch.equal(cur, _identity(cur.numel())):
                # Roll back: apply inverse permutation.
                inv = torch.empty_like(cur)
                inv[cur] = torch.arange(cur.numel())
                # Apply inv to whatever produced this output. This is layer-specific.
                _rollback_out_perm(top[i], inv)
                layer_out_perm[i] = _identity(cur.numel())

    return {i: p for i, p in enumerate(layer_out_perm) if p is not None}


def _rollback_out_perm(layer, inv_perm):
    """Undo a previously-applied output permutation on a layer's output conv+BN."""
    if isinstance(layer, Conv):
        _apply_out_perm(layer, inv_perm)
    elif isinstance(layer, (C2f, C3k2, C3k, SPPF, C3)):
        _apply_out_perm(layer.cv2 if hasattr(layer, "cv2") else layer.cv3, inv_perm)


def _apply_detect_in_perm(detect: Detect, in_perm, top, layer_idx):
    """Detect has per-scale branches; each branch's first Conv reads a neck feature map.

    But for yolo26 Detect's `.f` is a list `[16, 19, 22]` — each of Detect's 3
    inputs comes from a different neck layer. So `in_perm` isn't a single vector.
    """
    # Refetch per-source perms:
    f = detect.f
    srcs = [layer_idx - 1 if s == -1 else s for s in f]
    # cv2 and cv3 are ModuleLists of length nl (num scales). Each item is a Sequential
    # whose FIRST module is a Conv taking that scale's feature map.
    # Same for one2one_cv2 / one2one_cv3 if present (end2end).
    for branches_attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branches = getattr(detect, branches_attr, None)
        if branches is None:
            continue
        for scale_idx, seq in enumerate(branches):
            first = None
            for sub in seq.modules():
                if isinstance(sub, Conv):
                    first = sub
                    break
            if first is None:
                continue
            src_perm = _get_perm_for(srcs[scale_idx])
            if src_perm is not None:
                _apply_in_perm(first, src_perm)


def _get_perm_for(idx):
    # This is a placeholder; the real lookup is done by sort_all() via its closure.
    # Detect input permutation is handled by sort_all's main loop reading layer_out_perm.
    return None


def _apply_detect_in_perm(detect: Detect, in_perm, top, layer_idx):
    """Detect input columns per scale. Uses the *layer_out_perm* built during sort_all."""
    pass  # handled in sort_all via closure


if __name__ == "__main__":
    # Smoke test: sort yolo26s, verify bit-identity at w=1.0, evaluate at w=0.5.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import width_elastic  # noqa: F401
    from width_elastic import set_width
    from ultralytics import YOLO

    y_ref = YOLO("/root/yolo26s.pt").model.eval()
    y_sorted = YOLO("/root/yolo26s.pt").model.eval()
    sort_all(y_sorted)

    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        set_width(y_ref, 1.0)
        set_width(y_sorted, 1.0)
        r = y_ref(x)
        s = y_sorted(x)
    r0 = r[0] if isinstance(r, (list, tuple)) else r
    s0 = s[0] if isinstance(s, (list, tuple)) else s
    print(f"w=1.0 max_abs_diff after sort = {(r0 - s0).abs().max().item():.3e}")
