"""Block-level depth elasticity: drop whole C3k blocks from a C3k2's `.m` list.

WHY THIS AXIS
-------------
C3k *inner*-bottleneck depth trains beautifully (~80 % of the gap recovered in
one short run, +0.24/+0.27) but saves only 5.2 % / 12.7 % of MACs, so the trained
sub-nets are heavily Pareto-dominated. The diagnosis was not "depth doesn't
work" but "that dimension is too small a share of compute". This is the same
axis applied one level up, where the compute actually lives.

STRUCTURALLY DIFFERENT FROM INNER DEPTH
---------------------------------------
`C2f.forward` is:

    y = list(cv1(x).chunk(2, 1))          # 2 segments
    y.extend(m(y[-1]) for m in self.m)    # + n segments
    return cv2(torch.cat(y, 1))           # cv2 in = (2 + n) * c

Inner-bottleneck depth is *sequential*, so it preserves shapes. Dropping an item
from `.m` removes one CONCATENATED SEGMENT, so cv2's input shrinks by c channels.
That is a change in the group COUNT, not in each group's size — hence
`ChannelPlan.select(..., n_groups=)` and the `_ofa_in_groups` limit on cv2.

WHERE IT EXISTS
---------------
`n` comes from the yaml repeats scaled by the depth multiplier, so
`n = max(round(repeats * depth), 1)`:
  * yolo26s (depth 0.50): every C3k2 has n=1 -> **nothing to drop**
  * yolo26l (depth 1.00): the 7 blocks with `repeats=2` (L2, L4, L6, L8, L13,
    L16, L19) have n=2 -> each can drop to 1, removing 7 whole sub-blocks.
    L22 has repeats=1 (n=1), so the attention block is untouched — convenient,
    since it is the one block whose internals resist slicing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f

from channel_plan import _IN_GROUPS, _OUT_PLAN

__all__ = ["install_block_depth", "set_block_depth", "block_depth_report"]


def _elastic_c2f_block_forward(self: C2f, x: torch.Tensor) -> torch.Tensor:
    """C2f/C3k2 forward running only the first `active_m` items of `.m`."""
    d = getattr(self, "active_m", len(self.m))
    y = list(self.cv1(x).chunk(2, 1))
    for i, m in enumerate(self.m):
        if i >= d:
            break
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


_INSTALLED = False


def install_block_depth() -> None:
    """Patch C2f.forward (inherited by C3k2) to honour `active_m`."""
    global _INSTALLED
    if _INSTALLED:
        return
    C2f.forward = _elastic_c2f_block_forward
    _INSTALLED = True


def set_block_depth(model: nn.Module, keep: int) -> tuple[int, int]:
    """Keep the first `keep` items of every C3k2 `.m` that has more than one.

    Also sets the `_ofa_in_groups` limit on that block's cv2, so its input
    columns select only the (2 + keep) segments still being concatenated.
    Blocks with n=1 are left alone (nothing to drop, and that includes the
    attention block on yolo26l).

    Returns (blocks touched, sub-blocks dropped).
    """
    touched = dropped = 0
    for m in model.modules():
        if not isinstance(m, C2f):
            continue
        n = len(m.m)
        if n <= 1:
            continue
        d = max(1, min(int(keep), n))
        m.active_m = d
        cv2 = m.cv2
        if hasattr(cv2, _OUT_PLAN):
            if d == n:
                if hasattr(cv2, _IN_GROUPS):
                    delattr(cv2, _IN_GROUPS)
            else:
                setattr(cv2, _IN_GROUPS, 2 + d)
        touched += 1
        dropped += n - d
    return touched, dropped


def block_depth_report(model: nn.Module) -> list[tuple[str, int]]:
    """(module type, len(.m)) for every C2f/C3k2, to show where headroom is."""
    out = []
    for i, L in enumerate(getattr(model, "model", [])):
        for sub in L.modules():
            if isinstance(sub, C2f):
                out.append((f"L{i} {type(sub).__name__}", len(sub.m)))
                break
    return out
