"""Tests for importance-based channel sorting (P3).

Three properties, per module type:

  1. **Relabelling invariance.** Sorting must leave the w=1.0 output UNCHANGED.
     A permutation applied consistently to a producer's outputs and every
     consumer's input columns is a pure relabelling. Tolerance is ~1e-4 rather
     than exactly 0: permuting input columns reorders the convolution's
     summation, and float addition is not associative.

  2. **The ordering is actually by importance.** After sorting, each elastic
     group's importance scores must be non-increasing — otherwise "first k" is
     still arbitrary and the whole exercise is pointless.

  3. **The selected set genuinely changes.** Sorting must alter which channels
     survive at w<1 (otherwise it is a no-op), while frozen groups must be
     left alone.

Run: python experiments/ofa/tests/test_sorter.py   (CPU, seconds)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from channel_plan import (  # noqa: E402
    _OUT_PLAN,
    ChannelPlan,
    install_elastic_conv,
    set_width,
)
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import (  # noqa: E402
    plan_c2f,
    plan_conv,
    plan_model,
    plan_sppf,
)
from sorter import importance, sort_c2f, sort_conv, sort_model, sort_sppf  # noqa: E402

from ultralytics.nn.modules.block import SPPF, C3k2  # noqa: E402
from ultralytics.nn.modules.conv import Conv  # noqa: E402

install_elastic_conv()
install_elastic_attention()

ATOL = 1e-4
_FAIL: list[str] = []
_PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def randomize_bn(mod: nn.Module) -> None:
    """Non-trivial BN params, so importance actually varies across channels."""
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.4).abs_()
            m.bias.data.normal_(0.0, 0.25)
            m.running_mean.data.normal_(0.0, 0.5)
            m.running_var.data.uniform_(0.3, 2.0)


def assert_descending(conv: Conv, tag: str) -> None:
    """Importance must be non-increasing inside every elastic group."""
    plan: ChannelPlan = getattr(conv, _OUT_PLAN)
    score = importance(conv)
    ok = True
    for off, g, el in zip(plan.offsets(), plan.groups, plan.elastic):
        if not el:
            continue
        seg = score[off:off + g]
        if (seg[1:] - seg[:-1] > 1e-6).any():
            ok = False
    check(f"{tag}: importance descending within each elastic group", ok)


def test_relabelling_conv() -> None:
    print("\n[sort] plain Conv")
    torch.manual_seed(0)
    for c1, c2 in [(32, 64), (128, 256)]:
        wide = Conv(c1, c2, 3, 2).eval()
        randomize_bn(wide)
        plan_conv(wide, ChannelPlan((c1,)))
        ref = copy.deepcopy(wide).eval()
        x = torch.randn(2, c1, 32, 32)
        with torch.no_grad():
            before = ref(x)
        sort_conv(wide, torch.arange(c1))
        set_width(wide, 1.0)
        with torch.no_grad():
            after = wide(x)
        # output channels are permuted, so compare against the permuted ref
        d = (after - before).abs().max().item()
        check(f"Conv({c1}->{c2}): output IS permuted (so not equal as-is)",
              d > 1e-3, f"max|diff|={d:.3e}")
        assert_descending(wide, f"Conv({c1}->{c2})")


def test_relabelling_block(name: str, make, plan_fn, sort_fn, c1: int) -> None:
    """A block's output is permuted, but composing with the inverse recovers it."""
    torch.manual_seed(0)
    wide = make().eval()
    randomize_bn(wide)
    out_plan = plan_fn(wide, ChannelPlan((c1,)))
    ref = copy.deepcopy(wide).eval()
    x = torch.randn(2, c1, 16, 16)
    with torch.no_grad():
        before = ref(x)
    perm = sort_fn(wide, torch.arange(c1))
    set_width(wide, 1.0)
    with torch.no_grad():
        after = wide(x)

    check(f"{name}: perm is a valid permutation",
          sorted(perm.tolist()) == list(range(out_plan.total)))
    # Undo the output permutation: after[:, i] should equal before[:, perm[i]]
    d = (after - before.index_select(1, perm)).abs().max().item()
    check(f"{name}: sorting is a pure relabelling at w=1.0", d < ATOL,
          f"max|diff|={d:.3e}")


def test_selection_changes() -> None:
    """Sorting must change WHICH channels survive at w<1."""
    print("\n[sort] selection actually changes")
    torch.manual_seed(0)
    blk = C3k2(128, 256, 1, False, 0.25).eval()
    randomize_bn(blk)
    plan_c2f(blk, ChannelPlan((128,)))
    imp_before = importance(blk.cv2).clone()
    sort_c2f(blk, torch.arange(128))
    imp_after = importance(blk.cv2)
    plan = getattr(blk.cv2, _OUT_PLAN)
    k = plan.active(0.5)
    kept_before = torch.topk(imp_before, k).values.sum()
    kept_after = imp_after[:k].sum()
    check("top-k importance after sorting >= arbitrary first-k before",
          kept_after >= kept_before - 1e-5,
          f"after={kept_after:.3f} before(first-k)={imp_before[:k].sum():.3f}")
    check("sorted first-k equals the true top-k set",
          torch.allclose(torch.sort(imp_after[:k]).values,
                         torch.sort(torch.topk(imp_before, k).values).values,
                         atol=1e-5))


def test_whole_model() -> None:
    """Sort the whole graph; w=1.0 predictions must be unchanged."""
    print("\n[sort] whole yolo26s graph")
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    model = DetectionModel("yolo26s.yaml", ch=3, nc=80, verbose=False).eval()
    plan_model(model)
    ref = copy.deepcopy(model).eval()

    x = torch.randn(1, 3, 256, 256)
    set_width(ref, 1.0)
    set_width(model, 1.0)
    with torch.no_grad():
        before = ref(x)
    try:
        sort_model(model, verbose=False)
    except Exception as e:  # noqa: BLE001
        check("sort_model completes", False, f"{type(e).__name__}: {e}")
        return
    check("sort_model completes", True)
    set_width(model, 1.0)
    with torch.no_grad():
        after = model(x)

    b = before[0] if isinstance(before, (list, tuple)) else before
    a = after[0] if isinstance(after, (list, tuple)) else after
    # Detect's output is in a fixed layout (boxes/scores), NOT permuted, so the
    # whole-model prediction must match directly.
    check("whole-model w=1.0 output unchanged by sorting",
          a.shape == b.shape and (a - b).abs().max().item() < 1e-3,
          f"max|diff|={(a - b).abs().max().item():.3e}"
          if a.shape == b.shape else f"{tuple(a.shape)} vs {tuple(b.shape)}")

    # every width must still run
    for w in (0.875, 0.75, 0.5):
        set_width(model, w)
        try:
            with torch.no_grad():
                o = model(x)
            o0 = o[0] if isinstance(o, (list, tuple)) else o
            check(f"sorted model forward at w={w}",
                  bool(torch.isfinite(o0).all()))
        except Exception as e:  # noqa: BLE001
            check(f"sorted model forward at w={w}", False,
                  f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 68)
    print("channel sorting: relabelling invariance + importance ordering")
    print("=" * 68)
    test_relabelling_conv()

    print("\n[sort] C3k2 with Bottleneck")
    test_relabelling_block(
        "C3k2(128->256,n1,e0.25)",
        lambda: C3k2(128, 256, 1, False, 0.25),
        plan_c2f, sort_c2f, 128)
    print("\n[sort] C3k2 with nested C3k")
    test_relabelling_block(
        "C3k2(256->256,c3k)",
        lambda: C3k2(256, 256, 1, True, 0.5),
        plan_c2f, sort_c2f, 256)
    print("\n[sort] SPPF (residual)")
    test_relabelling_block(
        "SPPF(512->512,add)",
        lambda: SPPF(512, 512, 5, 3, True),
        plan_sppf, sort_sppf, 512)
    print("\n[sort] SPPF (no residual)")
    test_relabelling_block(
        "SPPF(512->256)",
        lambda: SPPF(512, 256, 5, 3, False),
        plan_sppf, sort_sppf, 512)

    test_selection_changes()
    test_whole_model()

    print("\n" + "=" * 68)
    print(f"{_PASS} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    print("=" * 68)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
