"""The elastic==narrow oracle — the correctness check that was missing.

CLAIM UNDER TEST
----------------
A correctly width-sliced elastic module is not merely "shape-valid at w<1"; it
is *definitionally the same computation* as a genuinely narrow module holding
the corresponding gathered weights. So:

    elastic(wide_module, w)(x_narrow)  ==  narrow_module_with_gathered_weights(x_narrow)

exactly (up to fp round-off from a different cuDNN/BLAS path, hence a tight
atol rather than bit-equality).

This is the assertion the previous attempt never had. Its absence let two
structural bugs hide behind a single end-to-end mAP number for an entire study:
shapes worked out at every width, so nothing raised, and the only symptom was
"0.0 mAP" — which was misread as a capacity ceiling.

Run:  python -m experiments.ofa.tests.test_oracle
  or: python experiments/ofa/tests/test_oracle.py
CPU-only, no weights, no COCO, seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # experiments/ofa

from channel_plan import ChannelPlan, install_elastic_conv, set_plans, set_width  # noqa: E402

from ultralytics.nn.modules.conv import Conv  # noqa: E402

install_elastic_conv()

WIDTHS = (1.0, 0.875, 0.75, 0.625, 0.5)
ATOL = 1e-5

_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSES
    if cond:
        _PASSES += 1
        print(f"  PASS  {name}")
    else:
        _FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def copy_conv_slice(
    wide: Conv,
    narrow: Conv,
    out_sel: torch.Tensor,
    in_sel: torch.Tensor | None,
) -> None:
    """Copy the gathered [out_sel, in_sel] sub-tensor of `wide` into `narrow`."""
    wc, nc = wide.conv, narrow.conv
    if wc.groups == 1:
        assert in_sel is not None
        nc.weight.data.copy_(wc.weight.data[out_sel][:, in_sel])
    else:
        nc.weight.data.copy_(wc.weight.data[out_sel])
    if wc.bias is not None and nc.bias is not None:
        nc.bias.data.copy_(wc.bias.data[out_sel])
    if isinstance(wide.bn, nn.BatchNorm2d) and isinstance(narrow.bn, nn.BatchNorm2d):
        narrow.bn.weight.data.copy_(wide.bn.weight.data[out_sel])
        narrow.bn.bias.data.copy_(wide.bn.bias.data[out_sel])
        narrow.bn.running_mean.data.copy_(wide.bn.running_mean.data[out_sel])
        narrow.bn.running_var.data.copy_(wide.bn.running_var.data[out_sel])
        narrow.bn.eps = wide.bn.eps


def randomize_bn(mod: nn.Module) -> None:
    """Give BN non-trivial affine params and running stats.

    Fresh BN has weight=1, bias=0, mean=0, var=1 — which makes several wrong
    slicings look right. Randomizing removes that false pass.
    """
    for m in mod.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.25)
            m.bias.data.normal_(0.0, 0.25)
            m.running_mean.data.normal_(0.0, 0.5)
            m.running_var.data.uniform_(0.5, 1.5)


# ---------------------------------------------------------------------------
# M1 — plain Conv
# ---------------------------------------------------------------------------


def test_m1_plain_conv() -> None:
    print("\n[M1] plain Conv (yolo26s layers 0,1,3,5,7,17,20)")
    torch.manual_seed(0)
    cases = [
        # (c_in, c_out, k, s) — mirrors the real stem/downsample shapes
        (3, 32, 3, 2),
        (32, 64, 3, 2),
        (128, 128, 3, 2),
        (256, 256, 3, 2),
    ]
    for c_in, c_out, k, s in cases:
        wide = Conv(c_in, c_out, k, s).eval()
        randomize_bn(wide)
        in_plan = ChannelPlan((c_in,))
        out_plan = ChannelPlan((c_out,))
        set_plans(wide, in_plan, out_plan)

        for w in WIDTHS:
            in_sel = in_plan.select(w)
            out_sel = out_plan.select(w)
            narrow = Conv(in_sel.numel(), out_sel.numel(), k, s).eval()
            copy_conv_slice(wide, narrow, out_sel, in_sel)

            x_full = torch.randn(2, c_in, 32, 32)
            x = x_full[:, in_sel]

            set_width(wide, w)
            with torch.no_grad():
                got = wide(x)
                want = narrow(x)
            d = (got - want).abs().max().item()
            check(
                f"Conv({c_in}->{c_out},k{k},s{s}) w={w}",
                d < ATOL,
                f"max|diff|={d:.3e}",
            )


# ---------------------------------------------------------------------------
# ChannelPlan unit behaviour
# ---------------------------------------------------------------------------


def test_plan_semantics() -> None:
    print("\n[plan] ChannelPlan selection semantics")
    p = ChannelPlan((128, 128))  # a C3k2 cv1 output: two branches of c=128
    sel = p.select(0.75)
    check("two-group plan keeps per-group prefixes",
          sel.tolist() == list(range(0, 96)) + list(range(128, 224)),
          f"got {sel[:3].tolist()}..{sel[-3:].tolist()}")
    check("two-group total at w=0.75 is 192", p.active(0.75) == 192,
          f"got {p.active(0.75)}")
    check("multi-group selection is NOT a contiguous prefix",
          not p.is_contiguous_at(0.75))

    # This is precisely the old bug: a contiguous prefix of the same length
    # crosses the semantic boundary at index c.
    contiguous = list(range(p.active(0.75)))
    check("contiguous prefix differs from correct selection (the old bug)",
          contiguous != sel.tolist())
    crossed = [i for i in contiguous if i >= 128]
    check("old contiguous prefix would pull from branch-2 territory",
          len(crossed) > 0, f"{len(crossed)} indices >=128")

    single = ChannelPlan((256,))
    check("single-group plan IS contiguous (why plain Convs never broke)",
          single.is_contiguous_at(0.75))
    check("cat composes groups",
          (ChannelPlan((64,)) + ChannelPlan((32, 32))).groups == (64, 32, 32))
    check("repeat models SPPF's repeated concat",
          ChannelPlan.repeat(ChannelPlan((256,)), 4).groups == (256,) * 4)
    check("uniform splits evenly",
          ChannelPlan.uniform(256, 2).groups == (128, 128))
    check("equal groups => identical selection (residual ties)",
          torch.equal(ChannelPlan((64,)).select(0.5), ChannelPlan((64,)).select(0.5)))


def test_invariant_fires() -> None:
    """The runtime guard must reject a producer/consumer plan mismatch."""
    print("\n[guard] channel-structure mismatch is loud, not silent")
    torch.manual_seed(0)
    conv = Conv(256, 128, 1, 1).eval()
    # Declare a two-group input plan, then feed a contiguously-sliced tensor of
    # the size the OLD code would have produced. Sizes happen to match here, so
    # only an explicit assert can catch it... construct a real size mismatch:
    set_plans(conv, ChannelPlan((128, 128)), ChannelPlan((128,)))
    set_width(conv, 0.5)
    bad = torch.randn(1, 100, 8, 8)  # wrong channel count
    raised = False
    try:
        with torch.no_grad():
            conv(bad)
    except AssertionError:
        raised = True
    check("wrong incoming channel count raises AssertionError", raised)


def main() -> int:
    print("=" * 68)
    print("elastic == narrow oracle")
    print("=" * 68)
    test_plan_semantics()
    test_invariant_fires()
    test_m1_plain_conv()
    print("\n" + "=" * 68)
    print(f"{_PASSES} passed, {len(_FAILURES)} failed")
    for f in _FAILURES:
        print(f"  FAILED: {f}")
    print("=" * 68)
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
