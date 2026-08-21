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

from channel_plan import (  # noqa: E402
    _WIDTH_ATTR,
    ChannelPlan,
    count_active_params,
    install_elastic_conv,
    set_plans,
    set_width,
)
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_c2f, plan_model, plan_sppf  # noqa: E402

from ultralytics.nn.modules.block import SPPF, C3k2  # noqa: E402
from ultralytics.nn.modules.conv import Conv  # noqa: E402

sys.path.insert(0, str(_HERE.parent))  # experiments/ofa/tests
from oracle_util import build_narrow_twin  # noqa: E402

install_elastic_conv()
install_elastic_attention()

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


def test_m2_c3k2_bottleneck() -> None:
    """C3k2 whose .m are Bottlenecks — yolo26s L2 and L4.

    This is the first module where the group structure actually bites: cv1's
    output feeds `chunk(2, 1)` and cv2's input is a 3-segment concat.
    """
    print("\n[M2] C3k2 with Bottleneck (yolo26s L2, L4)")
    torch.manual_seed(0)
    # (c1, c2, n, e) mirroring the real blocks, plus an n=2 generalisation
    cases = [
        (64, 128, 1, 0.25),   # L2
        (128, 256, 1, 0.25),  # L4
        (128, 128, 1, 0.5),   # e=0.5 shape family
        (64, 128, 2, 0.5),    # n=2: 4 concat segments
    ]
    for c1, c2, n, e in cases:
        wide = C3k2(c1, c2, n, False, e).eval()
        randomize_bn(wide)
        in_plan = ChannelPlan((c1,))
        out_plan = plan_c2f(wide, in_plan)

        tag = f"C3k2({c1}->{c2},n{n},e{e})"
        check(f"{tag} cv2 in_plan has {2 + n} segments",
              len(getattr(wide.cv2, "_ofa_in_plan").groups) == 2 + n,
              f"got {getattr(wide.cv2, '_ofa_in_plan').groups}")
        check(f"{tag} cv1 out_plan is two equal groups",
              getattr(wide.cv1, "_ofa_out_plan").groups == (wide.c, wide.c))

        for w in WIDTHS:
            twin = build_narrow_twin(wide, w)
            x = torch.randn(2, in_plan.active(w), 16, 16)
            set_width(wide, w)
            with torch.no_grad():
                got = wide(x)
                want = twin(x)
            check(f"{tag} w={w} shape",
                  got.shape == want.shape, f"{tuple(got.shape)} vs {tuple(want.shape)}")
            if got.shape == want.shape:
                d = (got - want).abs().max().item()
                check(f"{tag} w={w} equals narrow twin", d < ATOL, f"max|diff|={d:.3e}")
            check(f"{tag} w={w} out channels == out_plan.active",
                  got.shape[1] == out_plan.active(w),
                  f"{got.shape[1]} vs {out_plan.active(w)}")


def _legacy_contiguous_forward(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """The OLD (broken) slicing: one contiguous prefix, ignoring group structure."""
    w = getattr(self, _WIDTH_ATTR, 1.0)
    conv, bn = self.conv, self.bn
    if w >= 1.0:
        return self.act(bn(conv(x)))
    out_k = max(1, int(round(conv.out_channels * w)))
    in_k = x.shape[1]
    weight = conv.weight[:out_k, :in_k]
    y = torch.nn.functional.conv2d(x, weight, None, conv.stride, conv.padding,
                                   conv.dilation, 1)
    y = torch.nn.functional.batch_norm(
        y, bn.running_mean[:out_k], bn.running_var[:out_k],
        bn.weight[:out_k], bn.bias[:out_k], training=False,
        momentum=0.1, eps=bn.eps)
    return self.act(y)


def test_legacy_slicing_fails_the_oracle() -> None:
    """Regression guard: the ORIGINAL contiguous slicing must FAIL this oracle.

    If it passed, the oracle would have no teeth and would not have caught the
    bug that invalidated the previous study. It must fail on C3k2 (which has
    internal group structure) while still passing on a plain Conv (single group,
    where a contiguous prefix happens to be correct — which is exactly why the
    bug hid for so long).
    """
    print("\n[regression] the old contiguous slicing must FAIL the oracle")
    torch.manual_seed(0)
    good_forward = Conv.forward
    try:
        # -- plain Conv: legacy slicing is CORRECT here (single group) --
        conv = Conv(64, 128, 3, 2).eval()
        randomize_bn(conv)
        set_plans(conv, ChannelPlan((64,)), ChannelPlan((128,)))
        twin = build_narrow_twin(conv, 0.5)
        x = torch.randn(2, 32, 16, 16)
        Conv.forward = _legacy_contiguous_forward
        set_width(conv, 0.5)
        with torch.no_grad():
            d_conv = (conv(x) - twin(x)).abs().max().item()
        Conv.forward = good_forward
        check("legacy slicing still OK for single-group Conv (why it hid)",
              d_conv < ATOL, f"max|diff|={d_conv:.3e}")

        # -- C3k2: legacy slicing must be WRONG --
        blk = C3k2(128, 256, 1, False, 0.25).eval()
        randomize_bn(blk)
        plan_c2f(blk, ChannelPlan((128,)))
        twin2 = build_narrow_twin(blk, 0.5)
        x2 = torch.randn(2, 64, 16, 16)
        Conv.forward = _legacy_contiguous_forward
        set_width(blk, 0.5)
        with torch.no_grad():
            legacy_out = blk(x2)
            want = twin2(x2)
        Conv.forward = good_forward
        d_legacy = (legacy_out - want).abs().max().item()
        check("legacy slicing FAILS on C3k2 (the real bug, now caught)",
              d_legacy > 1e-3, f"max|diff|={d_legacy:.3e} (should be large)")

        # -- and the fixed path passes on the same block --
        set_width(blk, 0.5)
        with torch.no_grad():
            d_fixed = (blk(x2) - want).abs().max().item()
        check("ChannelPlan slicing PASSES on the same C3k2",
              d_fixed < ATOL, f"max|diff|={d_fixed:.3e}")
        print(f"        legacy err {d_legacy:.3e}  vs  fixed err {d_fixed:.3e}")
    finally:
        Conv.forward = good_forward


def test_m3_c3k2_nested_c3k() -> None:
    """C3k2 whose .m are C3k — yolo26s L6, L8, L13, L16, L19 (5 of 8 blocks)."""
    print("\n[M3] C3k2 with nested C3k (yolo26s L6, L8, L13, L16, L19)")
    torch.manual_seed(0)
    cases = [
        (256, 256, 1, 0.5),   # L6
        (512, 512, 1, 0.5),   # L8
        (768, 256, 1, 0.5),   # L13 (concat-fed)
        (512, 128, 1, 0.5),   # L16
        (384, 256, 1, 0.5),   # L19
    ]
    for c1, c2, n, e in cases:
        wide = C3k2(c1, c2, n, True, e).eval()  # c3k=True -> .m are C3k
        randomize_bn(wide)
        inner = wide.m[0]
        check(f"C3k2({c1}->{c2}) .m[0] is C3k",
              type(inner).__name__ == "C3k", f"got {type(inner).__name__}")
        in_plan = ChannelPlan((c1,))
        out_plan = plan_c2f(wide, in_plan)
        tag = f"C3k2({c1}->{c2},c3k,e{e})"
        check(f"{tag} inner cv3 in_plan has 2 segments",
              len(getattr(inner.cv3, "_ofa_in_plan").groups) == 2,
              f"got {getattr(inner.cv3, '_ofa_in_plan').groups}")

        for w in WIDTHS:
            twin = build_narrow_twin(wide, w)
            x = torch.randn(2, in_plan.active(w), 16, 16)
            set_width(wide, w)
            with torch.no_grad():
                got, want = wide(x), twin(x)
            if got.shape != want.shape:
                check(f"{tag} w={w} shape", False,
                      f"{tuple(got.shape)} vs {tuple(want.shape)}")
                continue
            d = (got - want).abs().max().item()
            check(f"{tag} w={w} equals narrow twin", d < ATOL, f"max|diff|={d:.3e}")


def test_m4_sppf() -> None:
    """SPPF — repeated concat plus the `y + x` residual (yolo26s L9)."""
    print("\n[M4] SPPF with repeated concat + residual (yolo26s L9)")
    torch.manual_seed(0)
    for (c1, c2, n, shortcut) in [(512, 512, 3, True),   # L9, add=True
                                  (256, 256, 3, True),
                                  (512, 256, 3, False)]:  # no residual
        wide = SPPF(c1, c2, 5, n, shortcut).eval()
        randomize_bn(wide)
        in_plan = ChannelPlan((c1,))
        out_plan = plan_sppf(wide, in_plan)
        tag = f"SPPF({c1}->{c2},n{n},add={wide.add})"
        check(f"{tag} cv2 in_plan has {n + 1} repeated segments",
              len(getattr(wide.cv2, "_ofa_in_plan").groups) == n + 1,
              f"got {getattr(wide.cv2, '_ofa_in_plan').groups}")
        if wide.add:
            check(f"{tag} residual ties out_plan to in_plan",
                  out_plan.groups == in_plan.groups
                  and out_plan.elastic == in_plan.elastic)

        for w in WIDTHS:
            twin = build_narrow_twin(wide, w)
            x = torch.randn(2, in_plan.active(w), 16, 16)
            set_width(wide, w)
            with torch.no_grad():
                got, want = wide(x), twin(x)
            if got.shape != want.shape:
                check(f"{tag} w={w} shape", False,
                      f"{tuple(got.shape)} vs {tuple(want.shape)}")
                continue
            d = (got - want).abs().max().item()
            check(f"{tag} w={w} equals narrow twin", d < ATOL, f"max|diff|={d:.3e}")


def test_m56_whole_model() -> None:
    """M5/M6: plan the whole yolo26s graph, then check the standing invariants.

    An exact narrow twin is impossible for the FULL net (attention caches head
    dims, so a channel-sliced twin would be invalid — oracle_util refuses).
    Per-module exactness is already established above; here we check the two
    whole-graph properties that matter:
      1. bit-identity at w=1.0 (planning must be a no-op at full width)
      2. a clean, finite forward at every width, with the head's fixed output
    """
    print("\n[M5/M6] whole-model plan walk (yolo26s)")
    import copy as _copy

    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    model = DetectionModel("yolo26s.yaml", ch=3, nc=80, verbose=False).eval()
    reference = _copy.deepcopy(model).eval()  # never planned

    try:
        plans = plan_model(model, verbose=True)
    except Exception as e:  # noqa: BLE001
        check("plan_model completes", False, f"{type(e).__name__}: {e}")
        return
    check("plan_model completes", True)

    n_planned = sum(
        1 for m in model.modules()
        if isinstance(m, Conv) and hasattr(m, "_ofa_out_plan")
    )
    n_conv = sum(1 for m in model.modules() if isinstance(m, Conv))
    print(f"        planned {n_planned}/{n_conv} Convs "
          f"({n_conv - n_planned} left frozen at width 1.0)")

    x = torch.randn(1, 3, 256, 256)

    set_width(model, 1.0)
    with torch.no_grad():
        a = model(x)
        b = reference(x)
    a0 = a[0] if isinstance(a, (list, tuple)) else a
    b0 = b[0] if isinstance(b, (list, tuple)) else b
    d = (a0 - b0).abs().max().item()
    check("w=1.0 is bit-identical to the unplanned model", d == 0.0,
          f"max|diff|={d:.3e}")

    for w in WIDTHS:
        set_width(model, w)
        try:
            with torch.no_grad():
                out = model(x)
        except Exception as e:  # noqa: BLE001
            check(f"whole-model forward at w={w}", False, f"{type(e).__name__}: {e}")
            continue
        o = out[0] if isinstance(out, (list, tuple)) else out
        finite = bool(torch.isfinite(o).all())
        params = count_active_params(model, w)
        check(f"whole-model forward at w={w} finite", finite,
              f"shape={tuple(o.shape)}")
        print(f"        w={w:<6} out={tuple(o.shape)}  active params={params/1e6:.2f}M")


def test_m7_attention() -> None:
    """P5: C2PSA and C3k2-attn must be exact too, now that they are elastic.

    This is the module whose forward reads CACHED integers (num_heads,
    key_dim, head_dim, scale), so the narrow twin only becomes valid once
    oracle_util rescales them -- which is exactly what makes this test the
    check on the P5 design rather than on the slicing alone.
    """
    print("\n[M7/M8] attention: C2PSA (L10) and C3k2-attn (L22)")
    torch.manual_seed(0)
    from ultralytics.nn.modules.block import C2PSA
    from plan_builder import plan_c2psa

    for c1 in (512, 256):
        wide = C2PSA(c1, c1, 1).eval()
        randomize_bn(wide)
        in_plan = ChannelPlan((c1,))
        out_plan = plan_c2psa(wide, in_plan)
        attn = wide.m[0].attn
        tag = f"C2PSA({c1})"
        check(f"{tag} nh={attn.num_heads} hd={attn.head_dim} kd={attn.key_dim}"
              " qkv plan is (kd,kd,hd)*nh",
              getattr(attn.qkv, "_ofa_out_plan").groups
              == (attn.key_dim, attn.key_dim, attn.head_dim) * attn.num_heads)
        for w in WIDTHS:
            twin = build_narrow_twin(wide, w)
            x = torch.randn(2, in_plan.active(w), 16, 16)
            set_width(wide, w)
            with torch.no_grad():
                got, want = wide(x), twin(x)
            if got.shape != want.shape:
                check(f"{tag} w={w} shape", False,
                      f"{tuple(got.shape)} vs {tuple(want.shape)}")
                continue
            d = (got - want).abs().max().item()
            check(f"{tag} w={w} equals narrow twin", d < ATOL, f"max|diff|={d:.3e}")

    # the real L22: C3k2 with attn=True -> .m = Sequential(Bottleneck, PSABlock)
    wide = C3k2(768, 512, 1, True, 0.5, True).eval()
    randomize_bn(wide)
    in_plan = ChannelPlan((768,))
    plan_c2f(wide, in_plan)
    check("L22 C3k2-attn planned (has Attention inside)",
          any(type(m).__name__ == "Attention" for m in wide.modules()))
    for w in WIDTHS:
        twin = build_narrow_twin(wide, w)
        x = torch.randn(2, in_plan.active(w), 16, 16)
        set_width(wide, w)
        with torch.no_grad():
            got, want = wide(x), twin(x)
        if got.shape != want.shape:
            check(f"C3k2-attn(768->512) w={w} shape", False,
                  f"{tuple(got.shape)} vs {tuple(want.shape)}")
            continue
        d = (got - want).abs().max().item()
        check(f"C3k2-attn(768->512) w={w} equals narrow twin", d < ATOL,
              f"max|diff|={d:.3e}")


def main() -> int:
    print("=" * 68)
    print("elastic == narrow oracle")
    print("=" * 68)
    test_plan_semantics()
    test_invariant_fires()
    test_m1_plain_conv()
    test_m2_c3k2_bottleneck()
    test_m3_c3k2_nested_c3k()
    test_m4_sppf()
    test_m7_attention()
    test_legacy_slicing_fails_the_oracle()
    test_m56_whole_model()
    print("\n" + "=" * 68)
    print(f"{_PASSES} passed, {len(_FAILURES)} failed")
    for f in _FAILURES:
        print(f"  FAILED: {f}")
    print("=" * 68)
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
