"""Correctness + payoff check for block-level depth elasticity.

Three things, in order of what would sink the idea fastest:

  1. **Payoff.** How many MACs does dropping whole C3k blocks actually save?
     Inner-bottleneck depth only saved 5.2 % / 12.7 %, which is why its
     excellent trainability did not matter. If this axis is also small, stop
     here rather than spending GPU.
  2. **Full depth is a no-op.** With every `.m` item kept, the patched forward
     must be bit-identical to stock — otherwise the harness itself is changing
     the model.
  3. **Reduced depth is well-formed.** Correct shapes, finite output, and
     cv2's input selection must match the number of segments actually
     concatenated (the runtime assert in channel_plan is what catches a
     mismatch, so this test would fail loudly rather than silently).

CPU-only, no COCO, seconds.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from block_depth import (  # noqa: E402
    block_depth_report,
    install_block_depth,
    set_block_depth,
)
from channel_plan import install_elastic_conv, set_width  # noqa: E402
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_model  # noqa: E402

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

_FAIL: list[str] = []
_PASS = 0


def check(name, cond, detail=""):
    global _PASS
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def macs(model, imgsz=640) -> float:
    from thop import profile
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        f, _ = profile(model, inputs=(x,), verbose=False)
    return f / 1e9


def main() -> int:
    install_elastic_conv()
    install_elastic_attention()
    install_block_depth()
    torch.manual_seed(0)

    print("=" * 68)
    print("block-level depth (drop whole C3k blocks from C3k2.m)")
    print("=" * 68)

    for cfg in ("yolo26s.yaml", "yolo26l.yaml"):
        print(f"\n[{cfg}]")
        model = DetectionModel(cfg, ch=3, nc=80, verbose=False).eval()
        rep = block_depth_report(model)
        multi = [(k, v) for k, v in rep if v > 1]
        print(f"  C2f/C3k2 blocks: {len(rep)}; with n>1 (droppable): {len(multi)}")
        for k, v in multi:
            print(f"    {k}: n={v}")
        if not multi:
            check(f"{cfg}: no block-depth headroom (expected for s)", True)
            continue

        plan_model(model)
        ref = copy.deepcopy(model).eval()
        x = torch.randn(1, 3, 256, 256)

        # 2. full depth must be a no-op
        set_block_depth(model, 99)
        set_width(model, 1.0)
        with torch.no_grad():
            a = model(x)
            b = ref(x)
        a0 = a[0] if isinstance(a, (list, tuple)) else a
        b0 = b[0] if isinstance(b, (list, tuple)) else b
        d = (a0 - b0).abs().max().item()
        check(f"{cfg}: full depth is bit-identical to stock", d == 0.0,
              f"max|diff|={d:.3e}")

        # 3. reduced depth is well-formed
        touched, dropped = set_block_depth(model, 1)
        check(f"{cfg}: set_block_depth(1) touched {touched} blocks, "
              f"dropped {dropped} sub-blocks", touched == len(multi))
        try:
            with torch.no_grad():
                o = model(x)
            o0 = o[0] if isinstance(o, (list, tuple)) else o
            check(f"{cfg}: keep=1 forward is finite",
                  bool(torch.isfinite(o0).all()), f"shape={tuple(o0.shape)}")
        except Exception as e:  # noqa: BLE001
            check(f"{cfg}: keep=1 forward", False, f"{type(e).__name__}: {e}")
            continue

        # 1. the payoff
        set_block_depth(model, 99)
        f_full = macs(model)
        set_block_depth(model, 1)
        f_min = macs(model)
        print(f"  MACs: full={f_full:.2f}G  keep=1={f_min:.2f}G  "
              f"saving={(f_min - f_full) / f_full * 100:+.1f}%")
        check(f"{cfg}: block depth saves >15% MACs (inner depth managed 12.7%)",
              (f_full - f_min) / f_full > 0.15,
              f"only {(f_full - f_min) / f_full * 100:.1f}%")

    print("\n" + "=" * 68)
    print(f"{_PASS} passed, {len(_FAIL)} failed")
    for f in _FAIL:
        print(f"  FAILED: {f}")
    print("=" * 68)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
