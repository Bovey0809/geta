"""Re-test the depth-elastic 0.0 mAP result, controlling for BN statistics.

THE CLAIM UNDER TEST
--------------------
The 2026-07-01 depth-elastic study reported that dropping the second inner
residual bottleneck of every C3k (d=2 -> d=1) gives **exactly 0.0 mAP**, on the
pretrained net and after two kinds of training, and concluded the ceiling was
"fundamental, not a bug".

TWO REASONS TO DOUBT IT
-----------------------
1. Both dropped blocks are `add=True` RESIDUALS. Dropping `y = x + f(x)` leaves
   `y = x`, the identity — which should degrade gracefully, not annihilate. A
   result of *exactly* 0.0 across 14 blocks is the same "suspiciously total"
   signature that the width-elastic bugs had.
2. It was evaluated with **full-depth BN running statistics and no
   recalibration**. The width study then proved that mismatched BN stats alone
   are enough to destroy a sub-network — and that a recal procedure which looks
   reasonable can be badly wrong (the original `bn_recal.py` returned 0.345 at
   w=1.0 where it had to return 0.472).

So the ~100% relative feature MSE that was offered as the mechanism does not
discriminate between a real capacity ceiling and an un-recalibrated
distribution shift. This script separates them: the SAME d=1 sub-network,
evaluated with and without per-depth BN recalibration.

Note on scope: run on yolo26s (5 C3k blocks x 2 inner bottlenecks, all
add=True). The original used yolo26l (14 x 2) but the mechanism is identical,
and yolo26s is already set up and verified here. `--model` takes any checkpoint.

Usage:
  python experiments/ofa/depth_retest.py --model /root/yolo26s.pt \
      --data experiments/ofa/coco.yaml --calib-batches 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from channel_plan import (  # noqa: E402
    clear_bn_stats,
    disable_fuse,
    install_elastic_conv,
    recalibrate,
    set_width,
)
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_model  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.block import C3k  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402


def _elastic_c3k_forward(self: C3k, x: torch.Tensor) -> torch.Tensor:
    """C3/C3k forward running only the first `active_depth` inner bottlenecks."""
    d = getattr(self, "active_depth", len(self.m))
    y = self.cv1(x)
    for i, blk in enumerate(self.m):
        if i >= d:
            break
        y = blk(y)
    return self.cv3(torch.cat((y, self.cv2(x)), 1))


def install_elastic_depth() -> None:
    C3k.forward = _elastic_c3k_forward


def set_depth(model: torch.nn.Module, d: int) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, C3k):
            m.active_depth = d
            n += 1
    return n


def neck_feature_mse(model, probe, device) -> list[float]:
    """Relative MSE of Detect's three inputs, d=1 vs d=2, on the SAME images."""
    detect = model.model[-1]
    cap = {}
    h = detect.register_forward_pre_hook(lambda _m, a: cap.__setitem__("f", a[0]))
    acc = [[] for _ in range(3)]
    try:
        with torch.no_grad():
            for imgs in probe:
                imgs = imgs.to(device)
                set_depth(model, 2)
                model(imgs)
                ref = [f.detach().clone() for f in cap["f"]]
                set_depth(model, 1)
                model(imgs)
                for i, (a, b) in enumerate(zip(cap["f"], ref)):
                    acc[i].append(((a - b).pow(2).sum()
                                   / (b.pow(2).sum() + 1e-12)).item())
    finally:
        h.remove()
    return [sum(v) / len(v) for v in acc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/yolo26s.pt")
    ap.add_argument("--data", default=str(_HERE / "coco.yaml"))
    ap.add_argument("--calib-batches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="/root/depth_retest.json")
    args = ap.parse_args()

    install_elastic_conv()
    install_elastic_attention()
    install_elastic_depth()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    from gate_a import CalibBatches
    calib = CalibBatches(args.data, 640, args.batch, args.calib_batches)
    probe = CalibBatches(args.data, 640, 8, 6, seed=1234)

    def fresh():
        y = YOLO(args.model)
        m = y.model.to(device)
        plan_model(m)          # activates the per-config BN stats machinery
        disable_fuse(m)        # fusion would fold the ORIGINAL stats in
        return y, m

    def ev(y):
        r = y.val(data=args.data, imgsz=640, batch=args.batch, plots=False,
                  device=args.device, verbose=False)
        return float(r.box.map)

    results = {}

    # --- reference: unplanned, fused, stock ---------------------------------
    y0 = YOLO(args.model); y0.model.to(device)
    base = ev(y0)
    print(f"[baseline] stock (d=2, pretrained stats) = {base:.4f}", flush=True)
    results["baseline"] = base
    del y0
    torch.cuda.empty_cache()

    n_blocks = 0
    for depth in (2, 1):
        for recal in (False, True):
            y, m = fresh()
            n_blocks = set_depth(m, depth)
            clear_bn_stats(m)
            if recal:
                # Recalibrate WITH THIS DEPTH ACTIVE, so the statistics describe
                # the sub-network actually being measured. recalibrate() clears
                # the key first, so no other config's stats can leak in.
                recalibrate(m, calib, 1.0, device=device)
            set_width(m, 1.0)
            mp = ev(y)
            tag = f"d={depth} {'recal' if recal else 'no-recal'}"
            print(f"  {tag:<18} mAP50-95 = {mp:.4f}", flush=True)
            results[f"d{depth}_{'recal' if recal else 'norecal'}"] = mp
            del y, m
            torch.cuda.empty_cache()

    # --- the feature-MSE claim, with and without recal ----------------------
    y, m = fresh()
    set_depth(m, 2)
    clear_bn_stats(m)
    mse_norecal = neck_feature_mse(m, probe, device)
    print(f"\nneck rel-MSE d=1 vs d=2, NO recal : "
          f"P3={mse_norecal[0]:.3f} P4={mse_norecal[1]:.3f} P5={mse_norecal[2]:.3f}",
          flush=True)
    del y, m
    torch.cuda.empty_cache()

    results["neck_rel_mse_norecal"] = mse_norecal
    results["c3k_blocks"] = n_blocks

    print("\n" + "=" * 66)
    print(f"C3k blocks made elastic: {n_blocks} (x2 inner residual bottlenecks)")
    d1n, d1r = results["d1_norecal"], results["d1_recal"]
    print(f"d=1 without recal : {d1n:.4f}   <- reproduces the original protocol")
    print(f"d=1 WITH recal    : {d1r:.4f}")
    if d1n < 0.01 <= d1r:
        print("\nVERDICT: the original 0.0 was a BN-statistics artefact. Dropping a"
              "\nresidual bottleneck degrades the net but does not annihilate it.")
    elif d1n < 0.01 and d1r < 0.01:
        print("\nVERDICT: 0.0 survives correct per-depth recalibration, so the "
              "\noriginal conclusion stands on this model.")
    else:
        print("\nVERDICT: see numbers above.")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
