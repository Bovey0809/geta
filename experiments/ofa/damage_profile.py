"""Where does width-slicing damage enter, and how does it amplify?

Gate A failed softly: w=0.875 scored 0.0395 -- not the old exact 0.0, but far
below the 0.20 bar. Before either declaring the approach dead or investing in
importance sorting, we need to know WHICH SHAPE the failure has:

  (a) damage concentrated at one or two layers  -> a residual bug or a single
      critical layer, i.e. a lead worth chasing;
  (b) damage small everywhere but compounding   -> information is genuinely
      distributed across channels, so arbitrary first-k selection is the
      problem and importance sorting is the only lever that can help.

Method: run the planned model twice on the same input, once at w=1.0 and once
at the target width, capturing every top-level layer's output. At w<1 a layer
emits fewer channels, so we compare against the CORRESPONDING channels of the
full-width run (via that layer's ChannelPlan selection) -- like for like.

    rel_mse[L] = ||y_w[L] - y_1[L][sel]||^2 / ||y_1[L][sel]||^2

A per-layer `rel_mse` plus its increments tells us where error is injected
versus merely inherited. This is the same measurement that diagnosed the
depth-elastic study (calib_kd.py), reused with correct channel alignment.

Usage:
  python experiments/ofa/damage_profile.py --model /root/yolo26s.pt \
      --data experiments/ofa/coco.yaml --widths 0.875 0.75 --batches 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from channel_plan import (  # noqa: E402
    _OUT_PLAN,
    ChannelPlan,
    disable_fuse,
    install_elastic_conv,
    recalibrate,
    set_width,
)
from plan_builder import plan_model  # noqa: E402
from sorter import sort_model  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402


def capture_layer_outputs(model, x):
    """Forward once, returning {layer_idx: output tensor} for top-level layers."""
    caught: dict[int, torch.Tensor] = {}
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                caught[i] = out.detach()
        return hook

    for i, L in enumerate(model.model):
        if isinstance(L, Detect):
            continue
        handles.append(L.register_forward_hook(mk(i)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return caught


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/yolo26s.pt")
    ap.add_argument("--data", default=str(_HERE / "coco.yaml"))
    ap.add_argument("--widths", type=float, nargs="+", default=[0.875, 0.75])
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=100)
    ap.add_argument("--sort", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="/root/damage_profile.json")
    args = ap.parse_args()

    install_elastic_conv()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    from gate_a import CalibBatches

    calib = CalibBatches(args.data, 640, args.batch, args.calib_batches)
    probe = CalibBatches(args.data, 640, args.batch, args.batches, seed=1234)

    y = YOLO(args.model)
    model = y.model.to(device)
    plans = plan_model(model)
    if args.sort:
        sort_model(model)
        print("channel sorting APPLIED")
    disable_fuse(model)
    model.eval()

    # per-layer output plans, so we can align channels between widths
    layer_plan: dict[int, ChannelPlan] = {
        i: p for i, p in enumerate(plans) if p is not None
    }

    results = {}
    for w in args.widths:
        print(f"\n=== width {w} ===", flush=True)
        recalibrate(model, calib, w, device=device)
        acc: dict[int, list[float]] = {}
        for imgs in probe:
            imgs = imgs.to(device)
            set_width(model, 1.0)
            full = capture_layer_outputs(model, imgs)
            set_width(model, w)
            nar = capture_layer_outputs(model, imgs)

            for i, yn in nar.items():
                yf = full.get(i)
                if yf is None or i not in layer_plan:
                    continue
                sel = layer_plan[i].select(w, device=yn.device)
                if sel.numel() != yn.shape[1] or yf.shape[1] < sel.numel():
                    continue
                ref = yf.index_select(1, sel)
                num = (yn - ref).pow(2).sum().item()
                den = ref.pow(2).sum().item() + 1e-12
                acc.setdefault(i, []).append(num / den)

        prof = {i: sum(v) / len(v) for i, v in sorted(acc.items())}
        results[str(w)] = prof
        print(f"{'L':>4} {'module':<10} {'rel_mse':>10} {'increment':>10}")
        prev = 0.0
        for i, v in prof.items():
            name = type(model.model[i]).__name__
            print(f"{i:>4} {name:<10} {v:>10.4f} {v - prev:>+10.4f}")
            prev = v
        set_width(model, 1.0)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
