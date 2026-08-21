"""Gate A — first honest measurement of the width-elastic yolo26s supernet.

Order matters, and it is the order the previous attempt got wrong:

  1. RECAL SANITY GATE. Recalibrate BN at w=1.0 and evaluate. A correct
     procedure MUST return ~0.472, because w=1.0 is the pretrained network.
     The earlier `bn_recal.py` returned 0.345 here — it calibrated on *val*
     images and mishandled momentum — which made every w<1 number from that
     run uninterpretable. If this gate fails, nothing below is worth reading.

  2. GATE A itself. With correct group-structured slicing (P1) and per-width
     BN stats, measure mAP at each width with NO sorting and NO training.
     Pass condition: **w=0.875 > 0.20 mAP**. Dropping 12.5% of channels from
     a converged detector should leave it degraded but alive. Still ~0.0 would
     mean something structural remains, and we stop rather than pile training
     on top.

Recalibration images come from **train2017**, not val — using val leaks the
evaluation distribution into the statistics and was one of the two defects in
the old run.

Usage:
  python experiments/ofa/gate_a.py --model /root/yolo26s.pt \
      --data experiments/ofa/coco.yaml --calib-batches 200 --batch 32
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
    bn_stats_coverage,
    count_active_params,
    disable_fuse,
    install_elastic_conv,
    recalibrate,
    set_width,
)
from plan_builder import plan_model  # noqa: E402
from sorter import set_criterion, sort_model  # noqa: E402

from ultralytics import YOLO  # noqa: E402

# n -> s straight line: default n (2.6M, 0.395) to default s (9.5M, 0.472).
# A width only counts as a NEW Pareto point if it beats this.
_N = (2.6, 0.395)
_S = (9.5, 0.472)
_SLOPE = (_S[1] - _N[1]) / (_S[0] - _N[0])


def interp_bar(params_m: float) -> float:
    return _N[1] + (params_m - _N[0]) * _SLOPE


class CalibBatches:
    """Re-iterable stream of normalised TRAIN-split image batches.

    Two properties this must have, both learned the hard way:

    * **Streamed, not materialised.** Holding 200 batches of 32x3x640x640 on
      the GPU is ~31 GB and OOMs a 32 GB card instantly. Each batch is loaded
      on demand and only moved to the device inside the recal loop.
    * **Identical images for every width.** Widths are only comparable if they
      were calibrated on the same data, so we draw one seeded index subset up
      front and iterate it with `shuffle=False` — re-iterating a shuffled
      loader would advance its generator and give each width different images.
    """

    def __init__(self, data_yaml: str, imgsz: int, batch: int, n_batches: int,
                 seed: int = 0, mode: str = "val"):
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset(data_yaml)
        cfg = get_cfg()
        cfg.imgsz = imgsz
        cfg.rect = False
        # mode="val" -> deterministic letterbox, no augmentation. We want the
        # train *distribution*, not train-time augmentation.
        # mode="val": clean letterbox. mode="train": the same augmentation
        # pipeline the original BN statistics were accumulated under (mosaic,
        # scale, flip, HSV), which is a genuinely different distribution.
        ds = build_yolo_dataset(cfg, data["train"], batch, data, mode=mode,
                                stride=32)
        g = torch.Generator().manual_seed(seed)
        want = min(n_batches * batch, len(ds))
        idx = torch.randperm(len(ds), generator=g)[:want].tolist()
        self._loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx),
            batch_size=batch, shuffle=False, num_workers=8,
            collate_fn=getattr(ds, "collate_fn", None), drop_last=True,
        )
        self.n_batches = len(self._loader)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for b in self._loader:
            yield b["img"].float() / 255.0


def fresh_planned_model(weights: str, device, sort: bool = False):
    """Load yolo26s, plan it, optionally sort it, make it safe for eval.

    Sorting must happen BEFORE recalibration: stored per-width statistics are
    indexed by channel position, so permuting afterwards would silently
    associate every channel with another channel's statistics.
    """
    y = YOLO(weights)
    model = y.model.to(device)
    plan_model(model)
    if sort:
        sort_model(model)
    disable_fuse(model)
    return y, model


def evaluate(y: YOLO, data_yaml: str, batch: int, device_arg: str) -> float:
    m = y.val(data=data_yaml, imgsz=640, batch=batch, plots=False,
              device=device_arg, verbose=False, save_json=False)
    return float(m.box.map)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/yolo26s.pt")
    ap.add_argument("--data", default=str(_HERE / "coco.yaml"))
    ap.add_argument("--widths", type=float, nargs="+",
                    default=[1.0, 0.875, 0.75, 0.625, 0.5])
    ap.add_argument("--calib-batches", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--momentum", type=float, default=None,
                    help="EMA momentum; omit for a cumulative average (correct "
                         "for a one-shot recal over a fixed batch set)")
    ap.add_argument("--calib-mode", default="val", choices=["val", "train"],
                    help="val = clean letterbox; train = the augmentation "
                         "pipeline the original BN stats were fit under")
    ap.add_argument("--tolerance", type=float, default=0.020,
                    help="max |delta| vs baseline at w=1.0. Catches a BROKEN "
                         "procedure (the old one lost 12.6 pts); it cannot be "
                         "tight enough to demand exact reproduction, because "
                         "recal necessarily perturbs a co-adapted network")
    ap.add_argument("--sort", action="store_true",
                    help="apply importance-based channel sorting (P3) so "
                         "first-k becomes top-k; this is Gate B")
    ap.add_argument("--criterion", default="gamma",
                    choices=["gamma", "gamma_over_sigma", "out_l1"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="/root/gate_a.json")
    args = ap.parse_args()

    install_elastic_conv()
    set_criterion(args.criterion)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print("building calibration batches from the TRAIN split "
          f"({args.calib_batches} x {args.batch})...", flush=True)
    calib = CalibBatches(args.data, 640, args.batch, args.calib_batches,
                         mode=args.calib_mode)
    print(f"  {len(calib)} batches ({len(calib) * args.batch} images), "
          f"mode={args.calib_mode}, streamed and identical for every width",
          flush=True)

    results = {}

    # ---- baseline: no planning at all, for reference -----------------------
    y0 = YOLO(args.model)
    y0.model.to(device)
    base = evaluate(y0, args.data, args.batch, args.device)
    print(f"\n[baseline] stock yolo26s (unplanned, fused): mAP50-95 = {base:.4f}",
          flush=True)
    results["baseline_stock"] = base
    del y0
    torch.cuda.empty_cache()

    # ---- 1. recal sanity gate at w=1.0 ------------------------------------
    print("\n" + "=" * 66)
    print("1. RECAL SANITY GATE  (recal at w=1.0 must reproduce the baseline)")
    print("=" * 66, flush=True)
    y, model = fresh_planned_model(args.model, device, sort=args.sort)
    n = recalibrate(model, calib, 1.0, momentum=args.momentum, device=device)
    have, need = bn_stats_coverage(model, 1.0)
    print(f"  recalibrated on {n} batches; stats present for {have}/{need} convs",
          flush=True)
    set_width(model, 1.0)
    recal_full = evaluate(y, args.data, args.batch, args.device)
    delta = recal_full - base
    sane = abs(delta) <= args.tolerance
    print(f"  mAP50-95 @ w=1.0 after recal = {recal_full:.4f} "
          f"(baseline {base:.4f}, delta {delta:+.4f})")
    print(f"  SANITY GATE: {'PASS' if sane else 'FAIL'} "
          f"(need |delta| <= {args.tolerance:.3f})", flush=True)
    results["recal_w1.0"] = recal_full
    results["recal_penalty"] = delta
    results["recal_sanity_pass"] = bool(sane)
    print(f"  recal penalty at full width = {delta:+.4f} -- this is the cost of "
          f"REPLACING co-adapted stats,\n  not a width effect, so w<1 results are "
          f"reported against BOTH references below.", flush=True)
    del y, model
    torch.cuda.empty_cache()

    if not sane:
        print("\nRecal procedure is not faithful at w=1.0 — every w<1 number "
              "below would be uninterpretable. Stopping.", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        return 1

    # ---- 2. Gate A: mAP at each width, with recal, no sorting/training ----
    print("\n" + "=" * 66)
    print(f"2. {'GATE B (recal + IMPORTANCE SORTING)' if args.sort else 'GATE A (recal only, no sorting)'}"
          "  -- no weight training either way")
    print("=" * 66, flush=True)
    header = f"{'w':>6} {'params':>9} {'mAP50-95':>9} {'n->s bar':>9} {'vs bar':>8}"
    rows = []
    for w in args.widths:
        y, model = fresh_planned_model(args.model, device, sort=args.sort)
        params = count_active_params(model, w) / 1e6
        recalibrate(model, calib, w, momentum=args.momentum, device=device)
        set_width(model, w)
        m = evaluate(y, args.data, args.batch, args.device)
        bar = interp_bar(params)
        rows.append((w, params, m, bar))
        results[f"w{w}"] = {"params_M": params, "map5095": m, "interp_bar": bar}
        print(f"  w={w:<6} params={params:5.2f}M  mAP={m:.4f}  bar={bar:.4f}  "
              f"{'ABOVE' if m > bar else 'below'}", flush=True)
        del y, model
        torch.cuda.empty_cache()

    print()
    print(f"  reference A: stock yolo26s, pretrained stats  = {base:.4f}")
    print(f"  reference B: w=1.0 under the SAME recal       = {recal_full:.4f}")
    print()
    print(f"{'w':>6} {'params':>9} {'mAP':>8} {'vs A':>8} {'vs B':>8} "
          f"{'n->s bar':>9} {'verdict':>9}")
    for w, p, m, bar in rows:
        print(f"{w:>6.3f} {p:>8.2f}M {m:>8.4f} {m - base:>+8.4f} "
              f"{m - recal_full:>+8.4f} {bar:>9.4f} "
              f"{'ABOVE' if m > bar else 'below':>9}")

    got = {w: m for w, _, m, _ in rows}
    a_target = 0.875
    gate_a = got.get(a_target, 0.0) > 0.20
    print(f"\nGATE A: w={a_target} mAP = {got.get(a_target, float('nan')):.4f} "
          f"{'>' if gate_a else '<='} 0.20  -> {'PASS' if gate_a else 'FAIL'}")
    results["sorted"] = bool(args.sort)
    results["gate_a_pass"] = bool(gate_a)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0 if gate_a else 2


if __name__ == "__main__":
    raise SystemExit(main())
