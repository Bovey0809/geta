"""Single-rung progressive-shrinking probe: can one width step be recovered?

The P5 sweep left a LADDER of healthy footholds (w=0.99 -> 0.4250, 0.98 ->
0.3806, 0.96 -> 0.3008, 0.94 -> 0.2141, 0.92 -> 0.1427). Progressive shrinking
is supposed to descend that ladder, recovering at each rung. Before funding the
whole descent, measure ONE rung: sandwich-train {1.0, 0.98} and see how much of
w=0.98's gap to w=1.0 gets recovered.

That single number is the per-rung recovery rate, which is what decides whether
the full schedule is worth the GPU.

WHY THIS CAN WORK NOW AND COULD NOT BEFORE
------------------------------------------
Every earlier training attempt was doomed by two bugs, since fixed:
  * the sliced forward mis-aligned every chunk/cat boundary, so the "student"
    was not a sub-network at all;
  * `bn.running_mean[:k]` was a view, so narrow train-mode passes corrupted the
    shared statistics and destroyed the teacher (kd=10 and kd=1000 collapsed
    identically -- the tell that it was never a gradient-balance problem).
On top of that, the student now STARTS at 0.3806 rather than 0.0, so its
gradients are of the same order as the teacher's instead of swamping them.

SANDWICH RULE (OFA)
-------------------
Per step, on the same batch:
  max sub-net  (w=1.0)  -> detection loss WITH gradient, and its Detect-input
                          features are cached as distillation targets
  small sub-net(w=0.98) -> detection loss + feature-KD against those targets
Training the max arm keeps w=1.0 anchored, which is the point of a sandwich;
the earlier runs that froze it were compensating for the BN corruption.

Usage:
  python experiments/ofa/rung_train.py --widths 1.0 0.98 \
      --epochs 6 --fraction 0.15 --batch 32 --lr 2e-4 --kd 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
from block_depth import install_block_depth, set_block_depth  # noqa: E402
from depth_retest import install_elastic_depth, set_depth  # noqa: E402
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_model  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def patch_sandwich(trainer, configs, kd_lambda, state, apply_cfg):
    """Plan the trainer's model and install the sandwich loss."""
    m = trainer.model
    plan_model(m)
    n_planned = sum(1 for _ in m.modules())
    state["planned"] = True

    # EMA holds a separate copy whose weights never see our sandwich updates,
    # and Ultralytics' save_model serialises EMA -- so keep it in sync (this
    # bug previously made last.pt byte-identical to the pretrained weights).
    if getattr(trainer, "ema", None) is not None:
        trainer.ema.enabled = False

    detect = m.model[-1]
    cap = {}
    detect.register_forward_pre_hook(lambda _mod, args: cap.__setitem__("f", args[0]))
    orig_loss = m.loss
    ctr = {"n": 0}

    def sloss(batch, preds=None):
        # --- max sub-net: real loss, and cache KD targets ---
        apply_cfg(m, configs[0])
        l_max, items = orig_loss(batch) if preds is None else orig_loss(batch, preds)
        tfeat = [f.detach() for f in cap["f"]]

        total = l_max
        for cfg in configs[1:]:
            apply_cfg(m, cfg)
            l_s, _ = orig_loss(batch) if preds is None else orig_loss(batch, preds)
            kd = 0
            for s, t in zip(cap["f"], tfeat):
                k = s.shape[1]
                kd = kd + F.mse_loss(s, t[:, :k])
            total = total + l_s + kd_lambda * kd

        apply_cfg(m, configs[0])
        ctr["n"] += 1
        if ctr["n"] <= 2 or ctr["n"] % 200 == 0:
            print(f"[rung {ctr['n']}] l_max={l_max.sum().item():.3f} "
                  f"l_small={l_s.sum().item():.3f} kd={float(kd):.4f} "
                  f"total={total.sum().item():.3f}", flush=True)
        return total, items

    m.loss = sloss
    print(f"[rung] sandwich installed: configs={configs} kd={kd_lambda}", flush=True)


def evaluate_configs(weights, data, configs, calib, batch, device_arg, device,
                     apply_cfg, is_depth, label):
    """Recalibrate and evaluate each config on a freshly loaded checkpoint.

    The config is applied BEFORE recalibration, so the stored statistics
    describe the sub-network actually measured. Getting that order wrong
    silently stores the other config's stats -- which is precisely the mistake
    that produced the retracted depth-elastic 0.0.
    """
    out = {}
    for cfg in configs:
        y = YOLO(weights)
        m = y.model.to(device)
        plan_model(m)
        disable_fuse(m)
        apply_cfg(m, cfg)
        recal_w = 1.0 if is_depth else cfg
        recalibrate(m, calib, recal_w, device=device)
        have, need = bn_stats_coverage(m, recal_w)
        apply_cfg(m, cfg)          # recalibrate() re-sets width; re-assert depth
        set_width(m, recal_w)
        res = y.val(data=data, imgsz=640, batch=batch, plots=False,
                    device=device_arg, verbose=False)
        out[cfg] = {"map5095": float(res.box.map),
                    "params_M": count_active_params(m, recal_w) / 1e6,
                    "bn_stats": f"{have}/{need}"}
        print(f"  {label}={cfg:<6} params={out[cfg]['params_M']:5.2f}M  "
              f"mAP={out[cfg]['map5095']:.4f}", flush=True)
        del y, m
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/yolo26s.pt")
    ap.add_argument("--data", default=str(_HERE / "coco.yaml"))
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 0.98])
    ap.add_argument("--depths", type=int, nargs="+", default=None,
                    help="sandwich over C3k inner-bottleneck DEPTH instead "
                         "of width, e.g. --depths 2 1")
    ap.add_argument("--block-depths", type=int, nargs="+", default=None,
                    help="sandwich over BLOCK depth -- how many C3k blocks of "
                         "each C3k2's .m to keep, e.g. --block-depths 2 1. "
                         "Saves ~30%% MACs on yolo26l vs ~13%% for inner depth.")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--calib-batches", type=int, default=200)
    ap.add_argument("--name", default="rung_098")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="/root/rung_probe.json")
    args = ap.parse_args()

    install_elastic_conv()
    install_elastic_attention()
    install_elastic_depth()
    install_block_depth()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    is_depth = args.depths is not None or args.block_depths is not None
    if args.block_depths is not None:
        configs = args.block_depths
        label = "bd"
        def apply_cfg(m, c):
            set_block_depth(m, int(c))
    elif args.depths is not None:
        configs = args.depths
        label = "d"
        def apply_cfg(m, c):
            set_depth(m, int(c))
    else:
        configs = args.widths
        label = "w"
        def apply_cfg(m, c):
            set_width(m, float(c))
    assert configs[0] == max(configs), "configs[0] must be the max sub-net"

    from gate_a import CalibBatches
    calib = CalibBatches(args.data, 640, args.batch, args.calib_batches)

    print("=" * 66)
    print("BEFORE training")
    print("=" * 66, flush=True)
    before = evaluate_configs(args.model, args.data, configs, calib, args.batch,
                              args.device, device, apply_cfg, is_depth, label)

    print("\n" + "=" * 66)
    print(f"TRAINING sandwich {label}={configs}  ({args.epochs} ep @ "
          f"fraction={args.fraction})")
    print("=" * 66, flush=True)
    state = {}
    y = YOLO(args.model)
    y.add_callback("on_train_start",
                   lambda tr: patch_sandwich(tr, configs, args.kd, state, apply_cfg))
    y.add_callback("on_train_epoch_end",
                   lambda tr: tr.ema.ema.load_state_dict(tr.model.state_dict())
                   if getattr(tr, "ema", None) is not None else None)
    y.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=640,
            fraction=args.fraction, lr0=args.lr, amp=False, warmup_epochs=0,
            optimizer="SGD", nbs=args.batch, name=args.name, val=False,
            plots=False, device=args.device)

    ckpt = sorted(Path("/root/runs/detect").glob(f"{args.name}*/weights/last.pt"),
                  key=lambda p: p.stat().st_mtime)[-1]
    print(f"\ntrained checkpoint: {ckpt}", flush=True)
    # Confirm the weights actually moved -- the EMA-save bug once made this
    # checkpoint byte-identical to the input.
    import hashlib
    h_new = hashlib.md5(Path(ckpt).read_bytes()).hexdigest()
    h_old = hashlib.md5(Path(args.model).read_bytes()).hexdigest()
    print(f"checkpoint differs from input: {h_new != h_old}", flush=True)

    print("\n" + "=" * 66)
    print("AFTER training")
    print("=" * 66, flush=True)
    after = evaluate_configs(str(ckpt), args.data, configs, calib, args.batch,
                             args.device, device, apply_cfg, is_depth, label)

    small, big = configs[-1], configs[0]
    gap_before = before[big]["map5095"] - before[small]["map5095"]
    gap_after = after[big]["map5095"] - after[small]["map5095"]
    recovered = (gap_before - gap_after) / gap_before if gap_before > 0 else 0.0
    print("\n" + "=" * 66)
    print(f"RUNG RECOVERY at {label}={small}")
    print("=" * 66)
    print(f"  before: max={before[big]['map5095']:.4f} "
          f"small={before[small]['map5095']:.4f}  gap={gap_before:.4f}")
    print(f"  after:  max={after[big]['map5095']:.4f} "
          f"small={after[small]['map5095']:.4f}  gap={gap_after:.4f}")
    print(f"  small sub-net gain: "
          f"{after[small]['map5095'] - before[small]['map5095']:+.4f}")
    print(f"  gap closed: {recovered * 100:.1f}%")
    print(f"  max sub-net drift: "
          f"{after[big]['map5095'] - before[big]['map5095']:+.4f}")

    Path(args.out).write_text(json.dumps(
        {"before": {str(k): v for k, v in before.items()},
         "after": {str(k): v for k, v in after.items()},
         "gap_before": gap_before, "gap_after": gap_after,
         "fraction_recovered": recovered,
         "args": vars(args)}, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
