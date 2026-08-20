"""Progressive-shrinking sandwich training for width-elastic yolo26s.

Each optimizer step, on the same batch:
  1. TEACHER pass at max width (widths[0], default 1.0):
       * detection loss `L_teacher`
       * cache the 3 neck feature maps feeding the Detect head (detached).
  2. STUDENT passes at each smaller width (widths[1:]):
       * detection loss `L_student`
       * feature-KD MSE against cached teacher features
  total = L_teacher + sum_i(L_student_i + kd_lambda * KD_i)

The elastic Conv monkey-patch is applied by importing `width_elastic`. EMA is
disabled (it blurs the exact weight sharing OFA depends on). Training uses
amp=False (fp32) to match the earlier sandwich_kd_train.py, and val=False
because `val()` fuses the model, breaking BN presence assumed by both
`_elastic_conv_forward` and downstream training steps.

Stages:
  Stage 1: --widths 1.0 0.75           (introduce w=0.75 alone)
  Stage 2: --widths 1.0 0.75 0.5       (add w=0.5 to the sandwich)
The pretrained yolo26s (w=1.0 = 0.472) is the starting point for stage 1;
`last.pt` from stage 1 is the starting point for stage 2.

Usage:
  PYTHONPATH=/root/geta/experiments/ofa /root/miniconda3/bin/python \
      /root/geta/experiments/ofa/width_sandwich_train.py \
      --model /root/yolo26s.pt --widths 1.0 0.75 \
      --epochs 3 --fraction 0.1 --batch 32 --kd 5.0 --lr 1e-4 \
      --name ofa_ws_stage1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import width_elastic  # noqa: F401  (applies Conv/C2PSA monkey-patch on import)
from width_elastic import set_width

from ultralytics import YOLO


def patch_sandwich(trainer, widths, kd_lambda):
    """Install the sandwich loss on trainer.model.loss."""
    if getattr(trainer, "ema", None) is not None:
        trainer.ema.enabled = False
    m = trainer.model
    detect = m.model[-1]  # Detect head
    cap = {}
    ctr = {"n": 0}

    def pre_hook(_module, args):
        # args[0] is the list of 3 neck feature maps fed to Detect.forward
        cap["feats"] = args[0]

    detect.register_forward_pre_hook(pre_hook)
    orig_loss = m.loss

    def sloss(batch, preds=None):
        # Teacher pass at the largest width, NO gradient (in-place KD pattern):
        # its features are just distillation targets. If we let teacher's
        # detection loss backprop through the shared weights, it fights the
        # student's much larger loss on the same shared params and destabilises
        # the pretrained w=1.0 baseline (empirically: 0.472 -> 0.0001 in 3 ep).
        set_width(m, widths[0])
        with torch.no_grad():
            _ = orig_loss(batch) if preds is None else orig_loss(batch, preds)
        tfeat = [f.detach() for f in cap["feats"]]

        total = None
        items = None
        for w in widths[1:]:
            set_width(m, w)
            l_s, s_items = orig_loss(batch) if preds is None else orig_loss(batch, preds)
            sfeat = cap["feats"]
            # Student feats have fewer channels than teacher; KD compares
            # student channels against the FIRST-k teacher channels.
            kd = 0
            for s, t in zip(sfeat, tfeat):
                k = s.shape[1]
                kd = kd + F.mse_loss(s, t[:, :k])
            term = l_s + kd_lambda * kd
            total = term if total is None else total + term
            items = s_items  # report the last (usually smallest) student's loss items

        # Reset to the max width for the next batch's caller-side inspection.
        set_width(m, widths[0])
        ctr["n"] += 1
        if ctr["n"] <= 3 or ctr["n"] % 100 == 0:
            print(
                f"[sandwich step {ctr['n']}] l_student_sum={l_s.sum().item():.4f}"
                f" kd={float(kd):.4f} total={total.sum().item():.4f}"
                f" req_grad={total.requires_grad}",
                flush=True,
            )
        return total, items

    m.loss = sloss
    print(
        f"[sandwich] patched trainer.model.loss; widths={widths} kd_lambda={kd_lambda}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/yolo26s.pt")
    ap.add_argument("--data", default="/root/geta/experiments/ofa/coco.yaml")
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kd", type=float, default=5.0, help="KD (feature-MSE) weight per sub-net")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="ofa_ws_stage")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    assert args.widths[0] == max(args.widths), "widths[0] must be the largest (teacher)"
    print(f"widths={args.widths} epochs={args.epochs} fraction={args.fraction} "
          f"batch={args.batch} kd={args.kd} lr={args.lr}", flush=True)

    y = YOLO(args.model)
    y.add_callback("on_train_start", lambda tr: patch_sandwich(tr, args.widths, args.kd))

    # Ultralytics save_model() serializes trainer.ema.ema unconditionally. With
    # EMA disabled inside patch_sandwich(), ema.ema stays at initial state and
    # `last.pt` would be byte-identical to the pretrained weights. Sync ema.ema
    # from the live model at the end of every training epoch so save writes
    # the actual trained weights.
    def sync_ema(tr):
        if getattr(tr, "ema", None) is not None:
            tr.ema.ema.load_state_dict(tr.model.state_dict())

    y.add_callback("on_train_epoch_end", sync_ema)
    y.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        fraction=args.fraction,
        lr0=args.lr,
        amp=False,
        warmup_epochs=0,
        optimizer="SGD",
        nbs=args.batch,
        name=args.name,
        val=False,
        plots=False,
        device=args.device,
    )
    print("SANDWICH_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
