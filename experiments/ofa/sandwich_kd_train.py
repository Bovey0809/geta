"""Proper depth-elastic OFA for YOLO26 detection: sandwich + in-place feature KD.

Naive sandwich (sum d2+d1 detection losses) leaves d=1 at 0.0 mAP — the detection
head is acutely sensitive to backbone/neck feature statistics, and dropping C3k inner
bottlenecks shifts those features off-distribution. Fix: in-place knowledge distillation.

Each optimizer step, on the same batch:
  1. teacher pass  set_depth(2): full net -> detection loss L_det2, CACHE the 3 neck
     feature maps feeding the Detect head (detached targets).
  2. student pass  set_depth(1): sub-net -> detection loss L_det1 + KD loss
     L_kd = sum MSE(student_neck_feats, teacher_neck_feats).
  total = L_det2 + L_det1 + kd_lambda * L_kd

Depth changes don't alter feature-map shapes, so the per-scale MSE is well defined.
Starting from a fully-trained yolo26l = the "train-large-first" stage of progressive
shrinking already complete; this co-adapts the shared weights so d=1 also works.

Usage:
  PYTHONPATH=/root/geta python sandwich_kd_train.py --epochs 15 --fraction 0.2 \
      --batch 32 --kd 10.0 --name ofa_l_kd
"""
import sys, argparse
sys.path.insert(0, "/root/geta/experiments/ofa")
from elastic_yolo26 import set_depth  # applies the global C3k elastic patch on import
import torch
import torch.nn.functional as F
from ultralytics import YOLO

DEPTHS = [2, 1]  # teacher (full) + student (min)


def patch_sandwich_kd(trainer, kd_lambda):
    if getattr(trainer, "ema", None) is not None:
        trainer.ema.enabled = False
    m = trainer.model
    detect = m.model[-1]  # Detect head; its input is the list of 3 neck feature maps
    cap = {}

    def pre_hook(module, args):
        cap["feats"] = args[0]  # the list of feature maps passed to Detect.forward

    detect.register_forward_pre_hook(pre_hook)
    orig = m.loss

    def sloss(batch, preds=None):
        # teacher: full depth
        set_depth(m, 2)
        l2, items = orig(batch) if preds is None else orig(batch, preds)
        tfeat = [f.detach() for f in cap["feats"]]
        # student: min depth + feature KD to teacher
        set_depth(m, 1)
        l1, _ = orig(batch) if preds is None else orig(batch, preds)
        sfeat = cap["feats"]
        kd = sum(F.mse_loss(s, t) for s, t in zip(sfeat, tfeat))
        total = l2 + l1 + kd_lambda * kd
        return total, items

    m.loss = sloss
    print(f"[sandwich+KD] patched; depths {DEPTHS} kd_lambda={kd_lambda}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kd", type=float, default=10.0, help="KD (feature-MSE) loss weight")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="yolo26l.pt")
    ap.add_argument("--name", default="ofa_l_kd")
    args = ap.parse_args()
    y = YOLO(args.model)
    y.add_callback("on_train_start", lambda tr: patch_sandwich_kd(tr, args.kd))
    y.train(data="/root/geta/experiments/geta_yolo26/coco.yaml", epochs=args.epochs,
            batch=args.batch, imgsz=640, fraction=args.fraction, lr0=args.lr, amp=False,
            warmup_epochs=0, optimizer="SGD", nbs=args.batch, name=args.name,
            val=False, plots=False, device=0)
    print("SANDWICH_KD_DONE", flush=True)
