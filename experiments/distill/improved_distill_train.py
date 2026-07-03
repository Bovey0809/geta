"""Run distillation with the ImprovedDistillationModel (logit / cwd / fgd terms), by
monkeypatching the trainer to build our subclass instead of the stock DistillationModel.
With no --logit/--cwd/--fgd flags it reproduces the stock method (score-weighted L2),
so the same script A/B's stock vs improved.

Usage:
  # stock baseline:
  PYTHONPATH=/root/geta python improved_distill_train.py --name ab_stock
  # all three improvements:
  PYTHONPATH=/root/geta python improved_distill_train.py --logit --cwd --fgd --name ab_all
"""
import argparse
import sys
sys.path.insert(0, "/root/geta/experiments/distill")
import ultralytics.engine.trainer as trainer_mod
from improved_distill import ImprovedDistillationModel
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="yolo26m.pt")
    ap.add_argument("--teacher", default="yolo26x.pt")
    ap.add_argument("--data", default="/root/geta/experiments/geta_yolo26/coco.yaml")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup_epochs (low for from-pretrained FT)")
    ap.add_argument("--dis", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--logit", action="store_true")
    ap.add_argument("--cwd", action="store_true")
    ap.add_argument("--fgd", action="store_true")
    ap.add_argument("--logit_w", type=float, default=1.0)
    ap.add_argument("--cwd_w", type=float, default=4.0)
    ap.add_argument("--fgd_w", type=float, default=2e-4)
    ap.add_argument("--name", default="improved_distill")
    args = ap.parse_args()

    ImprovedDistillationModel.CFG = {
        "logit": args.logit, "cwd": args.cwd, "fgd": args.fgd,
        "logit_w": args.logit_w, "cwd_w": args.cwd_w, "fgd_w": args.fgd_w, "cwd_T": 4.0,
    }
    # trainer builds DistillationModel(student, teacher) internally -> make it build ours
    trainer_mod.DistillationModel = ImprovedDistillationModel
    print(f"[improved-distill] CFG={ImprovedDistillationModel.CFG}", flush=True)

    y = YOLO(args.student)
    kw = dict(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=640,
              distill_model=args.teacher, dis=args.dis, fraction=args.fraction,
              warmup_epochs=args.warmup, name=args.name, device=0)
    if args.lr is not None:
        kw["lr0"] = args.lr
    y.train(**kw)
    m = y.val(data=args.data, imgsz=640, batch=args.batch, device=0)
    print(f"IMPROVED_DISTILL_DONE name={args.name} logit={args.logit} cwd={args.cwd} fgd={args.fgd} "
          f"map5095={float(m.box.map):.4f} map50={float(m.box.map50):.4f}", flush=True)


if __name__ == "__main__":
    main()
