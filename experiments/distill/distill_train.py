"""Knowledge distillation via Ultralytics' built-in DistillationModel (ultralytics>=8.4.x).

Teacher (default yolo26x, 0.5626 mAP) -> student (default yolo26m, 0.518). Setting
`distill_model=<teacher>` makes the trainer wrap student+teacher in DistillationModel:
score-weighted-L2 feature KD (weight `dis`, default 6.0) with a learned 1x1 projector
bridging student<->teacher channel dims. Unlike pruning/OFA this ADDS a teacher signal
to a full-capacity student, so it can push m ABOVE its default at the SAME 20.4M params.

Goal: distilled-m > default-L (0.5375) at 20.4M params = the Pareto win pruning/OFA missed.

Usage:
  PYTHONPATH=/root/geta python distill_train.py --student yolo26m.pt --teacher yolo26x.pt \
      --epochs 40 --batch 48 --name distill_x2m
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="yolo26m.pt")
    ap.add_argument("--teacher", default="yolo26x.pt")
    ap.add_argument("--data", default="/root/geta/experiments/geta_yolo26/coco.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--dis", type=float, default=6.0, help="distillation loss weight")
    ap.add_argument("--lr", type=float, default=None, help="lr0 override (None = Ultralytics auto)")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="distill_x2m")
    args = ap.parse_args()

    y = YOLO(args.student)
    kw = dict(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=640,
              distill_model=args.teacher, dis=args.dis, fraction=args.fraction,
              name=args.name, device=0)
    if args.lr is not None:
        kw["lr0"] = args.lr
    res = y.train(**kw)
    # final COCO val of the distilled student
    m = y.val(data=args.data, imgsz=640, batch=args.batch, device=0)
    print(f"DISTILL_DONE student={args.student} teacher={args.teacher} "
          f"map5095={float(m.box.map):.4f} map50={float(m.box.map50):.4f}", flush=True)


if __name__ == "__main__":
    main()
