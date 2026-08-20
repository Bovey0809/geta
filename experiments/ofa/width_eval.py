"""Eval a width-elastic yolo26s checkpoint at multiple widths.

Loads the trained supernet, iterates over each requested width, and prints
mAP50-95 for each. Import width_elastic first so the loaded model's Conv
modules use the patched forward.

Usage:
  /root/miniconda3/bin/python /root/geta/experiments/ofa/width_eval.py \
      --model /root/runs/detect/ofa_ws_stage1/weights/last.pt \
      --data  /root/geta/experiments/ofa/coco.yaml \
      --widths 1.0 0.75 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import width_elastic  # noqa: F401
from width_elastic import set_width

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/root/geta/experiments/ofa/coco.yaml")
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 0.75, 0.5])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    print(f"model={args.model} widths={args.widths}", flush=True)
    for w in args.widths:
        y = YOLO(args.model)  # fresh load each time; val() marks tensors inference-mode
        set_width(y.model, w)
        m = y.val(
            data=args.data,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            plots=False,
            verbose=False,
        )
        print(f"WIDTH={w}  mAP50-95={float(m.box.map):.4f}  mAP50={float(m.box.map50):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
