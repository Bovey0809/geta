"""Measure the OFA payoff: eval a depth-elastic YOLO26 supernet at each depth.

Loads one trained supernet checkpoint, and for each active_depth runs COCO val +
GFLOPs. The OFA promise is that a single supernet yields several sub-networks
(here: depth-2 full and depth-1 shrunk) that are all usable with NO retraining.

Usage:
  PYTHONPATH=/root/geta python ofa_eval.py --weights runs/detect/ofa_l/weights/best.pt --depths 2 1
"""
import argparse, json, sys
sys.path.insert(0, "/root/geta/experiments/ofa")
import torch
from elastic_yolo26 import set_depth  # importing applies the global C3k elastic patch
from ultralytics import YOLO


def flops_at_depth(weights, d, imgsz):
    from thop import profile
    y = YOLO(weights)
    n = set_depth(y.model, d)
    y.model.eval()
    x = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        flops, _ = profile(y.model, inputs=(x,), verbose=False)
    return n, round(flops / 1e9, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="/root/geta/experiments/geta_yolo26/coco.yaml")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    res = {}
    for d in args.depths:
        # FLOPs on a fresh instance (thop adds hooks; keep it away from the val model)
        n, gflops = flops_at_depth(args.weights, d, args.imgsz)
        # Val on a clean instance at this depth
        y = YOLO(args.weights)
        set_depth(y.model, d)
        m = y.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=0)
        res[f"d{d}"] = {"elastic_blocks": n, "gflops": gflops,
                        "map5095": round(float(m.box.map), 4), "map50": round(float(m.box.map50), 4)}
        print(f"OFA_EVAL d={d} {json.dumps(res[f'd{d}'])}", flush=True)
    print("OFA_EVAL_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
