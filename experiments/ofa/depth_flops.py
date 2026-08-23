"""Measure what depth-elasticity actually saves, so the accuracy cost can be judged.

The depth rung probe recovered a lot (+0.24 on yolo26s, +0.27 on yolo26l). Whether
that is USEFUL depends entirely on how much compute d=1 saves, which the original
study reported as -13% but never re-derived here. Dropping the second inner
bottleneck of every C3k removes real work, but those bottlenecks are only a
fraction of total FLOPs.

Reports executed GFLOPs at d=2 and d=1, plus the default-family reference points,
so the result can be placed on a FLOPs/accuracy plane rather than argued about.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from channel_plan import install_elastic_conv  # noqa: E402
from depth_retest import install_elastic_depth, set_depth  # noqa: E402
from elastic_attn import install_elastic_attention  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def gflops(model, imgsz=640) -> float:
    from thop import profile
    x = torch.zeros(1, 3, imgsz, imgsz, next(model.parameters()).device.index
                    if next(model.parameters()).is_cuda else None)
    x = torch.zeros(1, 3, imgsz, imgsz).to(next(model.parameters()).device)
    with torch.no_grad():
        f, _ = profile(model, inputs=(x,), verbose=False)
    return f / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["/root/yolo26s.pt", "/root/yolo26l.pt"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    install_elastic_conv()
    install_elastic_attention()
    install_elastic_depth()

    print(f"{'model':<22} {'d=2 GFLOPs':>11} {'d=1 GFLOPs':>11} {'saving':>9}")
    for w in args.models:
        m = YOLO(w).model.eval()
        n = set_depth(m, 2)
        f2 = gflops(m, args.imgsz)
        set_depth(m, 1)
        f1 = gflops(m, args.imgsz)
        print(f"{Path(w).name:<22} {f2:>11.2f} {f1:>11.2f} "
              f"{(f1 - f2) / f2 * 100:>8.1f}%   ({n} C3k blocks)")
        del m
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
