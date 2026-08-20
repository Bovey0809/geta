"""BN recalibration for a width-elastic yolo26s subnet.

The elastic Conv passes `bn.running_mean[:k], bn.running_var[:k]` to
F.batch_norm. Those stats were fit for the FULL network — not for the k-channel
sub-net's activation distribution. Recalibration: set BN to train mode, forward
a few hundred calibration batches at the target width, letting BN's running
stats accumulate over the sliced activations. Then eval.

Usage:
  /root/miniconda3/bin/python /root/geta/experiments/ofa/bn_recal.py \
      --model /root/yolo26s.pt --data /root/geta/experiments/ofa/coco.yaml \
      --widths 1.0 0.75 0.5 --calib-batches 50 --batch 32
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import width_elastic  # noqa: F401  (applies monkey-patches)
from width_elastic import set_width

from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset


def reset_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


def set_bn_train(model, momentum=0.1):
    """Put ONLY the BN modules into train mode so F.batch_norm updates stats."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = momentum


def recalibrate_bn(model, dataloader, n_batches, device):
    """Forward n_batches at the current width to refit BN running stats."""
    model.eval()  # everything else stays eval (no dropout, etc.)
    set_bn_train(model)
    reset_bn_stats(model)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            img = batch["img"].to(device, non_blocking=True).float() / 255.0
            _ = model(img)  # forward-only; BN stats update in place
    # Return everything to eval
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def eval_model(y, data_yaml, device, batch=16):
    m = y.val(data=data_yaml, imgsz=640, batch=batch, plots=False, device=device, verbose=False)
    return float(m.box.map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/yolo26s.pt")
    ap.add_argument("--data", default="/root/geta/experiments/ofa/coco.yaml")
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 0.75, 0.5])
    ap.add_argument("--calib-batches", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Build a calibration dataloader from the val set (already extracted).
    from ultralytics.cfg import get_cfg
    cfg = get_cfg()
    cfg.data = args.data
    cfg.imgsz = 640
    # Use the val split for calibration (already downloaded).
    from ultralytics.data.utils import check_det_dataset
    data = check_det_dataset(args.data)
    dataset = build_yolo_dataset(cfg, data["val"], args.batch, data, mode="val", stride=32)
    calib_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4,
        collate_fn=getattr(dataset, "collate_fn", None),
    )

    print(f"widths={args.widths} calib_batches={args.calib_batches} batch={args.batch}")

    for w in args.widths:
        print(f"\n=== width={w} ===")
        # Fresh load each time (val() puts tensors in inference mode).
        y = YOLO(args.model)
        m = y.model.to(device)
        set_width(m, w)
        # Untrained subnet
        map_pre = eval_model(y, args.data, args.device, args.batch)
        print(f"  mAP w={w} pre-recal:  {map_pre:.4f}")
        # Load fresh again for recal (val fused the model)
        y2 = YOLO(args.model)
        m2 = y2.model.to(device)
        set_width(m2, w)
        recalibrate_bn(m2, calib_loader, args.calib_batches, device)
        # y2.model has updated BN stats; eval
        map_post = eval_model(y2, args.data, args.device, args.batch)
        print(f"  mAP w={w} post-recal: {map_post:.4f}   (delta={map_post - map_pre:+.4f})")


if __name__ == "__main__":
    main()
