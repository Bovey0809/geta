"""Initialise the width-0.375 model from the public Objects365 yolo26s checkpoint.

WHY NOT RANDOM INIT
-------------------
Every yolo26 release was trained as *150 epochs Objects365 -> COCO fine-tune*,
and the COCO stage is only ~70 epochs precisely because it starts from a
pretrained backbone. There is no `objv1-150` checkpoint at width 0.375, and
training one would mean 150 epochs of Objects365. Instead we inherit that stage
by slicing the width-0.50 O365 checkpoint down to 0.375.

This is sound because the two architectures are channel-compatible by
construction: `make_divisible(c*0.375, 8)` is EXACTLY 0.75x
`make_divisible(c*0.50, 8)` for every base channel in yolo26, so the sliced
tensors match the target's shapes exactly rather than approximately.

WHAT GETS TRANSFERRED
---------------------
Backbone + neck (layers 0..22) only. The Detect head is deliberately skipped:
the source has nc=365 (Objects365) and the target nc=80 (COCO), so its head
must be reinitialised — which is exactly what the official pipeline does at the
start of its COCO stage.

Channels are importance-SORTED before slicing, so the retained 75 % are the
highest-|gamma| channels rather than an arbitrary 75 %. Gate B showed sorting
gives no zero-training benefit, but as an *initialisation* it is free and can
only help, and it is the one place the sorter earns its keep.

Usage:
  python experiments/ofa/build_w375_init.py \
      --src /root/yolo26s-objv1-150.pt \
      --cfg experiments/ofa/yolo26s-w375.yaml \
      --out /root/yolo26-w375-objv1-init.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from channel_plan import _IN_PLAN, _OUT_PLAN, install_elastic_conv  # noqa: E402
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_model  # noqa: E402
from sorter import sort_model  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.conv import Conv  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

W = 0.75  # 0.375 / 0.50


def transfer(src_model: nn.Module, dst_model: nn.Module, max_layer: int = 22):
    """Slice-copy every planned Conv in layers [0, max_layer] from src to dst."""
    src_named = dict(src_model.named_modules())
    dst_named = dict(dst_model.named_modules())

    copied = skipped = mismatched = 0
    details = []
    for name, sm in src_named.items():
        if not isinstance(sm, Conv):
            continue
        # "model.<i>. ..." -> keep backbone/neck only
        parts = name.split(".")
        if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit():
            continue
        if int(parts[1]) > max_layer:
            continue
        dm = dst_named.get(name)
        if not isinstance(dm, Conv):
            skipped += 1
            continue
        if not hasattr(sm, _OUT_PLAN):
            skipped += 1
            continue

        in_plan = getattr(sm, _IN_PLAN)
        out_plan = getattr(sm, _OUT_PLAN)
        sc, dc = sm.conv, dm.conv
        if sc.groups == 1:
            out_sel = out_plan.select(W)
            in_sel = in_plan.select(W)
            want = (out_sel.numel(), in_sel.numel())
            got = (dc.out_channels, dc.in_channels)
            if want != got:
                mismatched += 1
                details.append(f"    MISMATCH {name}: sliced {want} vs target {got}")
                continue
            dc.weight.data.copy_(sc.weight.data[out_sel][:, in_sel])
        else:
            sel = in_plan.select(W)
            if sel.numel() != dc.out_channels:
                mismatched += 1
                details.append(f"    MISMATCH {name} (dw): {sel.numel()} vs "
                               f"{dc.out_channels}")
                continue
            out_sel = sel
            dc.weight.data.copy_(sc.weight.data[sel])
        if sc.bias is not None and dc.bias is not None:
            dc.bias.data.copy_(sc.bias.data[out_sel])
        if isinstance(sm.bn, nn.BatchNorm2d) and isinstance(dm.bn, nn.BatchNorm2d):
            dm.bn.weight.data.copy_(sm.bn.weight.data[out_sel])
            dm.bn.bias.data.copy_(sm.bn.bias.data[out_sel])
            dm.bn.running_mean.data.copy_(sm.bn.running_mean.data[out_sel])
            dm.bn.running_var.data.copy_(sm.bn.running_var.data[out_sel])
        copied += 1
    return copied, skipped, mismatched, details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/yolo26s-objv1-150.pt")
    ap.add_argument("--cfg", default=str(_HERE / "yolo26s-w375.yaml"))
    ap.add_argument("--out", default="/root/yolo26-w375-objv1-init.pt")
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--no-sort", action="store_true")
    args = ap.parse_args()

    install_elastic_conv()
    install_elastic_attention()

    print(f"source: {args.src}")
    y_src = YOLO(args.src)
    src = y_src.model.float().eval()
    src_nc = getattr(src, "nc", None) or getattr(src.model[-1], "nc", None)
    src_p = sum(p.numel() for p in src.parameters())
    print(f"  nc={src_nc}  params={src_p / 1e6:.3f}M")

    plan_model(src)
    if not args.no_sort:
        sort_model(src)
        print("  channels importance-sorted (retained 75% = top 75%)")

    dst = DetectionModel(args.cfg, ch=3, nc=args.nc, verbose=False).float().eval()
    dst_p = sum(p.numel() for p in dst.parameters())
    print(f"target: {args.cfg}\n  nc={args.nc}  params={dst_p / 1e6:.3f}M "
          f"({dst_p / src_p * 100:.1f}% of source)")

    copied, skipped, mismatched, details = transfer(src, dst)
    print(f"\ntransferred {copied} convs; skipped {skipped}; mismatched {mismatched}")
    for d in details[:10]:
        print(d)
    if mismatched:
        print("\nABORT: shape mismatches mean the architectures are not "
              "0.75x-compatible as assumed.")
        return 1
    if copied == 0:
        print("\nABORT: nothing transferred -- this would silently train from "
              "random init and look like a valid result.")
        return 1

    # sanity: the transferred model must produce a finite forward
    with torch.no_grad():
        out = dst(torch.randn(1, 3, 256, 256))
    o = out[0] if isinstance(out, (list, tuple)) else out
    assert torch.isfinite(o).all(), "non-finite forward after transfer"
    print(f"forward OK, out={tuple(o.shape)}")

    ckpt = {"model": dst.half(), "epoch": -1, "best_fitness": None,
            "date": None, "version": None,
            "train_args": {"note": "width-0.375 init: yolo26s-objv1-150 sliced "
                                   "at w=0.75 (importance-sorted), head reinit "
                                   "for nc=80"}}
    torch.save(ckpt, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
