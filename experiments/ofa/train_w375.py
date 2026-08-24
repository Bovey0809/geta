"""COCO stage for the width-0.375 model, using the OFFICIAL yolo26s recipe.

The recipe below is not guessed or copied from the docs tables -- it is read
verbatim from `yolo26s.pt`'s own `train_args`, which is the checkpoint the
official COCO stage produced. The docs explicitly say the tables omit smaller
non-default arguments and to read the checkpoint instead.

Two things this pins down that a docs-derived recipe would have got wrong:
  * **70 epochs**, not 80 (the number quoted in earlier notes);
  * `mixup 0.05` and `scale 0.9` for s, whereas the m recipe uses
    `mixup 0.427` / `scale 0.95` -- augmentation really is per-size, so using
    m's values would have handicapped this run.

Batch is kept at the official 128. Ultralytics does not auto-scale `lr0` with
batch size, so halving the batch would double the per-sample learning rate and
silently stop being the official recipe; if it does not fit, that is worth
saying rather than quietly adjusting.

Usage:
  python experiments/ofa/train_w375.py --model /root/w375_init.pt \
      --data experiments/ofa/coco.yaml --name w375_coco
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# verbatim from yolo26s.pt train_args (the official COCO stage for size s)
OFFICIAL_S_COCO = dict(
    epochs=70,
    batch=128,
    optimizer="MuSGD",
    lr0=0.00038,
    lrf=0.88219,
    momentum=0.94751,
    weight_decay=0.00027,
    warmup_epochs=0.98745,
    warmup_momentum=0.54064,
    warmup_bias_lr=0.05684,
    box=9.83241,
    cls=0.64896,
    dfl=0.95824,
    mosaic=0.99182,
    mixup=0.05,
    copy_paste=0.40413,
    cutmix=0.00082,
    scale=0.9,
    translate=0.27484,
    degrees=0.00012,
    shear=0.00136,
    perspective=0.00074,
    fliplr=0.30393,
    flipud=0.00653,
    hsv_h=0.01315,
    hsv_s=0.35348,
    hsv_v=0.19383,
    close_mosaic=10,
    imgsz=640,
    cos_lr=False,
    nbs=64,
    amp=True,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/w375_init.pt")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "coco.yaml"))
    ap.add_argument("--name", default="w375_coco")
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="override for smokes")
    ap.add_argument("--batch", type=int, default=None, help="override if OOM")
    ap.add_argument("--fraction", type=float, default=None, help="override for smokes")
    args = ap.parse_args()

    cfg = dict(OFFICIAL_S_COCO)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
        print(f"!! batch overridden to {args.batch} (official 128). Ultralytics "
              f"does not scale lr0 with batch, so this is NO LONGER the official "
              f"recipe and the result is not directly comparable.")
    if args.fraction is not None:
        cfg["fraction"] = args.fraction

    print("=" * 66)
    print(f"width-0.375 COCO stage | init={args.model}")
    print(f"recipe: official yolo26s COCO stage ({cfg['epochs']} ep, "
          f"batch {cfg['batch']}, {cfg['optimizer']})")
    print("=" * 66, flush=True)

    y = YOLO(args.model)
    y.train(data=args.data, name=args.name, device=args.device,
            val=True, plots=False, save_json=False, **cfg)
    print("W375_TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
