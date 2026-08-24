"""Convert a locally-extracted VOCdevkit into the YOLO layout ultralytics expects.

WHY VOC AT ALL
--------------
Every OFA result in this study is on COCO, and all of them are negative. But
"negative" is consistent with two very different causes:
  (a) our OFA pipeline is broken, or
  (b) yolo26-on-COCO genuinely has no redundant capacity to exploit.
The 187 correctness tests prove the *slicing* is exact; they say nothing about
whether the full pipeline (sandwich training -> usable sub-nets) can ever work.

VOC settles it. 16.5k images / 20 classes means yolo26s is heavily
OVER-parameterised, so redundancy certainly exists. If OFA fails here too, the
approach is wrong. If it works here, the machinery is validated and the COCO
result becomes a real statement about yolo26/COCO rather than a possible bug.

VOC is also cheap enough to run TRUE OFA -- a supernet trained as a supernet
from random init, sandwich-sampled throughout. Every elastic failure in this
study so far was *post-hoc* elasticity on an already-converged checkpoint, which
is not what OFA actually prescribes. On COCO that experiment costs 2-3x a full
run; on VOC it costs hours.

FIDELITY
--------
Class order, the `difficult != 1` filter, and the box formula (including its
`-1` offsets) are copied from ultralytics' own VOC.yaml download script, so the
labels are byte-comparable with the standard pipeline. The only change is
reading an already-extracted VOCdevkit instead of downloading ~2.8 GB.

Standard benchmark split: train on VOC2007 trainval + VOC2012 trainval
(16,551 images), evaluate on VOC2007 test (4,952 images).

Usage:
  python experiments/ofa/prepare_voc.py \
      --devkit /root/autodl-tmp/VOC_raw/VOCdevkit \
      --out /root/autodl-tmp/VOC
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

SETS = [("2012", "train"), ("2012", "val"),
        ("2007", "train"), ("2007", "val"), ("2007", "test")]


def convert_box(size, box):
    """Exactly ultralytics' formula, including the -1 offsets."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(devkit: Path, lb_path: Path, year: str, image_id: str) -> int:
    src = devkit / f"VOC{year}/Annotations/{image_id}.xml"
    root = ET.parse(src).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    n = 0
    with open(lb_path, "w", encoding="utf-8") as out:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in NAMES:
                continue
            if int(obj.find("difficult").text) == 1:  # standard VOC practice
                continue
            bb = obj.find("bndbox")
            box = [float(bb.find(k).text) for k in ("xmin", "xmax", "ymin", "ymax")]
            out.write(" ".join(str(a) for a in (NAMES.index(cls), *convert_box((w, h), box))) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit", default="/root/autodl-tmp/VOC_raw/VOCdevkit")
    ap.add_argument("--out", default="/root/autodl-tmp/VOC")
    ap.add_argument("--link", action="store_true",
                    help="hardlink images instead of copying (same filesystem only)")
    args = ap.parse_args()

    devkit, out = Path(args.devkit), Path(args.out)
    if not devkit.exists():
        print(f"ABORT: {devkit} not found")
        return 1

    total_imgs = total_boxes = 0
    for year, split in SETS:
        ids_file = devkit / f"VOC{year}/ImageSets/Main/{split}.txt"
        if not ids_file.exists():
            print(f"  skip {split}{year}: {ids_file} missing")
            continue
        ids = ids_file.read_text().strip().split()
        img_dir = out / "images" / f"{split}{year}"
        lbl_dir = out / "labels" / f"{split}{year}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        boxes = 0
        for image_id in ids:
            src_img = devkit / f"VOC{year}/JPEGImages/{image_id}.jpg"
            dst_img = img_dir / f"{image_id}.jpg"
            if not dst_img.exists():
                if args.link:
                    try:
                        dst_img.hardlink_to(src_img)
                    except OSError:
                        shutil.copy2(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
            boxes += convert_label(devkit, lbl_dir / f"{image_id}.txt", year, image_id)
        print(f"  {split}{year}: {len(ids)} images, {boxes} boxes")
        total_imgs += len(ids)
        total_boxes += boxes

    # dataset yaml, standard benchmark split
    yaml_path = out / "VOC.yaml"
    lines = [
        f"path: {out}",
        "train:",
        "  - images/train2012",
        "  - images/train2007",
        "  - images/val2012",
        "  - images/val2007",
        "val:",
        "  - images/test2007",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    yaml_path.write_text("\n".join(lines) + "\n")

    print(f"\ntotal {total_imgs} images, {total_boxes} boxes")
    print(f"wrote {yaml_path}")
    print("train = VOC07+12 trainval (expect 16551), val = VOC07 test (expect 4952)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
