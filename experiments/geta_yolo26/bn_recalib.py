"""After one-shot magnitude pruning, recalibrate BatchNorm running stats by forwarding
a few hundred COCO train images (NO gradients, no weight updates), then re-evaluate.
Reports mAP before vs after BN-recal at each sparsity. Usage:
  PYTHONPATH=/root/geta python bn_recalib.py --model yolo26x.pt --sparsities 0.02,0.05,0.1,0.2
"""
import argparse, os, json
import torch
import torch.nn as nn
from only_train_once import OTO
from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from sanity_check.test_yolo26 import yolo26_unprunable_names

HERE = os.path.dirname(__file__)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over the calibration batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26x.pt")
    ap.add_argument("--data", default=os.path.join(HERE, "coco.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sparsities", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--calib_batches", type=int, default=50)
    args = ap.parse_args()
    spars = [float(s) for s in args.sparsities.split(",")]

    data = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    calib_ds = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="val", stride=32)
    calib_loader = build_dataloader(calib_ds, args.batch, workers=4, shuffle=True)

    def prune_fresh(s):
        # magnitude pruning is deterministic, so a fresh prune reproduces the same subnet
        model = YOLO(args.model).model
        for n, p in model.named_parameters():
            if "running_mean" not in n:
                p.requires_grad = True
        oto = OTO(model, torch.rand(1, 3, args.imgsz, args.imgsz))
        oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        oto._graph.magnitude_set_zero_groups(target_group_sparsity=s)
        return model.cuda()

    def val(model):
        # NOTE: y.val() fuses Conv+BN, so only call on a model you won't recalibrate after.
        y = YOLO(args.model)
        y.model = model.eval().cuda()
        m = y.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=0, verbose=False)
        return float(m.box.map), float(m.box.map50)

    for s in spars:
        # before: fresh pruned model, val directly (fuses it — fine, discarded after)
        pre_map, pre_map50 = val(prune_fresh(s))

        # after: a second fresh prune, recalibrate BN BEFORE any val/fusion, then val
        model = prune_fresh(s)
        reset_bn(model)
        model.train()
        seen = 0
        with torch.no_grad():
            for batch in calib_loader:
                imgs = batch["img"].cuda().float() / 255.0
                model(imgs)
                seen += 1
                if seen >= args.calib_batches:
                    break
        post_map, post_map50 = val(model)

        rec = {"model": args.model, "sparsity": s,
               "map_before": round(pre_map, 4), "map_after_bnrecal": round(post_map, 4),
               "map50_before": round(pre_map50, 4), "map50_after": round(post_map50, 4),
               "calib_imgs": seen * args.batch}
        print("BNRECAL", json.dumps(rec))
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
