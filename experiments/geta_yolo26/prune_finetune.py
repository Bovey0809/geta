"""Prune + fine-tune a yolo26 model on COCO with GETA, then construct the pruned
subnet and (optionally) report COCO mAP. Usage:
  PYTHONPATH=/root/geta python prune_finetune.py --model yolo26n.pt --data .../coco.yaml \
      --epochs 100 --batch 16 --sparsity 0.5 --name geta_n_s50
"""
import argparse, os, json, glob
import torch

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--data", default=os.path.join(HERE, "coco.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--name", default="geta_ft")
    ap.add_argument("--device", default="0")
    ap.add_argument("--val", action="store_true", help="run COCO val on the pruned subnet at the end")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "/root/geta")
    from experiments.geta_yolo26.geta_trainer import GetaDetectionTrainer
    from sanity_check.test_yolo26 import _max_diff

    GetaDetectionTrainer.geta_sparsity = args.sparsity
    overrides = dict(model=args.model, data=args.data, epochs=args.epochs, batch=args.batch,
                     imgsz=args.imgsz, device=args.device, name=args.name,
                     amp=False, warmup_epochs=0.0, nbs=args.batch, val=False, plots=False,
                     optimizer="auto")
    trainer = GetaDetectionTrainer(overrides=overrides)
    trainer.train()

    out = os.path.join(HERE, "out", args.name)
    os.makedirs(out, exist_ok=True)
    trainer.oto.construct_subnet(out_dir=out)

    # verify the constructed subnet matches the trained (sparsified) model
    full = torch.load(trainer.oto.full_group_sparse_model_path, weights_only=False)
    comp = torch.load(trainer.oto.compressed_model_path, weights_only=False)
    x = torch.rand(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        diff = _max_diff(full(x), comp(x))
    n_full = sum(p.numel() for p in full.parameters()) / 1e6
    n_comp = sum(p.numel() for p in comp.parameters()) / 1e6
    print(f"SUBNET path={trainer.oto.compressed_model_path} params {n_full:.3f}M->{n_comp:.3f}M "
          f"construct_diff={diff:.3e}")

    if args.val:
        from ultralytics import YOLO
        y = YOLO(args.model)
        y.model = comp.eval().cuda()
        m = y.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=int(args.device))
        rec = {"model": args.model, "sparsity": args.sparsity, "params_M": round(n_comp, 4),
               "map5095": float(m.box.map), "map50": float(m.box.map50)}
        json.dump(rec, open(os.path.join(out, "pruned_val.json"), "w"), indent=2)
        print("PRUNED_VAL", json.dumps(rec))


if __name__ == "__main__":
    main()
