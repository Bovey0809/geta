"""No-retraining magnitude-pruning sweep: zero the lowest-L2-norm groups at several
sparsities, measure COCO mAP (on the zeroed model — identical to the constructed
model's accuracy), and report the real pruned param count via construct_subnet.
Finds the max sparsity that keeps mAP at baseline. Usage:
  PYTHONPATH=/root/geta python magnitude_sweep.py --model yolo26n.pt
"""
import argparse, os, json
import torch
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names, _max_diff

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--data", default=os.path.join(HERE, "coco.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sparsities", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    args = ap.parse_args()
    spars = [float(s) for s in args.sparsities.split(",")]
    out = os.path.join(HERE, "out", "magnitude_sweep")
    os.makedirs(out, exist_ok=True)

    results = []
    for s in spars:
        model = YOLO(args.model).model
        for n, p in model.named_parameters():
            if "running_mean" not in n:
                p.requires_grad = True
        oto = OTO(model, torch.rand(1, 3, args.imgsz, args.imgsz))
        oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        oto._graph.magnitude_set_zero_groups(target_group_sparsity=s)

        # accuracy: val the zeroed model directly (== constructed model's mAP)
        y = YOLO(args.model)
        y.model = model.eval().cuda()
        m = y.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=0, verbose=False)
        map5095, map50 = float(m.box.map), float(m.box.map50)

        # real pruned param count via construct (verify consistency)
        pruned_params, diff = None, None
        try:
            oto.construct_subnet(out_dir=out)
            comp = torch.load(oto.compressed_model_path, weights_only=False).cpu().eval()
            full = torch.load(oto.full_group_sparse_model_path, weights_only=False).cpu().eval()
            x = torch.rand(1, 3, args.imgsz, args.imgsz)
            with torch.no_grad():
                diff = _max_diff(full(x), comp(x))
            pruned_params = round(sum(p.numel() for p in comp.parameters()) / 1e6, 4)
        except Exception as e:
            diff = f"construct_err:{str(e)[:60]}"
        rec = {"sparsity": s, "map5095": round(map5095, 4), "map50": round(map50, 4),
               "pruned_params_M": pruned_params, "construct_diff": diff}
        results.append(rec)
        print("SWEEP", json.dumps(rec))
        del model, oto, y
        torch.cuda.empty_cache()

    json.dump(results, open(os.path.join(out, "sweep.json"), "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
