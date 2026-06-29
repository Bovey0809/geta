import sys, argparse
sys.path.insert(0, "/root/geta/experiments/ofa")
from elastic_yolo26 import set_depth  # applies global C3k elastic patch on import
from ultralytics import YOLO

DEPTHS = [2, 1]  # sandwich: full + min

def patch_sandwich(trainer):
    if getattr(trainer, "ema", None) is not None:
        trainer.ema.enabled = False
    m = trainer.model
    orig = m.loss
    def sloss(batch, preds=None):
        tot, items = None, None
        for d in DEPTHS:
            set_depth(m, d)
            l, it = orig(batch) if preds is None else orig(batch, preds)
            tot = l if tot is None else tot + l
            items = it
        return tot, items
    m.loss = sloss
    print("[sandwich] loss patched; depths", DEPTHS, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="ofa_l_sandwich")
    args = ap.parse_args()
    y = YOLO("yolo26l.pt")
    y.add_callback("on_train_start", patch_sandwich)
    y.train(data="/root/geta/experiments/geta_yolo26/coco.yaml", epochs=args.epochs,
            batch=args.batch, imgsz=640, fraction=args.fraction, lr0=1e-3, amp=False,
            warmup_epochs=0, optimizer="SGD", nbs=args.batch, name=args.name,
            val=False, plots=False, device=0)
    print("SANDWICH_TRAIN_DONE", flush=True)
