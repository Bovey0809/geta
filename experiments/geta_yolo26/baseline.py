import json, os
from ultralytics import YOLO

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(os.path.dirname(__file__), "coco.yaml")

def main():
    model = YOLO("yolo26n.pt")  # downloads pretrained weights
    metrics = model.val(data=DATA, imgsz=640, batch=16, device=0)
    # n_params / GFLOPs from the underlying module
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    rec = {
        "model": "yolo26n",
        "map5095": float(metrics.box.map),     # mAP50-95
        "map50": float(metrics.box.map50),
        "params_M": round(n_params, 4),
    }
    with open(os.path.join(OUT, "baseline_metrics.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("BASELINE", json.dumps(rec))

if __name__ == "__main__":
    main()
