"""Calibrate KD scale: measure the natural feature-MSE gap between d=2 and d=1 neck
features (Detect inputs) on the pretrained-l over a few real COCO val batches, plus the
detection-loss scale, so kd_lambda can be set so KD is comparable to the det loss.
"""
import sys
sys.path.insert(0, "/root/geta/experiments/ofa")
import torch, torch.nn.functional as F
from ultralytics import YOLO
from elastic_yolo26 import set_depth

y = YOLO("yolo26l.pt")
m = y.model.eval().cuda()
detect = m.model[-1]
cap = {}
detect.register_forward_pre_hook(lambda mod, args: cap.__setitem__("f", args[0]))

# a few real val images via the dataloader-free path: use random? no -> use val images
import glob, cv2, numpy as np
imgs = sorted(glob.glob("/root/autodl-tmp/coco/images/val2017/*.jpg"))[:16]
def load(p):
    im = cv2.imread(p); im = cv2.resize(im, (640, 640))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype("float32") / 255.0
    return torch.from_numpy(im)
x = torch.stack([load(p) for p in imgs]).cuda()

with torch.no_grad():
    set_depth(m, 2); _ = m(x); tf = [f.clone() for f in cap["f"]]
    set_depth(m, 1); _ = m(x); sf = [f.clone() for f in cap["f"]]
print("=== neck feature stats (Detect inputs) per scale ===")
tot_mse = 0.0
for i, (t, s) in enumerate(zip(tf, sf)):
    mse = F.mse_loss(s, t).item(); tot_mse += mse
    print(f"scale{i}: shape={tuple(t.shape)} teacher|mean|={t.abs().mean().item():.3f} "
          f"student|mean|={s.abs().mean().item():.3f} MSE(d1,d2)={mse:.4f} "
          f"relMSE={mse / (t.pow(2).mean().item() + 1e-9):.3f}")
print(f"SUM feature MSE(d1 vs d2) = {tot_mse:.4f}")
print("NOTE: det losses (box+cls+dfl) are typically O(1-3) each; pick kd_lambda so "
      f"kd_lambda*{tot_mse:.3f} ~ O(1). suggested kd_lambda ~ {1.0/max(tot_mse,1e-3):.2f}")
print("CALIB_DONE")
