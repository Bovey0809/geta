"""Zero-cost structural diagnosis of YOLO26-l depth elasticity.
Determines whether the elastic axis (currently C3k inner bottlenecks) is the right one,
and whether the d=1 forward produces sane (finite, above-threshold) raw detections.
"""
import sys
sys.path.insert(0, "/root/geta/experiments/ofa")
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k, C3k2
from elastic_yolo26 import set_depth  # applies C3k elastic patch

y = YOLO("yolo26l.pt")
m = y.model.eval()

print("=== module hierarchy (depth-bearing blocks) ===")
n_c3k2 = n_c3k = 0
for name, mod in m.named_modules():
    if isinstance(mod, C3k2):
        n_c3k2 += 1
        inner = list(mod.m)
        kinds = [type(b).__name__ for b in inner]
        print(f"C3k2 {name}: m has {len(inner)} -> {kinds}")
    elif isinstance(mod, C3k):
        n_c3k += 1
        inner = list(mod.m)
        sc = [getattr(b, 'add', '?') for b in inner]  # Bottleneck.add == residual shortcut active
        print(f"  C3k {name}: m has {len(inner)} bottlenecks, shortcut(add)={sc}")
print(f"TOTALS: C3k2={n_c3k2} C3k={n_c3k}")

print("\n=== raw output sanity at each depth (1 random image) ===")
x = torch.rand(1, 3, 640, 640)
for d in [2, 1]:
    set_depth(m, d)
    with torch.no_grad():
        out = m(x)
    t = out[0] if isinstance(out, (list, tuple)) else out
    # YOLO detect raw output: (1, 4+nc, num_anchors) or list; inspect the primary tensor
    print(f"d={d}: type={type(out).__name__} primary shape={tuple(t.shape)} "
          f"finite={torch.isfinite(t).all().item()} min={t.min().item():.3f} "
          f"max={t.max().item():.3f} mean={t.mean().item():.4f}")

print("\n=== predict() detection count at each depth (conf=0.001) ===")
for d in [2, 1]:
    set_depth(m, d)
    r = y.predict(x, conf=0.001, verbose=False, device="cpu")
    nb = len(r[0].boxes) if r and r[0].boxes is not None else 0
    print(f"d={d}: n_detections(conf>0.001)={nb}")
print("INSPECT_DONE")
