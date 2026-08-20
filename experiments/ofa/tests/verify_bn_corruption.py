"""Verify: do width<1.0 forward passes corrupt the SHARED BN running stats?

`_elastic_conv_forward` passes `bn.running_mean[:out_k]` to F.batch_norm. A basic
slice is a VIEW sharing storage with the full buffer. F.batch_norm(training=True)
updates running stats IN PLACE -> a w=0.5 pass rewrites the first 50% of every
BN's running stats with half-width statistics, which the w=1.0 pass then reads.

torch.no_grad() does NOT prevent buffer mutation, so even a "frozen teacher"
pass in train mode rewrites stats.

Prediction: after one train-mode forward at w=0.5, running_mean[:k] changes and
running_mean[k:] does not.
"""
import sys
sys.path.insert(0, "/home/rick/geta/experiments/ofa")

import torch
import torch.nn as nn
import width_elastic  # applies the patches
from width_elastic import set_width
from ultralytics.nn.tasks import DetectionModel

torch.manual_seed(0)
m = DetectionModel("yolo26s.yaml", ch=3, nc=80, verbose=False)

# Pick a mid-network BN that is elastic (not inside Detect/attention).
target = m.model[4].cv1  # C3k2 L4 cv1
bn = target.bn
full_c = bn.running_mean.numel()

x = torch.randn(2, 3, 256, 256)

print(f"target BN: {full_c} channels (model.4.cv1.bn)")

# --- Case 1: eval mode at w=0.5 (should NOT touch buffers) ---
m.eval()
set_width(m, 0.5)
before = bn.running_mean.clone()
with torch.no_grad():
    m(x)
after_eval = bn.running_mean.clone()
print(f"\n[eval mode, w=0.5]  buffer changed: {not torch.equal(before, after_eval)}  "
      f"(expected: False)")

# --- Case 2: TRAIN mode at w=0.5 under no_grad (the 'frozen teacher' pattern) ---
m.train()
set_width(m, 0.5)
before = bn.running_mean.clone()
with torch.no_grad():
    m(x)
after_train = bn.running_mean.clone()

k = max(1, int(round(full_c * 0.5)))
inner_changed = not torch.equal(before[:k], after_train[:k])
outer_changed = not torch.equal(before[k:], after_train[k:])
print(f"\n[train mode, w=0.5, under no_grad]")
print(f"  running_mean[:{k}]  changed: {inner_changed}   (expected: True  <-- CORRUPTION)")
print(f"  running_mean[{k}:]  changed: {outer_changed}   (expected: False <-- stale/mixed)")
print(f"  max|delta| inner = {(before[:k] - after_train[:k]).abs().max().item():.6f}")
print(f"  max|delta| outer = {(before[k:] - after_train[k:]).abs().max().item():.6f}")
print(f"  -> buffer is now a MIX of w=0.5 stats (inner) and w=1.0 stats (outer)")

# --- Case 3: does a w=1.0 pass then read the corrupted inner region? ---
m.eval()
set_width(m, 1.0)
with torch.no_grad():
    out_corrupt = m(x)
# restore clean stats and compare
bn.running_mean.data.copy_(before)
with torch.no_grad():
    out_clean = m(x)
o1 = out_corrupt[0] if isinstance(out_corrupt, (list, tuple)) else out_corrupt
o2 = out_clean[0] if isinstance(out_clean, (list, tuple)) else out_clean
print(f"\n[w=1.0 eval] output differs after ONE corrupting pass on ONE BN: "
      f"max|diff| = {(o1 - o2).abs().max().item():.6f}")
print("  (a real run corrupts ALL ~60 elastic BNs, every batch, for 370 batches/epoch)")
