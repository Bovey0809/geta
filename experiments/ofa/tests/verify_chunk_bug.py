"""Verify: does naive first-k output slicing break C2f/C3k2's internal chunk(2,1)?

C2f.forward does `y = list(self.cv1(x).chunk(2, 1))`, where cv1 outputs 2*c
channels and the SEMANTIC boundary between the two branches is at index c.

Naive elastic slicing takes cv1's output as trained-channels [0, out_k) where
out_k = round(2c * w). chunk(2,1) then splits THAT into [0, out_k/2) and
[out_k/2, out_k). The question: does the second chunk half stay inside the
second semantic half [c, 2c)?
"""
import torch
from ultralytics.nn.modules.block import C2f, C3k2, SPPF
import inspect

print("=== C2f.forward source ===")
print(inspect.getsource(C2f.forward))

print("=== SPPF.__init__ signature ===")
print(inspect.signature(SPPF.__init__))
print("=== SPPF.forward source ===")
print(inspect.getsource(SPPF.forward))

print("\n=== chunk-boundary arithmetic for a C3k2 ===")
# yolo26s layer 6: C3k2(256 -> 256, n=1, shortcut=True, e=0.5) => c = 128, cv1 out = 256
for (name, c2, e) in [("L2 C3k2(128,e=.25)", 128, 0.25),
                      ("L6 C3k2(256,e=.5)", 256, 0.5),
                      ("L13 C3k2(256,e=.5)", 256, 0.5)]:
    c = int(c2 * e)
    cv1_out = 2 * c
    print(f"\n{name}: hidden c={c}, cv1 out={cv1_out}, semantic boundary at index {c}")
    for w in (1.0, 0.95, 0.75, 0.5):
        out_k = max(1, int(round(cv1_out * w)))
        h = out_k // 2
        halfA = (0, h)
        halfB = (h, out_k)
        crosses = halfB[0] < c < halfB[1]
        wrong_lo = halfB[0] < c  # part of halfB comes from semantic half 1
        print(f"  w={w:<5} out_k={out_k:<4} halfA=[{halfA[0]},{halfA[1]}) "
              f"halfB=[{halfB[0]},{halfB[1]})  "
              f"halfB crosses semantic boundary={crosses}  "
              f"halfB polluted by semantic-half-1={wrong_lo}")

print("\n=== numerical proof on a real C3k2 ===")
import sys
sys.path.insert(0, "/home/rick/geta/experiments/ofa")

# Minimal reimplementation of the elastic conv patch (naive vs block-correct)
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

torch.manual_seed(0)
blk = C3k2(256, 256, n=1, shortcut=True).eval()
c = blk.c
x = torch.randn(1, 256, 32, 32)

with torch.no_grad():
    full = blk(x)

    # Emulate naive slicing of cv1's output at w=0.5 and see what the two chunk
    # halves actually contain vs. what they SHOULD contain.
    w = 0.5
    cv1_full = blk.cv1(x)                       # [1, 2c, H, W]
    out_k = int(round(2 * c * w))
    naive = cv1_full[:, :out_k]                 # trained channels [0, out_k)
    nA, nB = naive.chunk(2, 1)

    k1 = int(round(c * w))
    blockA = cv1_full[:, :k1]                   # first k1 of semantic half 1
    blockB = cv1_full[:, c:c + k1]              # first k1 of semantic half 2

    print(f"c={c}  w={w}  out_k={out_k}  k1={k1}")
    print(f"naive halfA == block halfA ? {torch.equal(nA, blockA)}")
    print(f"naive halfB == block halfB ? {torch.equal(nB, blockB)}")
    print(f"naive halfB vs block halfB max|diff| = {(nB - blockB).abs().max().item():.4f}")
    # Show naive halfB is actually a slice of semantic half 1
    print(f"naive halfB == cv1_full[:, {k1}:{k1+k1}] (still inside semantic half 1)? "
          f"{torch.equal(nB, cv1_full[:, k1:k1+k1])}")
