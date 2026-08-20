"""Full per-module inventory of yolo26s: internal cat/chunk structure + constraints.

Built from the yaml (no weights needed) so we can enumerate exactly what every
top-level layer does to channels, which is what the elastic slicing must respect.
"""
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import (
    Attention, Bottleneck, C2f, C2PSA, C3, C3k, C3k2, PSABlock, SPPF,
)
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.modules.head import Detect

m = DetectionModel("yolo26s.yaml", ch=3, nc=80, verbose=False)
top = m.model

def has_attn(mod):
    return any(isinstance(s, Attention) for s in mod.modules())

print(f"{'idx':>3} {'type':<10} {'f':<14} {'out_c':>6}  internals")
print("-" * 110)
for i, L in enumerate(top):
    t = type(L).__name__
    f = getattr(L, "f", "-")
    notes = []
    out_c = "?"
    if isinstance(L, Conv):
        out_c = L.conv.out_channels
        notes.append(f"plain conv {L.conv.in_channels}->{out_c} g={L.conv.groups}")
    elif isinstance(L, (C2f, C3k2)):
        out_c = L.cv2.conv.out_channels
        c = L.c
        nseg = 2 + len(L.m)
        notes.append(f"hidden c={c}")
        notes.append(f"cv1 {L.cv1.conv.in_channels}->{L.cv1.conv.out_channels} (=2c) CHUNK2 @ {c}")
        notes.append(f"cv2 in={L.cv2.conv.in_channels} (={nseg}*c, {nseg} CAT segs) ->{out_c}")
        notes.append(f"m=[{','.join(type(x).__name__ for x in L.m)}] attn={has_attn(L)}")
        for j, mm in enumerate(L.m):
            if isinstance(mm, Bottleneck):
                notes.append(f"  m[{j}] Bottleneck add={mm.add} "
                             f"cv1 {mm.cv1.conv.in_channels}->{mm.cv1.conv.out_channels} "
                             f"cv2 {mm.cv2.conv.in_channels}->{mm.cv2.conv.out_channels}")
            elif isinstance(mm, C3k):
                notes.append(f"  m[{j}] C3k c={mm.c_ if hasattr(mm,'c_') else '?'} "
                             f"cv1={mm.cv1.conv.in_channels}->{mm.cv1.conv.out_channels} "
                             f"cv2={mm.cv2.conv.in_channels}->{mm.cv2.conv.out_channels} "
                             f"cv3={mm.cv3.conv.in_channels}->{mm.cv3.conv.out_channels} "
                             f"nB={len(mm.m)} addB={[b.add for b in mm.m]}")
            elif isinstance(mm, nn.Sequential):
                notes.append(f"  m[{j}] Sequential[{','.join(type(z).__name__ for z in mm)}]")
    elif isinstance(L, SPPF):
        out_c = L.cv2.conv.out_channels
        nrep = 1 + getattr(L, "n", 3)
        notes.append(f"cv1 {L.cv1.conv.in_channels}->{L.cv1.conv.out_channels}")
        notes.append(f"cv2 in={L.cv2.conv.in_channels} (={nrep} REPEATED CAT segs)->{out_c}")
        notes.append(f"n={getattr(L,'n',3)} add={getattr(L,'add',False)} <-- residual y+x")
    elif isinstance(L, C2PSA):
        out_c = L.cv2.conv.out_channels
        notes.append(f"c={L.c} cv1->{L.cv1.conv.out_channels} SPLIT2 @ {L.c}")
        notes.append(f"cv2 in={L.cv2.conv.in_channels}->{out_c} attn={has_attn(L)} FROZEN")
    elif isinstance(L, Concat):
        notes.append(f"cat dim={L.d} sources={f}")
    elif isinstance(L, nn.Upsample):
        notes.append("passthrough")
    elif isinstance(L, Detect):
        notes.append(f"nl={L.nl} reg_max={getattr(L,'reg_max','?')} e2e={getattr(L,'end2end',False)}")
        for a in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            b = getattr(L, a, None)
            if b is not None:
                firsts = []
                for seq in b:
                    fc = next((s for s in seq.modules() if isinstance(s, Conv)), None)
                    firsts.append(fc.conv.in_channels if fc else "?")
                notes.append(f"  {a}: first-conv in_c per scale = {firsts}")
    print(f"{i:>3} {t:<10} {str(f):<14} {str(out_c):>6}  {notes[0] if notes else ''}")
    for n in notes[1:]:
        print(f"{'':>3} {'':<10} {'':<14} {'':>6}  {n}")
