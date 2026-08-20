> # ⚠ CONCLUSION RETRACTED (2026-08-20)
> The negative verdict in this document is **withdrawn**. The `0.0 mAP` results were
> caused by two implementation bugs, both since confirmed by direct measurement:
> **(A)** naive first-k slicing violates the internal `chunk(2,1)`/`cat` boundaries
> inside every C3k2 / nested C3k / SPPF block — **23 sites** in yolo26s, proven
> numerically (at w=0.5 the residual branch is fed *entirely from the wrong semantic
> half*); **(B)** `bn.running_mean[:k]` is a **view**, so every train-mode narrow pass
> — including the `no_grad` "frozen teacher" — overwrote the inner region of all ~60
> elastic BN buffers, mixing narrow-width and full-width statistics. The tell that
> should have caught this: `kd=10` and `kd=1000` collapsed near-identically, which no
> gradient-magnitude explanation fits.
>
> The `bn_recal.py` numbers here are also invalid — a correct recal at `w=1.0` must
> return ≈0.472 and it returned 0.345 (calibrated on *val* images).
>
> **Read `OFA_SUPERNET_PLAN.md` instead.** Reproduce the diagnoses with
> `tests/verify_chunk_bug.py` and `tests/verify_bn_corruption.py` (CPU, seconds).
> Everything below is kept as an accurate record of *what was run*, not of what it means.

# Width-elastic OFA on yolo26s — minimal PoC

Goal: build the smallest possible width-elastic OFA scaffold on yolo26s and
verify the properties needed before spending any GPU time on supernet training.
yolo26 n and s share depth=0.50; only width differs (n=0.25, s=0.50), so the
n↔s pair is intrinsically a WIDTH-elastic problem, not depth-elastic (which
was the dead end on yolo26l).

Hardware: `ofa_ws` — RTX PRO 6000 Blackwell 96GB, 西北B区, torch 2.12.1+cu130,
ultralytics 8.4.120. COCO val2017 at `/root/autodl-tmp/coco` (5000 images,
4952 labels — 48 val images have no annotations, expected).

## Mechanism (`experiments/ofa/width_elastic.py`)

Class-level monkey-patch of `Conv.forward` AND `Conv.forward_fuse` (fusion
swaps `.forward = .forward_fuse` at inference time; without patching both,
`val()` silently bypassed the elastic path — that was the "check 4 = 0.4717"
red herring during development). Each Conv reads its own `_active_width` (0,1]:

- **Fast path** when incoming `x.shape[1] == conv.in_channels` and the requested
  `out_k == out_full` and `groups == 1`: falls through to the original op,
  bit-identical.
- **Sliced path** otherwise: `weight[:out_k, :in_k]`, `bn.{weight,bias,mean,var}[:out_k]`
  (or fused `bias[:out_k]` after fuse). Views onto the trained parameters, so
  gradients flow back to shared weights.
- **DWConv**: `groups == in_channels == out_channels`; out is structurally
  forced to equal `in_k` regardless of the requested width.

`set_width(model, w)` writes `_active_width` on every Conv. Two categories are
FROZEN at 1.0:
- **`Detect`** — fixed output dims (`nc + reg*4`), can't change.
- **Any top-level `model[i]` block containing an `Attention` module** — this
  covers `C2PSA` at model.10 AND `C3k2(attn=True)` at model.22. Their
  attention internals bake `num_heads/key_dim/head_dim` at init and don't
  survive width slicing. Same exclusion GETA had to make for pruning
  (see `only_train_once/` history in the memory).

Result on yolo26s: **62 elastic Convs, 52 frozen** (of 114 total).

## PoC results

Five checks (`experiments/ofa/width_poc.py`):

| # | check | result |
|---|-------|--------|
| 1 | bit-identical forward at w=1.0 vs unpatched | max_abs_diff = **0.0** |
| 2 | forward at w=0.5 doesn't crash, output is finite | shape (1, 300, 6), finite ✓ |
| 3 | mAP50-95 at w=1.0 (baseline preservation) | **0.4717** (baseline: 0.472 ✓) |
| 4 | mAP50-95 at w=0.5 (untrained subnet) | **0.0000** |
| 5 | grad-flow: only `[:out_k, :in_k]` slice gets grad at w=0.5 | inside=3.0e+04, outside=−2e−3 (≈numerical noise, ratio 6e−8) ✓ |

## What check 4 means

**The width-elastic mechanism works, but a naive w=0.5 subnet of the pretrained
yolo26s is broken (0.0 mAP).** Two root causes, both structural, both fixable
only by supernet training:

1. **First-k channels are not the "most important" channels.** They're
   arbitrary — the pretrained weights weren't laid out so that the first half
   of each conv preserves the function of the whole. OFA (MIT) solves this
   with *progressive shrinking*: train large first, then shrink kernels/depth/
   width in stages, with each stage KD-guided from the previous. That reorders
   the weights so the first-k channels ARE the important ones.
2. **Concat channel alignment.** Every neck-side Concat consumer's Conv weight
   has columns laid out as `[source_A_full | source_B_full | …]`. If A and B
   are both sliced by w=0.5, the neck output has channels
   `[A_first_half | B_first_half]`. But `weight[:, :in_k]` sees columns
   `[A_first_half | first half of A_second_half]` — semantically wrong; the
   B-region columns are never touched. Fixing this requires per-source
   channel-offset bookkeeping at every Concat-fed Conv (what OFA's official
   impl does with "elastic channel maps").

Check 4 = 0.0 is therefore not evidence that width-elastic OFA can't work
on YOLO26 — it's evidence that supernet TRAINING is mandatory before any
subnet is usable (same as MIT OFA on ImageNet: their untrained MIN subnet
would also collapse; the published −7.5 pt drop at MIN reflects a fully
progressively-shrunk supernet).

## What check 5 means

Weight sharing is verified: after one backward pass at w=0.5, the
`[:out_k, :in_k]` slice of a mid-network Conv weight has grad magnitude
3.0e+04; the region outside that slice has grad ~2e-3 (i.e. ratio 6e-8,
essentially numerical noise from autograd). So the shared parameters used
by w=1.0 and w=0.5 forward passes will co-update — the OFA property required
for progressive shrinking.

## Next-step plan (out of scope for this PoC)

1. **Fix the Concat channel-alignment bug** — track per-source offsets at
   every neck concat consumer and slice `weight` as `cat([weight[:, off_i : off_i + k_i]])`.
2. **Progressive shrinking supernet training** on yolo26s:
   - stage 1: keep at w=1.0 for a few epochs (baseline confirmation with
     the elastic scaffold).
   - stage 2: sandwich {w=1.0, w=0.75} + feature-KD from teacher at w=1.0.
   - stage 3: sandwich {w=1.0, w=0.75, w=0.5} + feature-KD.
   - Verify at each stage that the smallest sub-net's mAP monotonically
     improves toward n's 0.395 baseline.
3. **Compare** the w=0.5 subnet (~2.5M params, same size as n) to the
   default yolo26n (0.395). If it exceeds n, the OFA supernet is delivering
   the promised "smaller-with-same-accuracy" property.

Attention blocks stay frozen at w=1.0 throughout — they cost parameters at
every sub-net, so the effective width at w=0.5 is not exactly 0.5 in params.
This is a known limitation of both this PoC and the depth-elastic prior work,
and matches what GETA had to accept for pruning.

## Reproduce

```bash
# on ofa_ws
python3 /root/geta/experiments/ofa/width_poc.py \
    --model /root/yolo26s.pt \
    --data /root/geta/experiments/ofa/coco.yaml \
    --device 0
```
Expected: `ALL_PASS` in ~30s (mostly val time).

---

## Update — progressive-shrinking training attempt (same day)

Followed the "next-step plan" above: implemented Concat channel-alignment
(`prepare_concat_alignment`, patched Concat.forward to record per-source runtime
sizes) so the two-elastic-source neck concats (yolo26s model.16 / model.19)
slice weight columns per source instead of contiguously. All 5 PoC checks still
pass. Then wrote `width_sandwich_train.py` and ran Stage 1: widths
`[1.0, 0.75]`, 3 epochs, fraction=0.1, batch=32, kd=10, lr=1e-4 on COCO.

Two Ultralytics interactions had to be worked around first:

1. **`save_model` writes `trainer.ema.ema`, not `trainer.model`.** With EMA
   disabled via `ema.enabled=False`, `ema.ema` stays at initial weights → the
   saved `last.pt` is byte-identical to `yolo26s.pt` even after training. Fix:
   `on_train_epoch_end` callback that does
   `tr.ema.ema.load_state_dict(tr.model.state_dict())`.
2. **Sandwich with `l_teacher + l_student + kd·KD` destabilises the pretrained
   teacher.** Teacher (0.472) is already near-optimal, student (0.0) has a
   huge loss (~340 vs teacher's ~64). Their gradients fight on the shared
   `weight[:0.75, :0.75]` region and destroy the teacher within one epoch
   (0.472 → 0.0001 mAP). Standard OFA-MIT pattern is "in-place distillation":
   forward teacher with `torch.no_grad()`, use its features only as KD
   targets. Switched to that.

### Results after all fixes

| widths | epochs × fraction | kd | teacher mAP after | student mAP after |
|---|---|---|---|---|
| [1.0, 0.75] | 3 × 0.1 | 10 | 0.4716 → **0.0001** (student's shared-weight grad kills it) | 0.0 → 0.0 |
| [1.0, 0.95] (narrow-gap diag) | 1 × 0.02 | 10 | 0.4716 → 0.0052 | 0.0 → 0.0 |
| [1.0, 0.95] + kd=1000 | 1 × 0.02 | 1000 | 0.4716 → 0.0052 | 0.0 → 0.0 |

Even a 5% width gap collapses the teacher on 74 batches, and even a KD weight
of 1000 (kd_lambda × KD ≈ 1000 vs det ≈ 320) can't hold it. Student stays at
0.0 in every configuration.

### Root cause (structural)

Sliced first-k channels in the pretrained network **are not in any semantically
meaningful order**. The trained convs distribute information across all
channels arbitrarily; naive `weight[:k, :k]` grabs an arbitrary subset. So:

- The student subnet's forward is fundamentally producing arbitrary-subset
  features. Feature-KD would need to *reorder the weights* to align the first-k
  channels with the "most important" ones — a global re-basis that gradient
  descent from the current basin can't reach in a few epochs.
- Any grad from the student that DOES flow back shifts the shared inner
  `weight[:k, :k]` slice. Teacher then reads a modified inner region + the
  unchanged outer region and its function is broken.

This is exactly what OFA-MIT's *progressive shrinking* solves — but only over
100+ epochs across multiple staged widths (`{1.0} → {1.0, 0.9} → {1.0, 0.9, 0.75}
→ …`), plus explicit importance-based channel sorting at each stage, plus
per-width BN recalibration at the end. That's a multi-day, full-COCO GPU
investment, not a 3-epoch smoke.

### Verdict for the yolo26 n↔s width-elastic supernet

Same wall as the two prior paradigms in `experiments/CONCLUSION.md`:

| paradigm | outcome | root cause |
|---|---|---|
| GETA structured pruning (yolo26x → 20 M) | 0.45 vs default-m 0.518 | net has no redundant capacity |
| depth-elastic OFA (yolo26l d=1) | 0.0 mAP | 2nd bottleneck is essential, not residual |
| **width-elastic OFA (yolo26s w<1)** | **0.0 mAP; naive sandwich destroys teacher** | **first-k channels aren't sorted; short-budget training can't reorder them** |

Width-elastic OFA on yolo26 is not obviously impossible, but making it work
would require the full MIT recipe (weight-importance sorting + progressive
shrinking + 100+ epochs + per-width BN recalibration). At the PoC scale here,
it fails the same way the other two paradigms did.

### Fixes shipped that outlive this negative result

- `Conv.forward` / `Conv.forward_fuse` monkey-patch supporting per-Conv
  `_active_width` (channel slicing at forward time, on both training and
  fused-inference paths).
- `Concat`-aware weight-column slicing for consumer convs that follow a
  multi-elastic-source Concat (`prepare_concat_alignment`).
- `C2PSA.forward` patch (`chunk(2, 1)` instead of `split((c, c), 1)`) so the
  frozen attention block at least accepts a sliced input without crashing.
- Ultralytics EMA-vs-model save workaround (`on_train_epoch_end` sync).

Each of these is required to run *any* elastic-width training on YOLO26 at
all; they weren't the reason things failed.
