# YOLO26n one-shot magnitude pruning — NO retraining (COCO val2017 mAP50-95)

Baseline (sparsity 0): mAP50-95 = 0.3949  (sanity check passed exactly -> pipeline sound)

## Per-layer (local) magnitude pruning — gentlest here
| group sparsity | mAP50-95 | Δ vs baseline | params (M) |
|---|---|---|---|
| 0%  | 0.3949 | —        | 2.409 |
| 1%  | 0.3923 | -0.7%    | 2.404 |
| 2%  | 0.3705 | -6.2%    | 2.384 |
| 3%  | 0.3590 | -9.1%    | 2.376 |
| 4%  | 0.2901 | -26.5%   | 2.332 |
| 5%  | 0.2620 | -33.7%   | 2.311 |

## Global (per-element threshold) magnitude pruning — worse (concentrates removal)
5% -> 0.024 ; 10% -> ~0 ; >=15% -> 0.0

## Conclusion
Without retraining, yolo26n tolerates only ~1% structured pruning losslessly (and that
removes ~5K params, negligible). It has very little structural redundancy; one-shot
structured pruning degrades mAP steeply (2% already -6%). Meaningful compression REQUIRES
fine-tuning. Accuracy measured on the magnitude-zeroed model (== constructed model;
sparsity-0 == baseline confirms validity).

## Full family — local magnitude pruning, NO retraining (mAP50-95)
Each model's own sparsity-0 row == its baseline (sanity check, construct_diff=0.0).

| sparsity | yolo26n (base .395) | yolo26s (.472) | yolo26m (.518) | yolo26x (.563) |
|---|---|---|---|---|
| 0%  | 0.395 | 0.472 | 0.518 | 0.563 |
| 1%  | 0.392 (-0.7%) | 0.463 (-1.9%) | 0.506 (-2.3%) | 0.555 (-1.4%) |
| 2%  | 0.371 (-6%)   | 0.407 (-14%)  | 0.477 (-8%)   | 0.538 (-4.4%) |
| 5%  | 0.262         | 0.141         | 0.242         | 0.333 |
| 10% | ~0            | ~0            | ~0            | ~0    |

Conclusion across the WHOLE family: one-shot structured pruning WITHOUT retraining
holds accuracy only to ~1% sparsity for every size. Bigger models retain marginally
more at 2-5% (x is best: -4.4% at 2%) but all collapse by ~10%. No yolo26 variant
tolerates meaningful (>~2%) structured pruning without fine-tuning.

## BN recalibration (NO gradients — 800 train imgs forward in train mode, recompute BN stats)
yolo26x (baseline 0.5626):
| sparsity | mAP no-recal | mAP + BN-recal | params (M) |
|---|---|---|---|
| 2%  | 0.538 | 0.541 (-3.8% vs base) | 54.6 |
| 5%  | 0.333 | 0.474 (-16%)          | 52.0 |
| 10% | 0.0005 -> 0.345 (-39%)| | 48.1 |
| 20% | 0.0 -> 0.033          | | 40.8 |

KEY FINDING: most of the one-shot pruning collapse was STALE BatchNorm running stats,
not lost capacity. BN recalibration (no backprop, ~seconds) recovers huge amounts:
5% 0.333->0.474, 10% ~0->0.345. So "pruning without retraining" is far more viable than
the raw one-shot numbers suggested -- with a free BN-recal pass, yolo26x prunes ~5-10%
at usable accuracy. Beyond ~10-20% needs real fine-tuning.

## BN-recal across the FULL family (mAP50-95 after recal; no gradients)
| sparsity | n (base .395) | s (.472) | m (.518) | x (.563) |
|---|---|---|---|---|
| 2%  | 0.372 (94%) | 0.424 (90%) | 0.480 (93%) | 0.541 (96%) |
| 5%  | 0.293 (74%) | 0.348 (74%) | 0.402 (78%) | 0.474 (84%) |
| 10% | 0.121       | 0.148       | 0.225       | 0.345 (61%) |
(% = fraction of baseline retained)

Recovery before->after BN-recal at 5%: n .262->.293, s .141->.348, m .242->.402, x .333->.474.
At 10% all were ~0 before; after: n .121, s .148, m .225, x .345.

TAKEAWAY: BN recalibration (free, no backprop) makes one-shot pruning genuinely useful.
With it, every yolo26 size keeps ~90-96% mAP at 2% sparsity, and the bigger the model the
more it prunes for free (x: 84% retained at 5%, still 0.345 at 10%). The earlier "collapse"
was mostly stale BN stats. Beyond ~10% (or for nano) real fine-tuning is still needed.

## Short fine-tune at 5% sparsity (yolo26n) — recovery ladder
| method (5% sparsity) | mAP50-95 | % of baseline (0.395) |
|---|---|---|
| one-shot, no recovery        | 0.262 | 66% |
| one-shot + BN-recal (free)   | 0.293 | 74% |
| 5-epoch FT @ lr0=0.01 (wrong)| 0.259 | 66% |
| 5-epoch FT @ lr0=1e-3        | 0.359 | 91% |
LR was the bottleneck: Ultralytics 'auto' forces lr0=0.01 (from-scratch LR); for a
fine-tune use ~1e-3. 5 epochs @ 1e-3 recovers to within ~9% of baseline (params -8%).
