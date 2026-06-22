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
