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
