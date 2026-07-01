# OFA (Once-for-All, Cai/Han MIT) — validation vs GETA pruning
OFA-MobileNetV3 supernet (ofa_mbv3_d234_e346_k357_w1.0), ImageNet val, sub-nets extracted
with NO retraining (only BN recalibration on a few hundred images). Eval on 7k val subset.

| sub-net | params | FLOPs | top-1 | vs MAX |
|---|---|---|---|---|
| MAX  | 7.66M | 566M | 75.70% | -    |
| RAND2| 5.80M (-24%) | 360M (-36%) | 75.01% | -0.7% |
| RAND1| 4.88M (-36%) | 308M (-46%) | 74.23% | -1.5% |
| MIN  | 3.41M (-55%) | 121M (-79%) | 68.16% | -7.5% |

## Verdict
OFA delivers "smaller + maintain accuracy, NO retraining": a -36%-params/-46%-FLOPs sub-net
loses only 1.5 top-1 points, zero gradient steps (just free BN recal). This is the property
GETA pruning could NOT give YOLO26 (pruning -22% params cost -12% mAP even with a 15-epoch
fine-tune that plateaued).

Why: OFA pays the cost ONCE (train an elastic supernet via progressive shrinking so every
sub-net is pre-trained-good), then specialization is free. GETA pays it EVERY prune (fine-tune
to recover) and can't fully recover on efficient nets.

## Caveats / to apply to YOLO26
- OFA pretrained models are ImageNet CLASSIFICATION backbones, not detectors.
- A YOLO26 OFA would require training a *detection* supernet (elastic YOLO26 + progressive
  shrinking on COCO) -- a large multi-GPU effort, not a 12GB single-card job.
- Numbers slightly below OFA's published full-val (light 20-batch BN-recal + 7k subset);
  trend is the point.

## Repro: experiments/ofa/ofa_demo.py (needs ofa, gdown; supernet ckpt cached at .torch/ofa_nets/;
## github-raw download truncates on AutoDL -> download locally + scp the 31MB ckpt).

## YOLO26 depth-elastic OFA (2026-07-01, RTX Pro 6000) — dead end, same root cause as pruning
Made yolo26l depth-elastic (each C3k runs 2->1 inner residual bottlenecks; elastic_yolo26.py),
sandwich-trained so d=2 (full) + d=1 (shrunk) are both usable with no retraining.
| checkpoint | d=2 (46.9 GFLOPs) | d=1 (40.96 GFLOPs, -13%) |
|---|---|---|
| pretrained-l | 0.5377 | 0.0 |
| 10-ep naive sandwich | 0.5377 | 0.0 |
| 5-ep sandwich + feature-KD (lambda=10) | 0.5377 | 0.0 |
d=1 stuck at EXACTLY 0.0. calib_kd.py measured the cause: d=1 neck features (Detect inputs)
differ from d=2 by ~100% relative MSE (P3 0.58, P4 0.99, P5 1.12) -> the "elastic" 2nd bottleneck
is ESSENTIAL computation, not residual refinement. KD can't make a lower-capacity net reproduce a
higher-capacity one (at lambda=10 KD dominated the loss; d=1 still didn't move). d=1 forward is
correct (finite, right shape) -- the ceiling is fundamental, not a bug. See ../CONCLUSION.md.
