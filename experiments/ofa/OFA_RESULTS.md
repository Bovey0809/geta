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
