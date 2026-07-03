# Can we make YOLO26 smaller at the same accuracy? — Definitive study

**Goal:** shrink Ultralytics YOLO26 (fewer params / FLOPs) while holding COCO mAP50-95,
via GETA structured pruning and, later, Once-for-All (OFA) elastic sub-networks.
**Answer: no — YOLO26 has no redundant capacity to remove.** Two independent compression
paradigms hit the same wall. You are consistently better off picking the right-sized
default model. The framework value delivered is the (non-trivial) engineering to run these
methods on YOLO26 at all, plus the evidence below.

Hardware: single **RTX PRO 6000 Blackwell 96 GB** (final runs); earlier work on RTX 3080 Ti 12 GB.
COCO val2017, mAP50-95, imgsz 640.

## YOLO26 default baselines (this study)
| model | params | mAP50-95 |
|---|---|---|
| yolo26n | 2.6 M | 0.395 |
| yolo26s | 9.5 M | 0.472 |
| yolo26m | 20.4 M | 0.518 |
| yolo26l | 24.8 M | 0.5375 |
| yolo26x | 59.0 M | 0.5626 |

## Paradigm 1 — GETA structured pruning (dead end)
Headline attempt "pruned-x beats default-L": prune yolo26x to below L's params, fine-tune
50 epochs on full COCO (lr 1e-3, batch 32), then construct + eval the real subnet.

| model | params | mAP50-95 | verdict |
|---|---|---|---|
| default-s | 9.5 M | 0.472 | — |
| **pruned-x (sparsity 0.5, 50ep FT)** | **20.37 M** | **0.450** | Pareto-dominated |
| default-m | 20.4 M | 0.518 | **beats pruned-x at = params** |
| default-l | 24.8 M | 0.5375 | target — not reached |

- Pruned-x recovered only **80 %** of dense-x (0.450 / 0.5626), even with a proper long run.
- **Same size as default-m, −0.068 mAP**; default-s beats it with <half the params.
- Speed/size of the 20.37 M subnet (bs1, 640): ONNX −65 % (236→82 MB), CPU latency −26 %
  (588→437 ms), **GPU latency ~0 %** (9.82→9.94 ms — bs1 is kernel-launch-bound on Blackwell;
  fewer params don't reduce it). So even the speed win is CPU/size only, at a real mAP cost.

Earlier phases (see `geta_yolo26/FINDINGS.md`, `RESULTS_no_retrain.md`, `RESULTS_speed.md`):
one-shot pruning collapses by ~10 % sparsity; BN-recal recovers some for free; fine-tune
recovery plateaus ~85-91 %. All consistent with **little structural redundancy**.

## Paradigm 2 — depth-elastic OFA (dead end, same root cause)
Make yolo26l depth-elastic (each C3k runs 2→1 inner residual bottlenecks; `elastic_yolo26.py`),
sandwich-train so both the full (d=2) and shrunk (d=1) sub-nets are usable with no retraining.

| checkpoint | d=2 (46.9 GFLOPs) | d=1 (40.96 GFLOPs, −13 %) |
|---|---|---|
| pretrained-l | 0.5377 | **0.0** |
| 10-ep naive sandwich | 0.5377 | **0.0** |
| 5-ep sandwich + feature-KD (λ=10) | 0.5377 | **0.0** |

d=1 stays **exactly 0.0** across pretrained and two training methods. Root cause, measured
directly (`calib_kd.py`): the d=1 neck features (Detect inputs) differ from d=2 by **~100 %
relative MSE** (P3 0.58, P4 0.99, P5 1.12). The "elastic" second bottleneck is *essential*
computation, not residual refinement — dropping it replaces the head's input with something
it has never seen. KD cannot fix this: a lower-capacity net cannot reproduce a higher-capacity
one's function (at λ=10 the KD term dominated the loss and d=1 still didn't move). Even the
best case is a mediocre sub-net at only −13 % FLOPs. The d=1 forward itself is correct
(finite outputs, right shape) — the ceiling is fundamental, not a bug.

## Paradigm 3 — knowledge distillation x→m (method improved; fine-tune recipe capped)
Instead of removing capacity, add a teacher: Ultralytics' built-in KD
(`distill_model=`, score-weighted-L2 neck-feature loss) distilling x (0.5626) into a
full-size m (20.4 M, default 0.518). We implemented and ablated three improvements
(details: `distill/DISTILL_RESULTS.md`):

| config (8 ep @ 50 % screening) | mAP50-95 |
|---|---|
| **+CWD (channel-wise KL)** | **0.5036** |
| +FGD global | 0.5022 |
| +logit KD | 0.5015 |
| stock (built-in) | 0.5014 |
| all three | 0.5011 |

**CWD > stock** (controlled, same recipe) — a real improvement to the loss; stacking all
terms hurts. But the full 40-epoch CWD run reached only **0.5031 — below default-m's
0.518**: KD-as-fine-tune first disrupts a converged student, then can't even recover the
baseline (8-ep screening ≈ 40-ep run). The honest remaining test is KD during **full
from-scratch training** (~500 ep, ~100 h+) — untested here.

## Unified conclusion
Structured pruning (remove channels) and depth-elastic OFA (remove blocks) both delete
capacity that YOLO26 actually uses, and neither fine-tuning nor knowledge distillation
recovers it. Post-hoc KD-fine-tuning of a right-sized student doesn't lift it above its
own baseline either. **YOLO26 is Pareto-efficient across its n/s/m/l/x family — the
right-sized default beats every cheap post-hoc size/accuracy intervention we tried.**
The one method improvement that survived controlled testing: **CWD channel-wise KL beats
the stock score-weighted L2 in Ultralytics' distillation loss** (upstream candidate).

### What *does* give "faster at the same mAP"
**TensorRT FP16 on the dense model** — lossless, −37 to −41 % GPU latency, no mAP change
(see `geta_yolo26/FINDINGS.md` §8). No pruning/OFA needed. INT8 QDQ is slower AND lossy for
YOLO26 (attention / concat / NMS-free head don't INT8-fuse).

## Reproduce
- Pruning: `geta_yolo26/prune_finetune.py`, `geta_trainer.py`, `profile_pruned_x.py`
  (+ GETA graph fixes in `only_train_once/`). Result: `out/geta_x_s50_full/pruned_val.json`,
  `out/yolo26x_pruned_profile.json`.
- OFA: `ofa/elastic_yolo26.py`, `sandwich_train.py`, `sandwich_kd_train.py`, `ofa_eval.py`,
  `inspect_depth.py`, `calib_kd.py`.
