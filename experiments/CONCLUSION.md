# Can we make YOLO26 smaller at the same accuracy? — Definitive study

**Goal:** shrink Ultralytics YOLO26 (fewer params / FLOPs) while holding COCO mAP50-95,
via GETA structured pruning and, later, Once-for-All (OFA) elastic sub-networks.
**Answer (as of the pruning + KD work): no via structured pruning — yolo26 has little
redundant capacity to remove, and post-hoc KD of a right-sized student doesn't beat its
own baseline.** You are consistently better off picking the right-sized default model.

> **RESOLVED (2026-08-23) — width-elastic OFA rebuilt correctly, and it still
> does not produce a Pareto win.** The 2026-08-20 retraction stands on its facts:
> the original `0.0 mAP` *was* two implementation bugs (internal `chunk`/`cat`
> boundary violation under first-k slicing; shared BN running-stat corruption),
> not a capacity limit. So the whole thing was rebuilt — `ChannelPlan`
> group-structured slicing, per-width BN, importance sorting, elastic attention —
> with **201 correctness tests** including an exact elastic-vs-narrow oracle per
> module type. The rebuilt version works, and the answer is now *measured* rather
> than inferred from a broken implementation:
>
> * a verified-correct sub-net at w=0.875 scores **0.0395**, not 0.0;
> * importance sorting cuts first-layer error 2.4× and gives **no end-to-end gain**;
> * elastic attention removes the parameter floor (w=0.5: 5.05 M → **2.87 M**, vs
>   yolo26n's 2.6 M) and is strictly better at matched params (3.6× at ~8.5 M);
> * one rung of sandwich training recovers **+0.044** at the trained width, with
>   transfer decaying to zero within ~4 width points — while the widths that save
>   real parameters need **0.30–0.43**.
>
> The measured obstacle is structural, not a tuning failure: **the widths that
> survive save almost no parameters (w ≥ 0.96 → ≤ 7 %), and the widths that save
> real parameters are dead (w ≤ 0.875).** Full detail: `ofa/OFA_SUPERNET_PLAN.md`.
>
> The depth-elastic `0.0` (Paradigm 2) remains **untested-suspect** — same class of
> bug was found in its sibling, and it was evaluated with full-depth BN stats.
> **Paradigm 1 (pruning) still stands** — it had a real 50-epoch full-COCO fine-tune.

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

## Paradigm 2 — depth-elastic OFA (**RETRACTED — the 0.0 was a BN artefact**)
> **Withdrawn 2026-08-23.** The `0.0 mAP` below was reproduced exactly
> (yolo26l, 14 C3k blocks, `0.0000`) and then corrected by a single controlled
> change — recalibrating BN for the depth actually being evaluated:
>
> | | d=2 | d=1 |
> |---|---|---|
> | no recal (original protocol) | 0.5375 | **0.0000** |
> | with per-depth recal | 0.5303 | **0.1473** |
>
> The d=2 control lands on the baseline (0.5375 vs 0.5375), so the harness is
> neutral; the only variable is BN. The claim below that "the elastic second
> bottleneck is *essential* computation, not residual refinement" and that "the
> ceiling is fundamental, not a bug" is therefore **unsupported** — as is the
> ~100 % feature-MSE argument, since that MSE reproduces (0.643/1.005/1.090)
> while the network nonetheless scores 0.147. High feature MSE never
> distinguished a capacity ceiling from an un-normalised distribution shift.
> The 10-ep sandwich and 5-ep feature-KD results are void for the same reason,
> plus the BN-corruption bug that affected all elastic training then.
> Whether depth-elastic is *useful* is open but unpromising: 0.147 from 0.530
> buys ~−13 % FLOPs. Detail: `ofa/OFA_SUPERNET_PLAN.md`, `ofa/depth_retest.py`.

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

## Paradigm 3 — width-elastic OFA (**RETRACTED — was two bugs, question reopened**)
> The narrative below is kept for the record, but its conclusion is **withdrawn**.
> The `0.0 mAP` was caused by (A) naive first-k slicing violating the internal
> `chunk(2,1)`/`cat` boundaries inside every C3k2/C3k/SPPF block — 23 sites, proven
> numerically — and (B) `bn.running_mean[:k]` being a *view*, so every train-mode
> narrow pass (including the `no_grad` teacher) overwrote the inner region of all
> ~60 elastic BN buffers. Tell-tale: kd=10 and kd=1000 collapsed identically, which
> a gradient-magnitude explanation cannot account for. Corrected plan + evidence:
> `ofa/OFA_SUPERNET_PLAN.md`.

Make yolo26s width-elastic (each Conv slices its output channels by a global
`_active_width` ∈ (0, 1]; `width_elastic.py`) so w=1.0 recovers the pretrained
0.472 baseline and w<1.0 is a shrunk sub-net. Target: an n↔s width supernet
(n and s share depth=0.50, differ only in width). All the correctness plumbing
was implemented and PoC-verified:

- **PoC (5/5 checks pass, `width_poc.py`):** bit-identical forward at w=1.0
  (max_abs_diff = 0.0), non-crash finite forward at w=0.5, mAP=0.4717 at w=1.0,
  gradient flow only in the `[:out_k, :in_k]` weight slice (weight-sharing verified).
- **Untrained subnet at w=0.5: 0.0000 mAP**, exactly matching the depth-elastic
  outcome — the sliced first-k channels aren't in any semantically meaningful
  order in the pretrained net.
- **Progressive-shrinking training tried three ways.** All failed:
  1. Sandwich [1.0, 0.75], grad-tracked teacher + student + KD (λ=10): 3 ep at
     fraction=0.1 → student stayed at 0.0; teacher's gradient fought student's
     on the shared `weight[:0.75, :0.75]` region.
  2. Same widths, **`no_grad` teacher** (OFA-MIT in-place-KD pattern): 3 ep at
     fraction=0.1 → w=1.0 collapsed **0.472 → 0.0001** (student grad shifted
     shared inner weights, breaking the outer teacher context); w=0.75 still 0.0.
  3. Narrow-gap diagnostic [1.0, 0.95] with kd=1000: same collapse (0.472 → 0.005)
     in 74 batches. Even 5% width step + KD-dominant loss couldn't hold the teacher.
- **BN recalibration alone (`bn_recal.py`):** untrained sub-net at w=0.5 stayed
  at 0.0; recalibration at w=1.0 on val-distribution stats actually *dropped*
  it to 0.345 (val ≠ train distribution).

Also fixed three real Ultralytics/YOLO26 interactions on the way (all shipped):
- **Concat channel alignment** at two-elastic-source neck concats (yolo26s
  model.16, model.19): naive `weight[:, :in_k]` mixed source-A columns into
  source-B territory; wrote `prepare_concat_alignment` for per-source column
  slicing.
- **Ultralytics `save_model` writes `ema.ema` unconditionally.** With EMA
  disabled, `ema.ema` stays at initial state → `last.pt` was byte-identical
  to `yolo26s.pt` after 3 epochs of "training." Fix: `on_train_epoch_end`
  callback syncing `ema.ema` from the live model.
- **C2PSA's `split((self.c, self.c), 1)` is width-hostile** — `self.c` is baked
  at init. Replaced with `chunk(2, 1)`.

**Root cause is the same as depth-elastic:** the pretrained net's channel
ordering is arbitrary, so first-k slicing keeps an arbitrary subset. To make
first-k meaningful requires importance-based channel sorting AND long
progressive-shrinking training (100+ ep on full COCO with per-width BN
recal) — the full OFA-MIT recipe. We wrote scaffolding for the sorting
step (`channel_sort.py`) but stopped before completing the module-by-module
graph walking needed to keep bit-identity through all of yolo26's attention
blocks, Concat consumers, Bottleneck residuals, and Detect branches. The
CONCLUSION doesn't change: yolo26 has no channels the pretrained network
doesn't use, so even a correct implementation is unlikely to produce a
sub-net that beats the free default (n at 0.395).

## Paradigm 4 — knowledge distillation x→m (method improved; fine-tune recipe capped)
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

## Unified conclusion (revised 2026-08-20)
**What holds:** structured pruning deletes capacity YOLO26 actually uses and a
50-epoch full-COCO fine-tune recovers only ~80 % of dense-x (0.450 at default-m's
param count vs m's 0.518). Post-hoc KD-fine-tuning of a right-sized student
doesn't lift it above its own baseline either. For **these** interventions, the
right-sized default wins.

**What does NOT hold:** the stronger claim that YOLO26 is Pareto-efficient against
*any* cheap intervention. Both elastic-OFA "dead ends" were measured with buggy
slicing (and, for width, corrupted BN statistics), so they are **not evidence**
about capacity. Width-elastic is reopened (`ofa/OFA_SUPERNET_PLAN.md`); depth-elastic
deserves one re-test.

The one method improvement that survived controlled testing: **CWD channel-wise KL
beats the stock score-weighted L2 in Ultralytics' distillation loss** (upstream
candidate) — though it loses on the official O365→COCO pipeline, so the gain is
regime-specific.

### What *does* give "faster at the same mAP"
**TensorRT FP16 on the dense model** — lossless, −37 to −41 % GPU latency, no mAP change
(see `geta_yolo26/FINDINGS.md` §8). No pruning/OFA needed. INT8 QDQ is slower AND lossy for
YOLO26 (attention / concat / NMS-free head don't INT8-fuse).

## Reproduce
- Pruning: `geta_yolo26/prune_finetune.py`, `geta_trainer.py`, `profile_pruned_x.py`
  (+ GETA graph fixes in `only_train_once/`). Result: `out/geta_x_s50_full/pruned_val.json`,
  `out/yolo26x_pruned_profile.json`.
- Depth-elastic OFA: `ofa/elastic_yolo26.py`, `sandwich_train.py`, `sandwich_kd_train.py`,
  `ofa_eval.py`, `inspect_depth.py`, `calib_kd.py`.
- Width-elastic OFA: `ofa/width_elastic.py` (Conv/Concat/C2PSA patches + Concat alignment),
  `width_poc.py` (5-check PoC), `width_sandwich_train.py` (in-place KD sandwich),
  `width_eval.py`, `bn_recal.py`, `channel_sort.py` (partial). Full write-up in
  `ofa/OFA_WIDTH_RESULTS.md`.
- KD improvements: `distill/improved_distill.py`, `improved_distill_train.py`.
