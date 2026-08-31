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
> *(Scope, added 2026-08-29: that holds for **post-hoc** elasticity only. Trained
> from scratch as a supernet, the deepest cut here — fraction 0.5 — scores
> **0.3439**, not 0. See Paradigm 7.)*
>
> The depth-elastic `0.0` (Paradigm 2) remains **untested-suspect** — same class of
> bug was found in its sibling, and it was evaluated with full-depth BN stats.
> **Paradigm 1 (pruning) still stands** — it had a real 50-epoch full-COCO fine-tune.

> **UPDATE (2026-08-29) — the "COCO has no slack" reading is RETRACTED.** True
> from-scratch OFA on COCO (**Paradigm 7**) yields sub-nets within **2–4 % relative**
> of individually-trained twins at every width, including the ones that "save real
> parameters." Every earlier COCO elastic result — including the ones quoted in the
> blockquote above — was measured **post-hoc on an already-converged checkpoint**.
> That protocol, not yolo26's capacity, is what produced the dead widths. The Pareto
> verdict is unchanged: OFA still never beats a dedicated model at the same width.

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
> **Training then measured too** (same recipe as the width probe, 6 ep @ 15 %
> COCO): d=1 recovers to **0.4209 on l** (+0.274) and **0.3823 on s** (+0.236) —
> ~80 % of the gap, and **5–6× the width axis's +0.044 per rung**. So the depth
> axis *is* trainable; the original's "KD cannot fix this" was also wrong.
>
> It is nonetheless **Pareto-dominated, for a mundane reason**: halving the C3k
> inner bottlenecks saves only **5.2 % (s) / 12.7 % (l)** of MACs. Trained
> l d=1 (0.4209 @ 40.96) loses to plain **yolo26s** (0.472 @ 11.42) — better
> accuracy at a quarter of the compute; trained s d=1 (0.3823 @ 10.82) loses to
> **yolo26n** (0.395 @ 3.05).
>
> So the practical conclusion survives but the explanation is entirely
> different: not "the removed computation is essential", but **"the elastic
> dimension is too small a share of compute to pay for what it costs"**.
>
> **That diagnosis was then acted on and confirmed.** Applying the same axis one
> level up — dropping *whole C3k blocks* from each C3k2's `.m` — saves **−30.4 %
> MACs** on yolo26l (46.90 → 32.63 G, 2.4× the inner-depth saving) and recovers
> **+0.2857** in one short run (0.1058 → **0.3915**, 71.8 % of the gap) — the
> largest absolute recovery anywhere in this study. The method works best
> exactly where the diagnosis said it would.
>
> It is nonetheless still short: **yolo26s (0.472 @ 11.42 G) dominates it** —
> better accuracy at 2.9× less compute — and the s→m bar at 32.63 G is ~0.509.
> Decisively, the *ceiling* is small: a sub-net matching its own teacher exactly
> (impossible, it executes less compute) would score 0.5303 @ 32.63 G, beating
> yolo26m by **+0.012 at −13 % MACs** — a win inside the noise of a training
> recipe. So the remaining experiment is well-defined (a full 80-epoch run to
> close the residual 0.119) but the prize is bounded and tiny.
> Detail: `ofa/OFA_SUPERNET_PLAN.md`, `ofa/depth_retest.py`, `ofa/rung_train.py`,
> `ofa/block_depth.py`.

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

## Paradigm 5 — intermediate model size, trained (the non-compression route)
Every paradigm above *removes* capacity from a trained checkpoint. This tests the
route that built the family in the first place: train a size that doesn't exist.
**width 0.375** (`ofa/yolo26-w375.yaml`) sits at **5.655 M**, in the empty gap
between n (2.572 M) and s (10.010 M).

Init: the public `yolo26s-objv1-150.pt` (Objects365, 150 ep) sliced to 0.375 via
`ChannelPlan` with importance-sorted channels — 78 convs transferred, 0
mismatched, since `make_divisible(c·0.375,8)` is exactly 0.75× its 0.50
counterpart. Recipe: the **official** yolo26s COCO stage read verbatim from
`yolo26s.pt`'s own `train_args` (70 ep, batch 128, MuSGD). 7.0 h on one PRO 6000.

| | params | mAP50-95 |
|---|---|---|
| yolo26n | 2.572 M | **0.395** |
| **w375** | 5.655 M | **0.3863** |
| *n→s bar* | 5.655 M | *0.427* |
| yolo26s | 10.010 M | 0.472 |

**Pareto-dominated by yolo26n** — lower mAP at 2.2× the parameters, and 0.041
short of the interpolation bar. Still improving at +0.001/ep at epoch 70, and
the expected `close_mosaic` bump at epoch 61 **never appeared**, so no generous
extrapolation reaches 0.427.

**What this does and does not show.** It establishes that *slicing s down to
0.375 and fine-tuning 70 epochs* fails. It does **not** establish that
intermediate widths are bad architectures. Landing *below n* is the tell: with
2.2× n's capacity and a genuine 150-epoch Objects365 pretrain, a 0.375 model
should beat n comfortably. Scoring under it implicates the **initialisation**,
matching this study's own finding that sliced-init sub-nets start near zero and
recover only partially. The clean test — from-scratch arms at matched budget,
~60–90 h GPU — was not run. Data: `ofa/out/w375_coco.json`.

## Paradigm 6 — VOC control: does our OFA pipeline work AT ALL?
Every result above is on COCO and negative, which is consistent with two very
different causes: **(a)** the pipeline is broken, or **(b)** yolo26-on-COCO has
no redundant capacity. 187 correctness tests prove the *slicing* is exact; they
say nothing about whether sandwich training can yield a usable sub-net. That
gap stayed open for the whole study.

VOC discriminates: 16.5k images / 20 classes leaves yolo26s heavily
over-parameterised, so redundancy certainly exists. It is also cheap enough to
run **TRUE OFA** — a supernet trained *as* a supernet from random init,
sandwich-sampled throughout — which every COCO experiment failed to do (all of
them applied elasticity *post-hoc* to an already-converged checkpoint).

**OFA tax** = mAP(width trained alone) − mAP(supernet sub-net at that width):

| width | params | alone | supernet | tax |
|---|---|---|---|---|
| 0.50 | 9.96 M | 0.5991 | 0.5707 | **+0.0285** |
| 0.375 | 5.8 M | 0.5792 | 0.5579 | **+0.0213** |
| 0.25 | 2.8 M | 0.5329 | **0.5337** | **−0.0008** |

**The pipeline works.** At the smallest width the sub-net matches its
individually-trained twin outright. The tax *grows with width* — the supernet
gives up its largest configuration to serve the smaller ones, the classic OFA
trade. One training run yields three deployable models, each within 0.029 of a
dedicated one. (Sub-nets are slightly *larger* than their baselines at low
width because the Detect head does not shrink. **Corrected 2026-08-29:** that makes
these taxes *optimistic*, not conservative — a size-matched sub-net would score
lower, so the true tax is somewhat higher. The COCO table in Paradigm 7 carries an
explicit size-matched column.)

**What this does and does not license.** It rules out "the code never worked" —
the machinery demonstrably delivers where capacity is slack. But it does **not**
cleanly prove the COCO failures were purely about redundancy, because two things
differ at once: VOC had (i) an over-parameterised task **and** (ii) true
from-scratch supernet training, while COCO had neither. **True from-scratch OFA
on COCO was never run** (2–3× a full run), so the COCO negatives were best read
as *some combination* of genuine lack of slack and the post-hoc protocol.

> **CLOSED (2026-08-29).** That run has now been done — see **Paradigm 7**. It
> holds the task fixed (COCO) and varies only the protocol, and it comes out
> decisively on the protocol side: the tax on COCO is **smaller** than on VOC.
> The "genuine lack of slack" half of the disjunction is retracted.

Two runs were needed: the first was invalid (absolute widths passed where an
elastic *fraction* was expected, making every sub-net ~3.5× smaller than its
baseline and leaving the full supernet untrained). Archived as
`ofa/out/voc_ofa_results_INVALID.json`. Data: `ofa/out/voc_ofa_results.json`.

## Paradigm 7 — true from-scratch OFA on COCO (**resolves the study's central ambiguity**)
Paradigm 6 left one confound standing: VOC changed *two* things at once — an
over-parameterised task **and** true from-scratch supernet training. This run
removes the first. Same COCO, same architecture family, same budget, both arms
from **random init**; the only difference between the arms is whether the network
was trained *as* a supernet.

**Protocol.** Supernet = width 0.5, sandwich-sampled from random init at fractions
1.0 / 0.75 / 0.5 (absolute widths 0.50 / 0.375 / 0.25). Baselines = each of those
three widths trained alone under an identical recipe, epochs (100), imgsz and data.
Every sub-net BN-recalibrated for its own configuration before eval (**96/96**
layers at all three widths) — the omission that produced two retracted conclusions
earlier in this study.

| width | alone (params → mAP) | OFA sub-net (params → mAP) | tax | rel | size-matched tax |
|---|---|---|---|---|---|
| 0.50 | 10.010 M → 0.4247 | 10.010 M → 0.4069 | **+0.0178** | 4.2 % | +0.0178 (4.2 %) |
| 0.375 | 5.655 M → 0.3972 | 5.880 M → 0.3865 | **+0.0107** | 2.7 % | +0.0121 (3.0 %) |
| 0.25 | 2.572 M → 0.3470 | 2.874 M → 0.3439 | **+0.0031** | 0.9 % | +0.0080 (2.3 %) |

The sub-nets at 0.375 and 0.25 are *larger* than their baselines (the Detect head
does not shrink), so the raw tax is **optimistic**. The size-matched column charges
that excess back at the local slope of the from-scratch baseline curve
(0.0063 mAP/M above 5.66 M, 0.0163 mAP/M above 2.57 M). Either way the tax is
**2–4 % relative across the whole ladder**.

**The headline: on COCO, true from-scratch OFA works.** Compare like for like, same
dataset and same architecture — post-hoc elasticity on a converged checkpoint scored
**0.0000**; a supernet trained as a supernet from scratch scores **0.3439** at the
same fraction. Nothing about yolo26's capacity changed between those two numbers.
**The protocol was the problem.** Note also that the tax is *smaller* on COCO than on
VOC (+0.0178 vs +0.0285 at the top width) — the opposite of what "COCO has no slack"
predicts.

**But it is not a Pareto win.** The OFA sub-net never beats its own baseline at any
width, and the small sub-net is nowhere near the large one (2.874 M → 0.3439 vs
10.010 M → 0.4069). OFA's value here is **logistical, not architectural**: one
training run instead of three, giving three deployable operating points at ~2–4 %
relative mAP each. It does not produce a model that is both smaller *and* better.

**Scope — read this before quoting the numbers.** This is a 100-epoch from-scratch
protocol. It is internally matched (both arms identical) but it is **not** comparable
to the official checkpoints in the baselines table above: w=0.5 trained alone reaches
0.4247 here versus **0.472** for the released yolo26s, which gets Objects365
pretraining and a much longer schedule. What this run licenses is the *retraction* —
the 0.0000s were protocol artefacts, not capacity limits. It does **not** license the
claim that the tax stays at 2–4 % under full official convergence, where the
baselines are stronger and harder to match. That test has not been run.

**Recipe (identical on both arms, recovered from the trainer args):** model built
from a `.yaml` so **random init**, epochs 100, batch 64, imgsz 640, SGD lr0 0.01,
close_mosaic 10, COCO train2017/val2017. Note this is the stock recipe, *not* the
official MuSGD schedule the released checkpoints use — another reason the absolute
mAPs here sit below the released ones while the *comparison between arms* stays valid.

Data: `ofa/out/coco_ofa_results.json` (merged, with the size-matched correction);
raw arms in `ofa/out/coco_base_raw.json` and `ofa/out/coco_sup_raw.json`; baseline
log in `ofa/out/coco_base.log`; supernet trainer args in `ofa/out/coco_sup_args.txt`.

## Paradigm 8 — the OFA tax at OFFICIAL convergence (first attempt RETRACTED; rerun in flight)
Paradigm 7 showed from-scratch OFA on COCO works, but its baselines topped out at
0.4247 against the released yolo26s's 0.4714. It therefore could not answer the
question that matters for deployment: does the tax survive when the baseline is
genuinely strong?

**The design writes itself.** The released `yolo26s.pt`'s own `train_args` record
`model = yolo26s-objv1-150.pt` — the shipped model *is* the public Objects365
checkpoint plus a 70-epoch COCO stage. So reproduce that pipeline and change
exactly one thing: leave elasticity on for the whole COCO stage. The supernet's
max arm then doubles as a reproduction check of the official pipeline.

Released checkpoints, re-measured under our own val call so no protocol
difference leaks into the tax (published values in brackets):

| model | params | measured | published |
|---|---|---|---|
| yolo26s | 10.010 M | **0.4714** | 0.472 |
| yolo26n | 2.572 M | **0.3949** | 0.395 |

### First attempt — RETRACTED (it trained from random init)
19.6 hours of training produced a max arm of **0.2375** against 0.4714. The cause
was neither elasticity nor the recipe. `apply_init()` loaded the checkpoint into
`y.model`, but ultralytics runs

```python
self.trainer.model = self.trainer.get_model(
    weights=self.model if self.ckpt else None, cfg=self.model.yaml)
```

and `YOLO(<yaml>)` leaves `self.ckpt` empty, so **the trainer rebuilt the model
from the yaml and discarded the loaded weights.** The run was a from-scratch
model trained 70 epochs at `lr0=0.00038` — a *fine-tuning* learning rate — which
explains 0.2375 exactly.

It surfaced only because two probes, one from the Objects365 init and one from
random init, returned an **identical epoch-1 mAP of 0.000332**. Six matching
significant figures is not a coincidence. Data:
`ofa/out/official_ofa_results_INVALID.json`.

Two claims die with it: that the 9 recipe keys public ultralytics cannot express
explained the shortfall, and that the missing `cls_w` multiplier was the
mechanism — the latter independently refuted by a probe that raised `cls` and
watched mAP collapse to 0.019.

**What it teaches.** The guard reported "696/708 tensors transferred" and was
worthless: it verified `y.model`, the object being thrown away. Every retraction
in this study has that same shape — **verify the object that is actually USED,
not the one you just touched.** The replacement asserts at *train start* that
`trainer.model` is bit-identical to the checkpoint on disk. A cheap detector for
the whole class: run one config from two different inits; identical metrics mean
one init never reached training.

### Corrected runs — PARTIAL, NOT A RESULT YET
Init now verified in the trainer (`696/708 src tensors identical`). Epoch 1 scores
**0.278**, already better after one epoch than the broken run reached in seventy.

Matched-epoch elastic vs non-elastic (identical init, recipe, batch and epochs):

| epoch | non-elastic | supernet | gap |
|---|---|---|---|
| 1 | 0.278 | 0.159 | +0.1190 |
| 2 | 0.368 | 0.341 | +0.0270 |
| 3 | 0.388 | 0.368 | +0.0200 |

**These are epochs 3–6 of 70 and must not be quoted as the tax.** The direction is
the expected one — the supernet starts behind, because it splits capacity three
ways, and closes quickly — but a 3-epoch gap is not a converged gap. The
non-elastic arm is at 0.414 by epoch 6 against the released 0.4714, so the
filtered recipe does look able to approach the released model, which would mean
those 9 unsupported keys are immaterial after all.

Three arms are training: `fixed_nonelastic` (batch 96) and `fixed_supernet`
(batch 96) form the matched pair, and `fidelity_b128` runs non-elastic at the
official batch 128. The headline will be
`fixed_nonelastic − fixed_supernet@1.0`, which shares init, recipe, batch and
epochs and so isolates the cost of elasticity from both the recipe gap and the
batch deviation. Deviation on record: the RTX 6000D has 85.6 GB and the sandwich
needs 85.5 GB at batch 128, so the matched pair runs at batch 96.

Pre-registered decision rule, written before any of these results existed:
`ofa/out/official_ofa_PREREGISTERED.md`.

## Unified conclusion (revised 2026-08-31)
**What holds:** structured pruning deletes capacity YOLO26 actually uses and a
50-epoch full-COCO fine-tune recovers only ~80 % of dense-x (0.450 at default-m's
param count vs m's 0.518). Post-hoc KD-fine-tuning of a right-sized student
doesn't lift it above its own baseline either. For **these** interventions, the
right-sized default wins.

**RETRACTED (2026-08-29) — "yolo26-on-COCO has no redundant capacity for OFA."**
This study leaned on that reading to explain a string of 0.0000s on COCO. It is
wrong. Paradigm 7 ran the experiment that was always missing — a supernet trained
*as* a supernet from random init on COCO — and its sub-nets land within **2–4 %
relative** of individually-trained twins at every width. The 0.0000s came from
applying elasticity **post-hoc to an already-converged checkpoint**, which is not
what OFA prescribes. Three mechanisms were blamed before the protocol was: buggy
slicing, corrupted BN statistics, and "no slack." The first two were real and were
fixed; the third never was.

**What still does NOT hold:** that OFA buys a *Pareto win*. It doesn't. Trained
correctly, the supernet's sub-nets are at best equal to — never better than — the
same width trained alone (Paradigm 7: −0.0031 to −0.0178 mAP), and the small sub-nets
stay far below the large ones. OFA's payoff is **one training run covering three
deployment points at ~2–4 % relative mAP each** — a real engineering benefit, not a
smaller-and-better model. Nothing in this study produces a yolo26 variant that is
simultaneously smaller than a default *and* better than it.

**Limits of the retraction.** Paradigm 7 matches its two arms to each other, not to
the released models: 100 epochs, random init, no Objects365. It establishes that the
elastic failures were procedural. It does **not** establish that the 2–4 % tax
survives at official convergence — that is Paradigm 8, which is **still running**.
Its first attempt was retracted for training from random init, and the corrected
runs are only a few epochs in. **No tax at official convergence has been measured
yet**, and the early matched-epoch gap (+0.020 at epoch 3, narrowing) is a
direction, not a number.

**The methodological result may outlast the empirical one.** Four separate
conclusions in this study were measurement artefacts, not findings: two BN
recalibration failures, one post-hoc-elasticity protocol error, and one discarded
initialisation. Every one produced a *plausible* number that survived review
because the guard checked the wrong object. The practices that actually caught
them — per-configuration BN recalibration, training elastic from scratch rather
than post-hoc, verifying weights inside the trainer, pre-registering the decision
rule, and re-running one config from two inits to see if the metrics move — are
the transferable output of this work.

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
- True from-scratch OFA (VOC + COCO): `ofa/voc_ofa_validate.py` — `baselines`,
  `supernet` and `report` sub-commands; `ofa/prepare_voc.py` for the VOC control.
  Results: `ofa/out/voc_ofa_results.json`, `ofa/out/coco_ofa_results.json`
  (+ `coco_base_raw.json`, `coco_sup_raw.json`, `coco_base.log`, `coco_sup_args.txt`).
  Note `voc_ofa_validate.py --widths` takes **absolute** yolo26 width multipliers
  and converts them to supernet fractions internally; `set_width()` takes the
  fraction. Conflating the two invalidated an entire run.
