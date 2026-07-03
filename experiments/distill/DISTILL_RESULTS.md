# Knowledge distillation x→m on YOLO26 — improving Ultralytics' built-in KD

Ultralytics (≥8.4.x) ships feature-based KD: `model.train(distill_model="teacher.pt")`
wraps student+teacher in `DistillationModel` — score-weighted L2 on the 3 neck feature
maps (a learned 1×1 projector aligns student→teacher channels; teacher head scores act
as spatial weights; weight `dis=6.0`). Teacher = yolo26x (0.5626), student = yolo26m
(20.4 M, default 0.518). Goal: distilled-m > default-L (0.5375) at m's size.

## Improvements implemented (`improved_distill.py`, drop-in subclass)
- **logit** — response KD on the Detect head class-scores (one2many+one2one), soft-BCE
  vs teacher probabilities. The stock method never distills the teacher's decisions.
- **cwd** — channel-wise distillation (Shu et al. 2021): per-channel spatial softmax +
  KL, replacing the raw score-weighted L2.
- **fgd** — Focal-and-Global-style global term: spatial/channel attention MSE + a
  parameter-free global-context relation.
All toggleable; all off = exact stock method, so the same script A/Bs cleanly
(`improved_distill_train.py` monkeypatches the trainer to build the subclass).

## Ablation (m←x, from pretrained-m, 8 ep @ 50 % COCO, batch 48, lr0 1e-3, warmup 1 — identical recipe, only KD terms vary)
| config | mAP50-95 | Δ vs stock |
|---|---|---|
| **cwd** | **0.5036** | **+0.0022** |
| fgd | 0.5022 | +0.0008 |
| logit | 0.5015 | +0.0001 |
| stock (built-in) | 0.5014 | — |
| all three | 0.5011 | −0.0003 |

Findings: **CWD is the best single change** (channel-normalized KL > raw L2 for dense
prediction, as in the literature). Logit KD adds nothing measurable on this recipe; FGD
marginal. **Stacking all three hurts** — naive term-summing interferes; replace, don't pile on.

## Full run on the winner (CWD-only, 40 ep full COCO, lr0 1e-3, batch 48)
| | mAP50-95 |
|---|---|
| CWD-distilled-m | **0.5031** |
| default-m | 0.518 |
| default-L (target) | 0.5375 |

Trajectory 0.450→0.473→0.488→0.500→0.503: the fine-tune first *disrupts* the converged
pretrained-m (mosaic + lr kick), then spends the entire run recovering — and does not
regain the starting 0.518. Telling: the 8-epoch screening (0.5036) ≈ the 40-epoch run
(0.5031). **Longer KD-fine-tuning does not help a converged student.**

## Conclusions
1. **CWD > stock score-weighted L2** — real, controlled improvement to the Ultralytics
   KD loss; candidate upstream patch.
2. **KD-as-fine-tune cannot lift a well-trained student above its own baseline**, let
   alone a larger model. `distill_model` is best used during **full from-scratch
   training**, where the teacher shapes learning before convergence (untested here:
   ~500 ep ≈ 100 h+ on an RTX Pro 6000).
3. Consistent with the pruning/OFA results (see `../CONCLUSION.md`): the YOLO26 family
   defaults sit on a Pareto frontier that cheap post-hoc methods do not beat.

## Files
`distill_train.py` (stock wrapper) · `improved_distill.py` (subclass: logit/cwd/fgd) ·
`improved_distill_train.py` (A/B runner) · runs: `ab_{stock,logit,cwd,fgd,all}`,
`distill_cwd_full` on the rtx6000 box.
