# GETA × Ultralytics YOLO26 — Pruning Study, Findings

Single RTX 3080 Ti (12 GB), AutoDL. COCO val2017 mAP50-95. GETA = this repo (OTO v3).

## 1. Making GETA work on YOLO26 (it didn't, out of the box)
YOLO26 is newer than anything GETA had seen. Three real bugs had to be fixed before it
could prune YOLO26 at all:
- **torch 2.8 compat**: `torch.onnx._optimize_graph` removed → shim to `torch.onnx.utils`;
  `torch.load(weights_only=False)` for full-module checkpoints.
- **Dead chunk/split handling** (core bug): `_replace_slice_with_chunk` was commented out and
  `post_process_chunk_node` never called, so every 2-way channel split (C2f/C3k2 + attention)
  kept `num_groups` doubled → crash. Re-enabled slice→chunk, added `onnx::Split` handling,
  and wired `post_process_chunk_node`. Group mismatches 33 → 0.
- **Concat-aux offset** (partial fix): a conv consuming a concat advanced its offset by the
  resolved group's `num_groups`, which over-counts when an input resolves to a merged/excluded
  group. Now uses the real per-input channel width. Residual index-misalignment remains on some
  random prunes; the importance-driven (HESSO) path is consistent.
- **C2PSA attention blocks** (`model.10/22…`) excluded from pruning (their multi-head internals +
  `self.c` split can't be rewired by construct).

Result: GETA traces, prunes, and reconstructs YOLO26 (gate: construct output diff < 1e-4).

## 2. Baselines (COCO val2017)
| model | params | mAP50-95 |
|---|---|---|
| yolo26n | 2.41M | 0.395 |
| yolo26s | 9.5M  | 0.472 |
| yolo26m | 20.4M | 0.518 |
| yolo26l | 24.8M | 0.5375 |
| yolo26x | 55.7M | 0.563 |

## 3. Speed / size — 50% structural prune, ONNX, bs1, 640 (architecture-faithful)
| model | params | ONNX size | GPU bs1 | CPU bs1 |
|---|---|---|---|---|
| n | -46% | -44% | ~0 | -16% |
| s | -46% | -45% | -8% | -36% |
| m | -57% | -57% | -20% | -55% |
| l | -58% | -58% | -22% | -56% |
| x | -58% | -58% | **-30%** | -57% |
Pruning reliably cuts params/size ~46-58% and CPU latency a lot. **GPU bs1 latency gains scale
with model size** (nano is launch-bound → flat; x is compute-bound → -30%).

## 4. Pruning WITHOUT retraining (one-shot magnitude)
Every size: lossless only to **~1% sparsity**; steep collapse by ~10% (sanity check: sparsity 0
== baseline exactly). YOLO26 has little structural redundancy.

### BN recalibration (free — forward ~800 train imgs, recompute BN stats, no gradients)
Most of the one-shot "collapse" was **stale BatchNorm statistics**, not lost capacity:
| sparsity | n | s | m | x |
|---|---|---|---|---|
| 2%  | 0.372 (94%) | 0.424 (90%) | 0.480 (93%) | 0.541 (96%) |
| 5%  | 0.293 (74%) | 0.348 (74%) | 0.402 (78%) | 0.474 (84%) |
| 10% | 0.121 | 0.148 | 0.225 | 0.345 (61%) |
(% of baseline retained.) Recovery scales with model size; x recovers most.

## 5. Pruning WITH a short fine-tune (yolo26n @ 5%)
| method | mAP | % baseline |
|---|---|---|
| one-shot | 0.262 | 66% |
| + BN-recal (free) | 0.293 | 74% |
| + 5-epoch FT @ **lr0=0.01** (wrong) | 0.259 | 66% |
| + 5-epoch FT @ **lr0=1e-3** | **0.359** | 91% |
**LR was the bottleneck**: Ultralytics `auto` forces the from-scratch lr0=0.01; a fine-tune wants
~1e-3. (The earlier 50-epoch / 50% run that landed 0.234 was the same wrong-LR recipe.)

## 6. "Pruned beats a larger default" (Pareto win) — attempted, not achieved here
Goal: a pruned model with fewer params AND higher mAP than an off-the-shelf model.
- **yolo26x can't be fine-tuned on 12 GB** (OOM even at batch 4; ~50 min/epoch). x is the variant
  with enough redundancy to make this easy, but it doesn't fit.
- **pruned-l @ 20%, 6-epoch probe** → 16.6M params / 0.440 mAP: fewer params than m (20.4M) but
  **lower mAP than m (0.518)** — not a win (even dominated by default-s 9.5M/0.472). Over-pruned
  (0.2 group sparsity = -37% params) and undertrained.

**Conclusion:** beating a well-trained default YOLO26 by pruning needs the pruned net to retain
~96%+ accuracy. That requires either the big-model redundancy (x) — which needs a ≥24 GB GPU —
or a long fine-tune of l/m we can only partially afford on a single 12 GB card.

## 7. Takeaways
1. GETA now prunes YOLO26 (3 core fixes landed).
2. Structured pruning buys real **size + CPU-latency**, and **GPU latency on the larger models**.
3. YOLO26 is efficiently designed — **one-shot pruning collapses fast**; meaningful pruning needs recovery.
4. Two cheap, high-value recovery levers: **BN recalibration** (free) and the **right fine-tune LR (1e-3)**.
5. A clean pruned-beats-default win is a **bigger-GPU + longer-training** exercise (run x on ≥24 GB).

## Repro
- Code: `experiments/geta_yolo26/` (magnitude_sweep, bn_recalib, prune_finetune, geta_trainer,
  prune_speed_profile) + GETA fixes in `only_train_once/graph/graph.py`,
  `only_train_once/dependency_graph/pruning_dependency.py`, `only_train_once/graph/node_group.py`.
- Result files: `RESULTS_speed.md`, `RESULTS_no_retrain.md`, this file.

## 8. Final verdict: pruning vs TensorRT-FP16 for "faster + same mAP" (yolo26m demo, 2026-06-25)
| yolo26m | params | size | mAP50-95 | TRT-FP16 bs1 |
|---|---|---|---|---|
| default | 21.9M | 87.6MB | 0.518 | 9.28ms |
| pruned 10% + 15ep ft (lr1e-3) | 17.0M (-22%) | 68.3MB (-22%) | 0.456 (-12%) | 9.11ms (-2%) |
- default-m CUDA ~14.8ms -> TensorRT FP16 9.28ms = -37%, LOSSLESS (no pruning).
- Pruning m: -22% size but only -2% bs1 latency, at -12% mAP (unrecoverable on this setup) = bad trade.

CONCLUSION (whole study):
- "Faster + same mAP" on YOLO26 = **TensorRT FP16 on the dense model** (-37 to -41%, zero mAP loss). No pruning needed.
- Pruning gives smaller files + CPU speedups + (only on huge x) ~ -11..-30% GPU latency, but costs mAP YOLO26
  won't fully give back via fine-tuning (recovery plateaus ~85-91%). At bs1 on GPU it barely helps latency on n/s/m.
- INT8 QDQ: slower AND lossy for YOLO26 (attention/concat/NMS-free head won't INT8-fuse). Avoid.
- YOLO26 is too efficiently designed to prune losslessly; the framework value here is the (now-fixed) ability to
  prune it at all, useful when size/CPU matter and a few mAP points are acceptable.

## 9. Definitive result: pruned-x full-COCO fine-tune (2026-07-01, RTX Pro 6000 96GB)
The §6 "pruned beats default" attempt, finally run properly on a big GPU: prune yolo26x ->
**20.37M params**, 50-epoch full-COCO fine-tune (lr1e-3, batch32), then construct + val.
**Result: mAP50-95 = 0.450** (mAP50 0.618), construct_diff 1.2e-4.
- Pareto-DOMINATED: default-m (20.4M) = 0.518 beats it at equal params; default-s (9.5M, 0.472)
  beats it with <half the params. Recovered only 80% of dense-x (0.5626).
- Speed/size of the 20.37M subnet (bs1, 640): ONNX -65% (236->82MB), CPU -26% (588->437ms),
  **GPU ~0%** (9.82->9.94ms, launch-bound on Blackwell).
CONCLUSION: structured pruning is a dead end for YOLO26 (no redundant capacity). See ../CONCLUSION.md.
