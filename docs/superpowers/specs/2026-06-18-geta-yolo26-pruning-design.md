# GETA × Ultralytics YOLO26 — Structured Pruning Experiment Design

**Date:** 2026-06-18
**Owner:** rick@ultralytics.com (GitHub: Bovey0809)
**Paper:** Qu et al., *Automatic Joint Structured Pruning and Quantization* (arXiv:2502.16638) — this repo (`microsoft/geta`, package `only_train_once`).

## 1. Goal & Success Criteria

Use GETA (OTO v3) to **structurally prune** `yolo26n` so the resulting model is **smaller and faster** while **COCO mAP50-95 stays the same** (within noise of the stock baseline). Deliver an **ONNX speed profile** of the pruned model vs. the stock model.

**Success = all of:**
- Pruned `yolo26n` reaches **COCO `val2017` mAP50-95 ≥ stock yolo26n − 0.3 pts** (target: parity).
- Pruned model is **smaller** (fewer params, lower GFLOPs, smaller file) than stock.
- Pruned ONNX is **measurably faster** (lower mean latency) than stock ONNX on the same 3080 + onnxruntime config.
- The whole pipeline is reproducible from the runbook in §8.

**Explicitly out of scope (phase 1):** quantization (GETA's mixed-precision path), variants other than `yolo26n`, multi-GPU. These are deferred follow-ups, not part of this spec.

## 2. Constraints & Environment

- **Hardware:** single NVIDIA RTX 3080 on an **AutoDL** instance. ~10–12 GB VRAM → small batch sizes; full-COCO fine-tune is the dominant cost and the schedule must fit one GPU.
- **Execution model:** **the user runs all commands on the server**; this plan is a copy-pasteable runbook. The agent has no direct server shell.
- **Code sync:** agent develops locally in `/home/rick/geta`, commits, and pushes to a **fork** (`Bovey0809/geta`) — `microsoft/geta` is not pushable (403). Server does `git pull` from the fork.
- **Data:** COCO is mounted in the AutoDL **public dataset** area (read-only, typically under `/root/autodl-pub/...`); we symlink it into the Ultralytics datasets dir rather than re-downloading. COCO8/COCO128 come from Ultralytics auto-download for smoke tests.
- **Framework:** `pytorch>=2.0`; `ultralytics` at a version that ships YOLO26.

## 3. Locked Decisions

| Decision | Choice |
|---|---|
| Compression scope (phase 1) | **Pruning only** (no quantization) |
| Variant (phase 1) | **`yolo26n` only** |
| De-risking | **Smoke-test on COCO8/128 first**, then full COCO |
| GETA ↔ YOLO integration | **Approach A** — GETA optimizer inside the Ultralytics `DetectionTrainer` |
| Server access | User runs; agent hands off scripts/commands |
| Code sync | Fork `Bovey0809/geta` + `git pull` on server |
| Target sparsity | Start at `target_group_sparsity = 0.3`–`0.5` (one value first; sweep later if parity holds) |

## 4. Architecture / Approach

### 4.1 GETA usage pattern (grounded in `sanity_check/test_yolov8.py`)
```python
from only_train_once import OTO
oto = OTO(model.model, dummy_input)            # raw nn.Module + dummy input
oto.mark_unprunable_by_param_names([...])      # YOLO26 detection-head output convs
oto.visualize(view=False, out_dir=...)         # inspect prunable groups
optimizer = oto.geta(target_group_sparsity=..., variant="adam", ...)  # step-based schedule
# ... train with optimizer ...
oto.construct_subnet(out_dir=...)              # emits full + compressed model
```
- The model passed to OTO is the **un-quantized** `model.model`, so `oto.geta(...)` acts as **pure structured pruning** (no quant layers are inserted; bit-width args are inert). If a dedicated pruning optimizer (`HESSO`) proves cleaner for pruning-only, use it instead — resolve during implementation.
- **Unprunable layers:** the v8 head marks `model.22.*` output convs (DFL + cv2/cv3 final 1×1s) unprunable so output channel counts stay valid. **YOLO26's head differs (NMS-free, DFL removed); its module index and param names must be discovered** (Phase 2) and marked accordingly. This is the central adaptation.

### 4.2 Trainer integration (Approach A)
- `GetaDetectionTrainer(ultralytics...DetectionTrainer)`:
  - Override `build_optimizer(...)` to construct the `OTO` graph on `self.model` and return `oto.geta(...)`.
  - Register a **callback** (`on_train_batch_end` / optimizer-step hook) so GETA's projection/pruning windows advance with the **global optimizer step** (`start_projection_step`, `projection_steps`, `start_pruning_step`, `pruning_steps` are all in units of optimizer steps = batches; size them from `len(train_loader) * epochs`).
  - **Start with AMP off and EMA off** for step-semantics correctness; re-enable once mAP looks right and re-verify.
  - Keep YOLO's native dataloader, augmentation, loss, and **official COCO mAP validator** untouched — this is why Approach A was chosen (trustworthy mAP).
- After training: `oto.construct_subnet()` → load `compressed_model_path` → wrap back into a YOLO model object for export/val.

### 4.3 ONNX profiling
- Export both stock and pruned to ONNX (Ultralytics `export(format="onnx")` and/or OTO's ONNX export), fixed input `640×640`, opset pinned, same dynamic/static setting.
- Profile with onnxruntime: warmup N iters, measure mean/p50/p95 latency over M iters, batch=1, on the 3080 (CUDAExecutionProvider) and on CPU. Report alongside params/GFLOPs/size and COCO mAP.

## 5. Phased Plan

- **Phase 0 — Repo sync.** Create fork `Bovey0809/geta`; add `fork` remote locally; verify push. Establish loop: edit → push `fork` → `git pull` on server.
- **Phase 1 — Server env bring-up.** Verify `nvidia-smi`, CUDA, `torch>=2.0` (`torch.cuda.is_available()`); install GETA deps (onnx, onnxruntime-gpu, graphviz, etc.) + `ultralytics`; symlink COCO from `autodl-pub`; **baseline check**: run `sanity_check/sanity_check.py` (or `test_yolov8.py`) and reproduce **stock `yolo26n` COCO mAP** to lock the parity target.
- **Phase 2 — GETA traces YOLO26n (GO/NO-GO gate).** New `sanity_check/test_yolo26.py` adapted from v8: trace graph, discover + mark head output convs unprunable, `visualize`, `random_set_zero_groups()` → `construct_subnet()`, assert output diff `< 1e-4`. **No COCO compute spent until this passes.**
- **Phase 3 — Trainer integration + small smoke.** Implement `GetaDetectionTrainer` + callback; smoke on **COCO8 then COCO128** (minutes): confirm loss decreases, group sparsity ramps to target, validation runs, `construct_subnet` yields a smaller net that still forward-passes.
- **Phase 4 — Full-COCO prune + fine-tune.** One `target_group_sparsity` on full COCO with a schedule sized to the dataloader; checkpoint; `construct_subnet()`. Tune batch/epochs to fit the 3080.
- **Phase 5 — Export, profile, compare.** Export pruned subnet to ONNX; profile latency vs stock; compute pruned COCO mAP50-95; fill the metrics table; write up findings.

## 6. Metrics & Reporting

| Metric | Stock yolo26n | Pruned yolo26n (sparsity=X) | Δ |
|---|---|---|---|
| COCO mAP50-95 (val2017) | | | |
| COCO mAP50 | | | |
| Params (M) | | | |
| GFLOPs @640 | | | |
| Model size (MB, .pt / .onnx) | | | |
| ONNX latency bs1 — 3080 (mean / p95 ms) | | | |
| ONNX latency bs1 — CPU (mean ms) | | | |

Profiling protocol fixed per §4.3 (same opset, input size, warmup/iters, provider) so the comparison is apples-to-apples.

## 7. Risks & Mitigations

- **R1 — GETA cannot trace YOLO26 (new modules / NMS-free head).** *Highest risk.* Mitigation: Phase 2 gate before any training; if tracing breaks, narrow via `skip_patterns`/`mark_unprunable`, inspect with `oto.visualize`, fall back to pruning only the backbone+neck. If the head is fundamentally untraceable, report it as a finding and scope to traceable subgraphs.
- **R2 — GETA schedule mis-meshes with Ultralytics AMP/EMA/grad-accum** → silent mAP loss. Mitigation: AMP+EMA off initially; verify sparsity ramp and loss on COCO8 before full runs.
- **R3 — Single-3080 compute** makes full-COCO fine-tune slow. Mitigation: smoke small first; size schedule to one epoch-multiple that fits; consider shorter fine-tune + parity check before long runs.
- **R4 — ONNX speedup doesn't materialize** (pruning reduces channels but runtime/kernels don't benefit at bs1). Mitigation: profile both CPU and CUDA EP; report GFLOPs/params reduction even if latency gain is provider-dependent; note honestly.
- **R5 — `construct_subnet` output diff fails** (>1e-4). Mitigation: treat as graph-correctness bug; debug unprunable marking before trusting any trained subnet.
- **R6 — COCO path/layout on AutoDL** differs from Ultralytics' expected `coco.yaml` layout. Mitigation: validate with the `dataset-check` skill before training.

## 8. Runbook (filled in as phases land)

Commands are authored per phase and handed to the user to run on the AutoDL server; outputs are pasted back. Phase 0–1 commands are produced first; later phases unlock after their predecessor's gate passes.

## 9. Open Questions (non-blocking)

- Exact `target_group_sparsity` for the headline run (start 0.3–0.5; sweep if parity holds).
- Whether to re-enable AMP/EMA for the final full-COCO run after correctness is confirmed.
- ONNX opset + dynamic-axes choice to standardize (pin once, document in runbook).
