# YOLO26 GETA structured pruning — speed/size (pruned vs default ONNX)

Setup: 50% group sparsity, structural prune verified full-vs-compressed diff < 1e-4,
ONNX (opset 17), batch=1, 640x640, onnxruntime on RTX 3080 Ti (CUDA EP) + CPU EP.
C2PSA attention blocks + detection-head output convs excluded from pruning.
NOTE: accuracy (COCO mAP) requires GETA fine-tuning on COCO — not yet measured.

| Model | Params M (def->pruned) | ONNX MB | GPU bs1 ms (def->pruned) | CPU bs1 ms (def->pruned) |
|-------|------------------------|---------|--------------------------|--------------------------|
| n | 2.57 -> 1.39 (-46%) | 10.4 -> 5.8 | 7.73 -> 8.07 | 68.8 -> 58.1 (-16%) |
| s | 10.0 -> 5.43 (-46%) | 40.1 -> 21.9 | 9.11 -> 8.35 (-8%) | 122.8 -> 78.2 (-36%) |
| m | 21.9 -> 9.45 (-57%) | 87.6 -> 38.0 | 14.24 -> 11.34 (-20%) | 308.9 -> 140.1 (-55%) |
| l | 26.3 -> 10.9 (-58%) | 105 -> 44 | 16.04 -> 12.52 (-22%) | 392.8 -> 172.4 (-56%) |
| x | 59.0 -> 24.5 (-58%) | 236 -> 98 | 24.06 -> 16.94 (-30%) | 756.7 -> 329.0 (-57%) |

GPU speedup scales with model size (nano overhead-bound; m/l/x compute-bound, 20-30% faster).
CPU speedups large across the board.

## group_divisible=16 (Tensor-Core alignment, per GETA issue #15) + TensorRT — sparsity 0.5
GPU latency bs1, pruned vs default (ms):
| model | TRT div2 | TRT div16 | CUDA div2 | CUDA div16 | speedup |
|---|---|---|---|---|---|
| n | 8.49/7.98 | 8.46/8.26 | 8.69/8.60 | 8.68/8.48 | ~flat (launch-bound) |
| m | 12.10/14.60 | 12.24/14.89 | 12.13/14.76 | 12.18/14.73 | -17% |
| x | 17.47/24.48 | 17.66/24.50 | 17.52/24.56 | 17.59/24.41 | -29% |

FINDINGS:
- **divisible=16 makes NO measurable difference for YOLO26** (same params, same latency as div=2).
  YOLO26's channel widths are already 16/32-friendly, so pruning already lands Tensor-Core-aligned.
  Issue #15's alignment fix mattered for arbitrary-width models (RF-DETR/ResNet), not YOLO26.
- **TensorRT ~= CUDA EP here**: no big fusion win, because there's no quantization to fuse.
- GPU speedup is real and scales with size (nano flat, m -17%, x -29%) on BOTH runtimes -- already
  present at div=2; pruning alone delivers it.
- The untapped lever for a large (~5x) speedup (per #15) is INT8 quantization + explicit QDQ export
  -> TensorRT INT8 fusion. Pruning alone gives the -17 to -29% measured above.

## INT8 QDQ export + REAL TensorRT (yolo26x, bs1, 640) — corrects earlier "TRT"=CUDA fallback
IMPORTANT: earlier "TensorRT_fp16" numbers were actually CUDA EP (TensorRT lib wasn't installed
-> silent fallback). After installing TensorRT 10.16 (libnvinfer.so.10), verified EP=TensorrtExecutionProvider:

| config | latency | note |
|---|---|---|
| default, CUDA EP        | 24.5 ms | (onnxruntime CUDA) |
| default, TensorRT FP16  | 14.4 ms | TRT fusion: -41% vs CUDA |
| pruned,  TensorRT FP16  | 12.8 ms | + pruning: -11% more (~-48% vs default-CUDA) |
| default, TensorRT INT8-QDQ | 40.0 ms | SLOWER |
| pruned,  TensorRT INT8-QDQ | 30.4 ms | SLOWER |
INT8 model files are ~4x smaller (236->60.7MB, 98->26.1MB).

FINDINGS:
- The big speedup for YOLO26 is **TensorRT FP16** (-41% vs onnxruntime CUDA), NOT INT8.
- **INT8 QDQ is ~2.5-3x SLOWER for YOLO26** even on real TensorRT: its attention + concat-heavy +
  NMS-free head can't stay in INT8, so TRT pays constant INT8<->FP reformatting overhead. The
  issue-#15 5x INT8 win was on ResNet/RF-DETR (cleanly fusible); it does NOT transfer to YOLO26.
- Best real-world config measured: **pruned + TensorRT FP16 = 12.8 ms** (vs 24.5 ms default-CUDA).
- INT8 still wins on model SIZE (~4x smaller) if storage/bandwidth-bound.
