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
