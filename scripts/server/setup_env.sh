#!/usr/bin/env bash
set -euo pipefail
echo "== GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== Torch =="
python - <<'PY'
import torch
assert torch.__version__ >= "2", f"need torch>=2.0, got {torch.__version__}"
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "== Install deps (GETA + ultralytics + onnx tooling) =="
pip install -q "ultralytics>=8.3" onnx onnxruntime-gpu graphviz
# GETA package is used in-place from the repo via PYTHONPATH; install its imports' deps:
pip install -q numpy pillow

echo "== Import checks =="
python - <<'PY'
import torch, ultralytics, onnx, onnxruntime
import sys; sys.path.insert(0, ".")
import only_train_once
print("ultralytics", ultralytics.__version__)
print("onnxruntime providers", onnxruntime.get_available_providers())
# YOLO26 availability probe (does NOT download weights yet)
from ultralytics import YOLO
print("YOLO import OK")
PY
echo "ENV OK"
