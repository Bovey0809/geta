# Prints the detection head's final/output conv param names so we can mark them unprunable.
import torch
from ultralytics import YOLO

m = YOLO("yolo26n.pt").model
# The detect head is the last module in model.model; print its index + leaf conv weights.
head_idx = len(m.model) - 1
print("HEAD_INDEX", head_idx, "type", type(m.model[head_idx]).__name__)
for name, p in m.named_parameters():
    if name.startswith(f"model.{head_idx}.") and name.endswith(".weight") and p.dim() == 4:
        print("HEAD_CONV", name, tuple(p.shape))
