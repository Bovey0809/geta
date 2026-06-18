"""Construct pruned model repeatedly; for each, compare the ACTUAL constructed
model.19.cv1 weight input dim against the actual concat source outputs
(model.13 + model.17). Stops at the first inconsistency with full breakdown.
Usage: PYTHONPATH=/root/geta python diag_concat.py yolo26m.pt"""
import sys, torch
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names

mp = sys.argv[1] if len(sys.argv) > 1 else "yolo26m.pt"

def last_conv_out(block):
    c = None
    for m in block.modules():
        if isinstance(m, torch.nn.Conv2d):
            c = m.weight.shape[0]
    return c

for attempt in range(12):
    model = YOLO(mp).model
    for n, p in model.named_parameters():
        if "running_mean" not in n:
            p.requires_grad = True
    oto = OTO(model, torch.rand(1, 3, 640, 640))
    oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
    oto.random_set_zero_groups(target_group_sparsity=0.5)
    oto.construct_subnet(out_dir="./cache")
    comp = torch.load(oto.compressed_model_path, weights_only=False)
    mods = comp.model
    out13 = mods[13].cv2.conv.weight.shape[0] if hasattr(mods[13], "cv2") else last_conv_out(mods[13])
    out17 = mods[17].conv.weight.shape[0] if hasattr(mods[17], "conv") else last_conv_out(mods[17])
    cv1_in = mods[19].cv1.conv.weight.shape[1]
    ok = (out13 + out17) == cv1_in
    print(f"attempt {attempt}: model13_out={out13} model17_out={out17} sum={out13+out17} cv1_weight_in={cv1_in} CONSISTENT={ok}")
    if not ok:
        print(f"INCONSISTENT: cv1 input weight={cv1_in} but concat={out13+out17} "
              f"(diff={cv1_in-(out13+out17)}; model13 contributes {out13}, model17 {out17})")
        break
