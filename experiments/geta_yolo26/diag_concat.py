"""Construct pruned model and report the model.19 concat-source channels vs what
model.19.cv1 expects, to localize the concat-dependency mismatch.
Usage: PYTHONPATH=/root/geta python diag_concat.py yolo26m.pt"""
import sys, torch
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names

mp = sys.argv[1] if len(sys.argv) > 1 else "yolo26m.pt"
model = YOLO(mp).model
for n, p in model.named_parameters():
    if "running_mean" not in n:
        p.requires_grad = True
oto = OTO(model, torch.rand(1, 3, 640, 640))
oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
oto.random_set_zero_groups(target_group_sparsity=0.5)
oto.construct_subnet(out_dir="./cache")
comp = torch.load(oto.compressed_model_path, weights_only=False).eval()

def last_out(block):
    c = None
    for m in block.modules():
        if isinstance(m, torch.nn.Conv2d):
            c = m.out_channels
    return c

mods = comp.model
# model.18 = Concat[-1(=17), 13]; model.19.cv1 input should equal out(17)+out(13)
out13 = mods[13].cv2.conv.out_channels if hasattr(mods[13], "cv2") else last_out(mods[13])
out17 = mods[17].conv.out_channels if hasattr(mods[17], "conv") else last_out(mods[17])
cv1_in = mods[19].cv1.conv.in_channels
cv1_out = mods[19].cv1.conv.out_channels
print(f"RESULT model13_out={out13} model17_out={out17} sum={out13+out17} | model19.cv1_in={cv1_in} (cv1_out={cv1_out})")
print(f"CONSISTENT={out13+out17==cv1_in}")
