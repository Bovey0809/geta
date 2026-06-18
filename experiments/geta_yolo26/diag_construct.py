"""Find the conv in the constructed (pruned) model whose weight in-channels does not
match the actual tensor it receives, and report the layer + its input producers.
Usage: PYTHONPATH=/root/geta python diag_construct.py yolo26m.pt"""
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

mismatches = []
hooks = []
def mk(name, m):
    def pre(mod, inp):
        x = inp[0]
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            exp = mod.weight.shape[1] * (mod.groups if hasattr(mod, "groups") else 1)
            if x.shape[1] != mod.weight.shape[1] and mod.groups == 1:
                mismatches.append((name, tuple(mod.weight.shape), x.shape[1]))
    return pre
for name, m in comp.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        hooks.append(m.register_forward_pre_hook(mk(name, m)))

try:
    with torch.no_grad():
        comp(torch.rand(1, 3, 640, 640))
except Exception as e:
    print("FORWARD_ERROR", str(e)[:160])

print("=== first mismatched conv ===")
for nm, wsh, actual in mismatches[:3]:
    print(f"MISMATCH conv={nm} weight={wsh} (expects in={wsh[1]}) actual_in={actual}")
    # show the original module's siblings around this layer index
