"""Diagnostic: find prunable param groups whose num_groups exceeds a member
param's prunable dimension (the cause of the index-out-of-bounds in
random_set_zero_groups). Run with PYTHONPATH=/root/geta."""
import torch
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import YOLO26_UNPRUNABLE

model = YOLO("yolo26n.pt").model
for n, p in model.named_parameters():
    if "running_mean" not in n:
        p.requires_grad = True
oto = OTO(model, torch.rand(1, 3, 640, 640))
oto.mark_unprunable_by_param_names(YOLO26_UNPRUNABLE)

mism = 0
for pg in oto._graph.get_param_groups():
    if not pg.get("is_prunable") or pg.get("is_auxiliary"):
        continue
    ng = pg.get("num_groups")
    for pn, pm, pt in zip(pg["p_names"], pg["params"], pg["p_transform"]):
        ts = str(pt)
        if "NO_PRUNE" in ts:
            continue
        # default/basic transform prunes dim0; TRANSPOSE prunes dim1
        dim = pm.shape[1] if "TRANSPOSE" in ts and pm.dim() > 1 else (pm.shape[0] if pm.dim() > 0 else 0)
        flag = "  <<< MISMATCH" if (ng and dim and ng > dim) else ""
        if flag:
            mism += 1
            print(f"ng_id={pg.get('id')} num_groups={ng} p={pn} shape={tuple(pm.shape)} transform={ts}{flag}")
print(f"=== total mismatched params: {mism} ===")
print(f"=== total prunable groups: {sum(1 for pg in oto._graph.get_param_groups() if pg.get('is_prunable') and not pg.get('is_auxiliary'))} ===")
