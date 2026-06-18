"""Inspect auxiliary (concat) node groups: for each, compare the offset accounting
(sum of important+redundant sizes of dependent groups) against num_groups, to find
where a chunk/MULTIHEAD_HEADDIM dependent group makes the concat indices wrong.
Usage: PYTHONPATH=/root/geta python diag_aux.py yolo26m.pt"""
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
g = oto._graph
g.set_pruning_redundant_idxes()

for ng in g.node_groups.values():
    if not getattr(ng, "is_auxiliary", False):
        continue
    deps = getattr(ng, "dependent_node_groups", [])
    if not deps:
        continue
    total_sz = 0
    rows = []
    for d in deps:
        imp = getattr(d, "pruning_important_idxes", [])
        red = getattr(d, "pruning_redundant_idxes", [])
        imp_sz = imp.size if hasattr(imp, "size") else len(imp)
        red_sz = red.size if hasattr(red, "size") else len(red)
        total_sz += imp_sz + red_sz
        rows.append(f"dep num_groups={d.num_groups} imp={imp_sz} red={red_sz} (imp+red={imp_sz+red_sz})")
    flag = "  <<< offset_sum != num_groups" if total_sz != ng.num_groups else ""
    print(f"AUX num_groups={ng.num_groups} sum(imp+red over deps)={total_sz}{flag}")
    for r in rows:
        print("   " + r)
