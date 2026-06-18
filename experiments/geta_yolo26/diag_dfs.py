"""On the failing seed, inspect the aux group feeding model.19.cv1: how many concat
nodes it has, and for each concat node what its incoming nodes resolve to (group +
param identity + how many groups each incoming node belongs to).
Usage: PYTHONPATH=/root/geta python diag_dfs.py yolo26m.pt"""
import sys, torch, numpy as np
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names

TARGET = "model.19.cv1.conv.weight"
mp = sys.argv[1] if len(sys.argv) > 1 else "yolo26m.pt"
np.random.seed(0)
model = YOLO(mp).model
for n, p in model.named_parameters():
    if "running_mean" not in n:
        p.requires_grad = True
oto = OTO(model, torch.rand(1, 3, 640, 640))
oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
np.random.seed(0)
oto.random_set_zero_groups(target_group_sparsity=0.5)
g = oto._graph

tn = next(n for n in g.nodes.values() if TARGET in getattr(n, "param_names", []))
src_ng = g.node_groups[tn.node_group_ids[0]]
inc_ngs, visited = set(), g.visited_dict()
def dfs(node):
    if src_ng.id not in node.node_group_ids and not src_ng.contain_node(node):
        inc_ngs.update(node.node_group_ids); return
    visited[node.id] = True
    for nin in g.incoming(node):
        if nin.is_stem():
            return
        if not visited[nin.id]:
            dfs(nin)
dfs(tn)

def pid(ngid):
    ng = g.node_groups[ngid]
    for nd in ng.nodes.values():
        if getattr(nd, "param_names", []):
            return ngid[:24] + " :: " + nd.param_names[0]
    return ngid[:24]

for ngid in inc_ngs:
    ng = g.node_groups[ngid]
    if not ng.is_auxiliary:
        continue
    cns = ng.get_concat_nodes()
    print(f"AUX {ngid[:24]} concat_nodes={[c.id for c in cns.values()] if hasattr(cns,'values') else [c.id for c in cns]}")
    clist = list(cns.values()) if hasattr(cns, "values") else list(cns)
    for c in clist:
        print(f"  concat {c.id} incoming:")
        for nin in g.incoming(c):
            if nin.id == "dummy_input":
                continue
            print(f"    in {nin.id} belongs_to_{len(nin.node_group_ids)}_groups -> uses[0]={pid(nin.node_group_ids[0])}")
