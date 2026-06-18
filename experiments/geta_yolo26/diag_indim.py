"""Replicate construct's in-dim resolution for model.19.cv1 to see exactly which
incoming node group is chosen and whether its redundant idxes match the concat.
Usage: PYTHONPATH=/root/geta python diag_indim.py yolo26m.pt"""
import sys, torch, numpy as np
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names

TARGET = "model.19.cv1.conv.weight"
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

# find the node holding TARGET
target_node = None
for n in g.nodes.values():
    if TARGET in getattr(n, "param_names", []):
        target_node = n; break
print("target_node", target_node.id if target_node else None, "node_group_ids", target_node.node_group_ids)
src_ng = g.node_groups[target_node.node_group_ids[0]]

# replicate find_incoming_node_group_stem_node
inc_ngs, inc_stems, visited = set(), set(), g.visited_dict()
def dfs(node):
    if src_ng.id not in node.node_group_ids and not src_ng.contain_node(node):
        inc_ngs.update(node.node_group_ids); return
    visited[node.id] = True
    for nin in g.incoming(node):
        if nin.is_stem():
            inc_stems.add(nin); return
        if not visited[nin.id]:
            dfs(nin)
dfs(target_node)
print("incoming_stem_nodes:", [s.id for s in inc_stems])
print("incoming_node_groups:", list(inc_ngs))

def szs(ng):
    imp = ng.pruning_important_idxes; red = ng.pruning_redundant_idxes
    return (ng.num_groups, imp.size if hasattr(imp,'size') else len(imp), red.size if hasattr(red,'size') else len(red),
            ng.is_prunable, ng.is_auxiliary)

for s in inc_stems:
    ng = g.node_groups[s.node_group_ids[0]]
    print(f"  STEM ng={s.node_group_ids[0]} num_groups/imp/red/prun/aux={szs(ng)}")
for ngid in inc_ngs:
    ng = g.node_groups[ngid]
    print(f"  NG  {ngid[:40]} num_groups/imp/red/prun/aux={szs(ng)}")
    if ng.is_auxiliary:
        for d in ng.dependent_node_groups:
            print(f"      dep num_groups/imp/red/prun/aux={szs(d)}")
print("cv1 expects input =", model.model[19].cv1.conv.weight.shape[1], "channels (original)")
