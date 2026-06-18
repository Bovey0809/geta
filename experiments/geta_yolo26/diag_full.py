"""Seed-controlled: find a failing seed, then dump the aux group feeding model.19.cv1
with each dependent group's num_groups / important / redundant / transform, plus the
constructed channel counts. One run pinpoints where the input-pruning count is lost.
Usage: PYTHONPATH=/root/geta python diag_full.py yolo26m.pt"""
import sys, torch, numpy as np
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names

TARGET = "model.19.cv1.conv.weight"
mp = sys.argv[1] if len(sys.argv) > 1 else "yolo26m.pt"

def last_conv_out(block):
    c = None
    for m in block.modules():
        if isinstance(m, torch.nn.Conv2d):
            c = m.weight.shape[0]
    return c

def build(seed):
    np.random.seed(seed)
    model = YOLO(mp).model
    for n, p in model.named_parameters():
        if "running_mean" not in n:
            p.requires_grad = True
    oto = OTO(model, torch.rand(1, 3, 640, 640))
    oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
    np.random.seed(seed)
    oto.random_set_zero_groups(target_group_sparsity=0.5)
    return oto

def szs(ng):
    imp = ng.pruning_important_idxes; red = ng.pruning_redundant_idxes
    iz = imp.size if hasattr(imp, 'size') else len(imp)
    rz = red.size if hasattr(red, 'size') else len(red)
    return ng.num_groups, iz, rz, ng.is_prunable, ng.is_auxiliary

for seed in range(30):
    oto = build(seed)
    g = oto._graph
    g.set_pruning_redundant_idxes()
    tn = next((n for n in g.nodes.values() if TARGET in getattr(n, "param_names", [])), None)
    src_ng = g.node_groups[tn.node_group_ids[0]]
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
    dfs(tn)
    # snapshot aux + dep sizes BEFORE construct mutates anything
    def pnames(ng):
        ns = []
        for nd in ng.nodes.values():
            ns += list(getattr(nd, "param_names", []))
        return ns[:3]
    snap = []
    for ngid in inc_ngs:
        ng = g.node_groups[ngid]
        deps = [(szs(d), pnames(d)) for d in ng.dependent_node_groups] if ng.is_auxiliary else []
        snap.append((ngid, szs(ng), deps))
    aux_red = None
    for ngid in inc_ngs:
        ng = g.node_groups[ngid]
        if ng.is_auxiliary:
            r = ng.pruning_redundant_idxes
            aux_red = r.size if hasattr(r, 'size') else len(r)
    oto.construct_subnet(out_dir="./cache")
    comp = torch.load(oto.compressed_model_path, weights_only=False)
    mods = comp.model
    out13 = mods[13].cv2.conv.weight.shape[0] if hasattr(mods[13], "cv2") else last_conv_out(mods[13])
    out17 = mods[17].conv.weight.shape[0] if hasattr(mods[17], "conv") else last_conv_out(mods[17])
    cv1_in = mods[19].cv1.conv.weight.shape[1]
    if out13 + out17 != cv1_in:
        print(f"FAIL seed={seed}: out13={out13} out17={out17} concat={out13+out17} cv1_in={cv1_in} aux_redundant={aux_red}")
        for ngid, s, deps in snap:
            print(f"  incoming {ngid[:30]} szs(num_groups,imp,red,prun,aux)={s}")
            for dszs, dnames in deps:
                print(f"      dep szs={dszs} params={dnames}")
        break
else:
    print("no failing seed found in range")
