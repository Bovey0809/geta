"""Evidence round 2: structure of YOLO26 split nodes (in/out channels) and the
slice node trace strings. Run with PYTHONPATH=/root/geta."""
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
g = oto._graph

def osh(n):
    return getattr(n, "output_shape", None)

# reverse edges for incoming lookup
inc = {}
for e in g.edges:
    inc.setdefault(e[1], []).append(e[0])

print("=== SPLIT nodes (structure) ===")
for nid, n in g.nodes.items():
    if str(n.op_name).lower() == "split":
        print(f"SPLIT {nid} out_shape={osh(n)} type={getattr(n.op,'_type',None)} cfg={getattr(n.op,'cfg_params',None)}")
        for src in inc.get(nid, []):
            sn = g.nodes.get(src)
            print(f"   in  {src} {getattr(sn,'op_name',None)} out_shape={osh(sn) if sn else None}")
        for o in g.outgoing(n):
            print(f"   out {o.id} {o.op_name} out_shape={osh(o)} params={o.param_names[:2]}")

print("=== SLICE node trace strings (first 8) ===")
c = 0
for nid, n in g.nodes.items():
    if str(n.op_name).lower() == "slice" and c < 8:
        print(f"SLICE {nid} out_shape={osh(n)}: {repr(getattr(n,'torch_graph_str',''))[:240]}")
        c += 1
