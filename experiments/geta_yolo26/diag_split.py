"""Evidence: how does YOLO26's split trace? Look for Split/Slice/chunk nodes and
count op_names. Run with PYTHONPATH=/root/geta."""
import torch, collections
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
opcount = collections.Counter(getattr(n.op, "_type", n.op_name) for n in g.nodes.values())
print("=== op_name/_type counts (split-ish + chunk) ===")
for k, v in sorted(opcount.items()):
    if any(s in str(k).lower() for s in ("chunk", "split", "slice")):
        print(f"  {k}: {v}")

print("=== nodes whose op_name is chunk/split/slice (first 12, with traced str) ===")
shown = 0
for nid, n in g.nodes.items():
    nm = str(n.op_name).lower()
    ty = str(getattr(n.op, "_type", "")).lower()
    s = getattr(n, "torch_graph_str", "") or ""
    if ("chunk" in nm or "split" in nm or "slice" in nm or "split" in s.lower()) and shown < 12:
        print(f"  id={nid} op_name={n.op_name} type={getattr(n.op,'_type',None)} params={n.param_names}")
        print(f"    trace: {s[:200]}")
        shown += 1
