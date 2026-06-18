"""Regression harness: for each yolo26 size, construct a pruned subnet over several
seeds and assert full-vs-compressed output diff < 1e-4 (the real correctness check).
Baseline before the dep-graph fix: n/s pass, m/l/x fail. Goal after fix: all pass,
n/s unchanged. Usage: PYTHONPATH=/root/geta python verify_family.py n s m  [seeds]"""
import sys, torch, numpy as np
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names, _max_diff

models = [a for a in sys.argv[1:] if a in ("n", "s", "m", "l", "x")]
seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0, 1, 2]
models = models or ["n", "s", "m"]

for size in models:
    mp = f"yolo26{size}.pt"
    npass = 0
    detail = []
    for seed in seeds:
        try:
            np.random.seed(seed)
            model = YOLO(mp).model
            for n, p in model.named_parameters():
                if "running_mean" not in n:
                    p.requires_grad = True
            oto = OTO(model, torch.rand(1, 3, 640, 640))
            oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
            np.random.seed(seed)
            oto.random_set_zero_groups(target_group_sparsity=0.5)
            oto.construct_subnet(out_dir="./cache")
            full = torch.load(oto.full_group_sparse_model_path, weights_only=False)
            comp = torch.load(oto.compressed_model_path, weights_only=False)
            with torch.no_grad():
                diff = _max_diff(full(torch.rand(1, 3, 640, 640)), comp(torch.rand(1, 3, 640, 640)))
            # NOTE: full vs compressed must match on the SAME input
            x = torch.rand(1, 3, 640, 640)
            with torch.no_grad():
                diff = _max_diff(full(x), comp(x))
            ok = diff <= 1e-4
            npass += ok
            detail.append(f"s{seed}:{'OK' if ok else f'diff={diff:.2e}'}")
        except Exception as e:
            detail.append(f"s{seed}:ERR({str(e)[:60]})")
    print(f"yolo26{size}: {npass}/{len(seeds)} pass  [{', '.join(detail)}]")
