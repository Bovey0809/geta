"""Structurally prune yolo26n with GETA (target group sparsity), construct the
subnet, export pruned + default to ONNX, and profile latency / params / size.

NOTE: this uses random group selection to produce the pruned *architecture* for a
faithful SPEED/SIZE comparison. Accuracy recovery needs GETA fine-tuning on COCO
(separate step); the mAP of this particular structurally-pruned model is not meaningful.
"""
import os, time, argparse, json
import numpy as np
import torch
from only_train_once import OTO
from ultralytics import YOLO
from sanity_check.test_yolo26 import yolo26_unprunable_names, _max_diff

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
OPSET = 17


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def export_onnx(module, path, imgsz=640):
    module = module.eval().cpu()
    dummy = torch.rand(1, 3, imgsz, imgsz)
    torch.onnx.export(module, dummy, path, opset_version=OPSET,
                      input_names=["images"], output_names=["output0"],
                      do_constant_folding=True)
    return path


def bench(path, providers, imgsz=640, warmup=20, iters=100):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean_ms": round(float(ts.mean()), 3), "p50_ms": round(float(np.percentile(ts, 50)), 3),
            "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--divisible", type=int, default=2,
                    help="group_divisible: 16/32 aligns pruned channels to Tensor Cores (issue #15)")
    args = ap.parse_args()
    tag = os.path.splitext(os.path.basename(args.model))[0]

    # 1) default model + ONNX
    y = YOLO(args.model)
    default_params = n_params(y.model)
    default_onnx = os.path.join(OUT, f"{tag}_default.onnx")
    export_onnx(YOLO(args.model).model, default_onnx, args.imgsz)

    # 2) prune (structural) + construct subnet. random_set_zero_groups can
    # intermittently produce a channel-inconsistent subnet for the larger
    # variants (multi-source concat pruning under some random zero patterns), so
    # retry until the constructed model forwards/exports cleanly.
    pruned_onnx = os.path.join(OUT, f"{tag}_pruned_s{int(args.sparsity*100)}_d{args.divisible}.onnx")
    compressed, pruned_params, last_err, attempts = None, None, None, 0
    for attempt in range(12):
        attempts = attempt + 1
        model = YOLO(args.model).model
        for nm, p in model.named_parameters():
            if "running_mean" not in nm:
                p.requires_grad = True
        oto = OTO(model, torch.rand(1, 3, args.imgsz, args.imgsz))
        oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        oto.random_set_zero_groups(target_group_sparsity=args.sparsity, num_group_divisible=args.divisible)
        oto.construct_subnet(out_dir=OUT)
        try:
            cand = torch.load(oto.compressed_model_path, weights_only=False)
            full = torch.load(oto.full_group_sparse_model_path, weights_only=False)
            # verify numerical correctness: compressed must match full (zeroed) net
            xv = torch.rand(1, 3, args.imgsz, args.imgsz)
            with torch.no_grad():
                diff = _max_diff(full(xv), cand(xv))
            if diff > 1e-4:
                last_err = f"output diff {diff:.3e} > 1e-4 (inconsistent prune)"
                continue
            export_onnx(cand, pruned_onnx, args.imgsz)
            compressed = cand
            pruned_params = n_params(cand)
            break
        except Exception as e:
            last_err = str(e)[:160]
            continue
    if compressed is None:
        raise RuntimeError(f"construct/export failed after {attempts} attempts: {last_err}")

    # 3) profile
    res = {"model": tag, "sparsity": args.sparsity, "divisible": args.divisible, "construct_attempts": attempts,
           "params_M": {"default": round(default_params, 3), "pruned": round(pruned_params, 3)},
           "onnx_size_MB": {"default": round(os.path.getsize(default_onnx) / 1e6, 2),
                            "pruned": round(os.path.getsize(pruned_onnx) / 1e6, 2)}}
    trt = ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                         "trt_engine_cache_path": OUT})
    runs = {"TensorRT_fp16": [trt, "CUDAExecutionProvider", "CPUExecutionProvider"],
            "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "CPU": ["CPUExecutionProvider"]}
    for key, provs in runs.items():
        prov = key  # backward-compat label; provs is the actual provider list
        try:
            res.setdefault("latency", {})[prov] = {
                "default": bench(default_onnx, provs, args.imgsz),
                "pruned": bench(pruned_onnx, provs, args.imgsz)}
        except Exception as e:
            res.setdefault("latency", {})[prov] = {"error": str(e)[:200]}
    with open(os.path.join(OUT, f"{tag}_speed_profile.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("PROFILE", json.dumps(res))


if __name__ == "__main__":
    main()
