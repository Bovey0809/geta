"""Speed/size profile of pruned yolo26 vs default — faithful to the FINE-TUNED model.

Exports the default model and a pruned model to ONNX (CPU-only, no GPU contention)
and benchmarks latency. The pruned model is either:
  --compressed PATH : the trained GETA construct_subnet output (DetectionModel_compressed.pt)
                      — the exact architecture+weights whose COCO mAP we report.
  (fallback)        : a fresh structural prune at --sparsity / --divisible, used to get
                      the representative pruned ARCHITECTURE before the fine-tune finishes
                      (latency depends only on channel counts, not on weight values).

Two phases so we can work while the GPU is busy training:
  --phase export_cpu : export both ONNX + params/size + CPU latency (safe during training)
  --phase gpu        : TensorRT-FP16 + CUDA latency (run on a CLEAN GPU, post-training)
  --phase all        : everything (default)

Usage:
  PYTHONPATH=/root/geta python profile_pruned_x.py --tag x --sparsity 0.5 --phase export_cpu
  PYTHONPATH=/root/geta python profile_pruned_x.py --tag x \
      --compressed experiments/geta_yolo26/out/geta_x_s50_full/DetectionModel_compressed.pt --phase gpu
"""
import os, time, json, argparse
import numpy as np
import torch

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


def build_pruned(model_pt, sparsity, divisible, imgsz):
    """Fresh structural prune -> constructed subnet (representative architecture)."""
    from only_train_once import OTO
    from ultralytics import YOLO
    from sanity_check.test_yolo26 import yolo26_unprunable_names, _max_diff
    last_err = None
    for _ in range(12):
        model = YOLO(model_pt).model
        for nm, p in model.named_parameters():
            if "running_mean" not in nm:
                p.requires_grad = True
        oto = OTO(model, torch.rand(1, 3, imgsz, imgsz))
        oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        oto._graph.random_set_zero_groups(target_group_sparsity=sparsity, num_group_divisible=divisible)
        oto.construct_subnet(out_dir=OUT)
        try:
            cand = torch.load(oto.compressed_model_path, weights_only=False)
            full = torch.load(oto.full_group_sparse_model_path, weights_only=False)
            with torch.no_grad():
                diff = _max_diff(full(torch.rand(1, 3, imgsz, imgsz)), cand(torch.rand(1, 3, imgsz, imgsz)))
            if diff > 1e-3:
                last_err = f"construct diff {diff:.2e}"; continue
            return cand
        except Exception as e:
            last_err = str(e)[:160]; continue
    raise RuntimeError(f"fresh prune failed: {last_err}")


def bench(path, providers, imgsz=640, warmup=20, iters=100):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    ep = sess.get_providers()[0]
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"ep": ep, "mean_ms": round(float(ts.mean()), 3),
            "p50_ms": round(float(np.percentile(ts, 50)), 3),
            "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--divisible", type=int, default=1, help="match the trained run (geta_trainer uses 1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--compressed", default=None, help="trained construct_subnet output .pt (preferred)")
    ap.add_argument("--phase", default="all", choices=["export_cpu", "gpu", "all"])
    args = ap.parse_args()
    from ultralytics import YOLO

    res_path = os.path.join(OUT, f"yolo26{args.tag}_pruned_profile.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {"tag": args.tag, "sparsity": args.sparsity}
    default_onnx = os.path.join(OUT, f"yolo26{args.tag}_default.onnx")
    pruned_onnx = os.path.join(OUT, f"yolo26{args.tag}_prunedft.onnx")

    if args.phase in ("export_cpu", "all"):
        # default
        ymodel = YOLO(f"yolo26{args.tag}.pt").model
        res["params_M"] = {"default": round(n_params(ymodel), 3)}
        export_onnx(ymodel, default_onnx, args.imgsz)
        # pruned
        pruned = None
        if args.compressed and os.path.exists(args.compressed):
            pruned = torch.load(args.compressed, weights_only=False)
            res["pruned_source"] = "trained_construct"
        else:
            # random structural prune is inconsistent on x (construct diff blows up);
            # only the trained HESSO construct is reliable. Degrade gracefully: do the
            # default-x baseline now, defer the pruned half to --compressed post-training.
            try:
                pruned = build_pruned(f"yolo26{args.tag}.pt", args.sparsity, args.divisible, args.imgsz)
                res["pruned_source"] = "fresh_structural_prune(representative)"
            except Exception as e:
                res["pruned_source"] = f"DEFERRED (fresh prune unreliable: {str(e)[:80]})"
        res["onnx_size_MB"] = {"default": round(os.path.getsize(default_onnx) / 1e6, 2)}
        cpu_lat = {"default": bench(default_onnx, ["CPUExecutionProvider"], args.imgsz)}
        if pruned is not None:
            res["params_M"]["pruned"] = round(n_params(pruned), 3)
            export_onnx(pruned, pruned_onnx, args.imgsz)
            res["onnx_size_MB"]["pruned"] = round(os.path.getsize(pruned_onnx) / 1e6, 2)
            cpu_lat["pruned"] = bench(pruned_onnx, ["CPUExecutionProvider"], args.imgsz)
        res.setdefault("latency", {})["CPU"] = cpu_lat
        json.dump(res, open(res_path, "w"), indent=2)
        print("EXPORT_CPU_DONE", json.dumps(res))

    if args.phase in ("gpu", "all"):
        trt = ("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                             "trt_engine_cache_path": OUT})
        runs = {"TensorRT_fp16": [trt, "CUDAExecutionProvider", "CPUExecutionProvider"],
                "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
        for key, provs in runs.items():
            entry = {}
            try:
                entry["default"] = bench(default_onnx, provs, args.imgsz)
                if os.path.exists(pruned_onnx):
                    entry["pruned"] = bench(pruned_onnx, provs, args.imgsz)
            except Exception as e:
                entry["error"] = str(e)[:200]
            res.setdefault("latency", {})[key] = entry
        json.dump(res, open(res_path, "w"), indent=2)
        print("GPU_PROFILE_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
