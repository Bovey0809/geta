"""Profile existing ONNX (FP32/FP16 + INT8 QDQ) on TensorRT, default vs pruned.
Verifies the TensorRT EP actually loaded (no silent CUDA fallback) by checking the
session's active provider. Usage: PYTHONPATH=/root/geta python profile_trt.py --tag x"""
import os, time, json, argparse
import numpy as np
import onnxruntime as ort

OUT = os.path.join(os.path.dirname(__file__), "out")


def trt(int8):
    o = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": OUT}
    if int8:
        o["trt_int8_enable"] = True
    return [("TensorrtExecutionProvider", o), "CUDAExecutionProvider", "CPUExecutionProvider"]


def bench(path, providers, imgsz=640, warmup=15, iters=80):
    sess = ort.InferenceSession(path, providers=providers)
    ep = sess.get_providers()[0]  # actual EP used (TensorrtExecutionProvider if loaded)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.mean(ts)), 3), ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x")
    ap.add_argument("--sparsity", type=int, default=50)
    ap.add_argument("--div", type=int, default=2)
    args = ap.parse_args()
    print("available providers:", ort.get_available_providers())
    t = args.tag
    files = {
        "default_fp16": (f"yolo26{t}_default.onnx", False),
        "default_int8": (f"yolo26{t}_default_int8qdq.onnx", True),
        "pruned_fp16": (f"yolo26{t}_pruned_s{args.sparsity}_d{args.div}.onnx", False),
        "pruned_int8": (f"yolo26{t}_pruned_s{args.sparsity}_d{args.div}_int8qdq.onnx", True),
    }
    res = {}
    for k, (fn, i8) in files.items():
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            res[k] = {"error": "missing"}; print("TRTPROF", k, "missing"); continue
        try:
            ms, ep = bench(p, trt(i8))
            res[k] = {"ms": ms, "ep": ep}
        except Exception as e:
            res[k] = {"error": str(e)[:100]}
        print("TRTPROF", k, json.dumps(res[k]))
    json.dump(res, open(os.path.join(OUT, f"yolo26{t}_trt_profile.json"), "w"), indent=2)
    print("TRTPROF_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
