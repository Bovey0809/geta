"""Post-training static INT8 quantization to explicit QDQ ONNX (the format TensorRT
fuses into INT8 kernels — per GETA issue #15), then profile TensorRT INT8 vs FP16 for
default vs pruned YOLO26. Calibrates on COCO val images.

Usage: PYTHONPATH=/root/geta python quantize_qdq.py --tag x --sparsity 50 --div 2 --calib 200
"""
import os, glob, time, json, argparse
import numpy as np
import cv2
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, QuantFormat, QuantType,
                                      CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
VAL_IMGS = "/root/autodl-tmp/coco/images/val2017"


def preprocess(path, imgsz=640):
    img = cv2.imread(path)
    img = cv2.resize(img, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
    return img[None]


class Reader(CalibrationDataReader):
    def __init__(self, imgs, input_name, imgsz):
        self.input_name, self.imgsz = input_name, imgsz
        self.it = iter(imgs)

    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.input_name: preprocess(p, self.imgsz)}


def make_qdq(fp32_onnx, qdq_onnx, n_calib, imgsz):
    iname = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    prepped = fp32_onnx.replace(".onnx", "_pp.onnx")
    quant_pre_process(fp32_onnx, prepped, skip_symbolic_shape=True)
    imgs = sorted(glob.glob(VAL_IMGS + "/*.jpg"))[:n_calib]
    quantize_static(prepped, qdq_onnx, Reader(imgs, iname, imgsz),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    return qdq_onnx


def bench(path, providers, imgsz=640, warmup=15, iters=80):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.mean(ts)), 3)


def trt(int8):
    opts = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": OUT}
    if int8:
        opts["trt_int8_enable"] = True
    return [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider", "CPUExecutionProvider"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x")
    ap.add_argument("--sparsity", type=int, default=50)
    ap.add_argument("--div", type=int, default=2)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    default_onnx = os.path.join(OUT, f"yolo26{args.tag}_default.onnx")
    pruned_onnx = os.path.join(OUT, f"yolo26{args.tag}_pruned_s{args.sparsity}_d{args.div}.onnx")
    res = {"tag": args.tag, "sparsity": args.sparsity}
    for label, fp32 in [("default", default_onnx), ("pruned", pruned_onnx)]:
        qdq = fp32.replace(".onnx", "_int8qdq.onnx")
        make_qdq(fp32, qdq, args.calib, args.imgsz)
        res[label] = {
            "fp32_MB": round(os.path.getsize(fp32) / 1e6, 1),
            "int8_MB": round(os.path.getsize(qdq) / 1e6, 1),
            "trt_fp16_ms": None, "trt_int8_ms": None,
        }
        try:
            res[label]["trt_fp16_ms"] = bench(fp32, trt(False), args.imgsz)
        except Exception as e:
            res[label]["trt_fp16_ms"] = f"err:{str(e)[:80]}"
        try:
            res[label]["trt_int8_ms"] = bench(qdq, trt(True), args.imgsz)
        except Exception as e:
            res[label]["trt_int8_ms"] = f"err:{str(e)[:80]}"
        print("QDQ", json.dumps({label: res[label]}))
    json.dump(res, open(os.path.join(OUT, f"yolo26{args.tag}_qdq_profile.json"), "w"), indent=2)
    print("QDQ_DONE", json.dumps(res))


if __name__ == "__main__":
    main()
