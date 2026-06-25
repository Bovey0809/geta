import os, time, glob
import numpy as np, torch
import onnxruntime as ort
from ultralytics import YOLO
HERE=os.path.dirname(__file__); OUT=os.path.join(HERE,"out"); OPSET=17

def export(module, path, imgsz=640):
    module=module.eval().cpu()
    torch.onnx.export(module, torch.rand(1,3,imgsz,imgsz), path, opset_version=OPSET,
                      input_names=["images"], output_names=["output0"], do_constant_folding=True)

def trt():
    return [("TensorrtExecutionProvider", {"trt_fp16_enable":True,"trt_engine_cache_enable":True,
            "trt_engine_cache_path":OUT}), "CUDAExecutionProvider","CPUExecutionProvider"]

def bench(path, imgsz=640, warmup=15, iters=80):
    s=ort.InferenceSession(path, providers=trt()); ep=s.get_providers()[0]; nm=s.get_inputs()[0].name
    x=np.random.rand(1,3,imgsz,imgsz).astype(np.float32)
    for _ in range(warmup): s.run(None,{nm:x})
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); s.run(None,{nm:x}); ts.append((time.perf_counter()-t)*1000)
    return round(float(np.mean(ts)),3), ep

# default-m onnx (export if missing)
dft=os.path.join(OUT,"yolo26m_default.onnx")
if not os.path.exists(dft): export(YOLO("yolo26m.pt").model, dft)
# pruned-m demo
comp=torch.load("experiments/geta_yolo26/out/geta_m_s10_demo/DetectionModel_compressed.pt", weights_only=False)
prn=os.path.join(OUT,"yolo26m_demo_pruned.onnx"); export(comp, prn)
import sys
dp=sum(p.numel() for p in YOLO("yolo26m.pt").model.parameters())/1e6
pp=sum(p.numel() for p in comp.parameters())/1e6
for label,path,par in [("default",dft,dp),("pruned",prn,pp)]:
    ms,ep=bench(path)
    print("MSPEED %s params=%.2fM TRTfp16=%.2fms ep=%s size=%.1fMB"%(label,par,ms,ep,os.path.getsize(path)/1e6))
print("MSPEED_DONE")
