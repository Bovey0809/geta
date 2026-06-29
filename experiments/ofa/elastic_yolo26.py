import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k

def elastic_c3k_forward(self, x):
    d = getattr(self, "active_depth", len(self.m))
    y = self.cv1(x)
    for i, blk in enumerate(self.m):
        if i >= d:
            break
        y = blk(y)
    return self.cv3(torch.cat((y, self.cv2(x)), 1))

C3k.forward = elastic_c3k_forward

def set_depth(model, d):
    n = 0
    for mod in model.modules():
        if isinstance(mod, C3k):
            mod.active_depth = d; n += 1
    return n

if __name__ == "__main__":
    from thop import profile
    m = YOLO("yolo26l.pt").model.eval()
    x = torch.rand(1, 3, 640, 640)
    nc3k = set_depth(m, 2)
    print(f"elastic C3k blocks: {nc3k}")
    for d in [2, 1]:
        set_depth(m, d)
        with torch.no_grad():
            out = m(x)
        flops, _ = profile(m, inputs=(x,), verbose=False)
        oshape = out[0].shape if isinstance(out, (list, tuple)) else out.shape
        print(f"DEPTH d={d}: forward OK out={tuple(oshape)} GFLOPs={flops/1e9:.2f}")
    print("ELASTIC_TEST_DONE")
