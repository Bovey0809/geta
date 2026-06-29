import torch, torchvision, time
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from ofa.model_zoo import ofa_net
try:
    from ofa.utils import count_net_flops
except Exception:
    count_net_flops = None

ROOT = "/root/autodl-tmp/imagenet"
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
print("loading ImageNet val (first run extracts the tar)...", flush=True)
ds = torchvision.datasets.ImageNet(ROOT, split="val", transform=tf)
print("val size", len(ds), flush=True)
# disjoint: BN-recal on [0:2000], eval on [2000:9000]
recal = DataLoader(Subset(ds, list(range(0, 2000))), batch_size=100, shuffle=False, num_workers=8)
evl = DataLoader(Subset(ds, list(range(2000, 9000))), batch_size=100, shuffle=False, num_workers=8)

ofa = ofa_net("ofa_mbv3_d234_e346_k357_w1.0", pretrained=True).cuda()

def subnet(kind):
    if kind == "MAX": ofa.set_active_subnet(ks=7, e=6, d=4)
    elif kind == "MIN": ofa.set_active_subnet(ks=3, e=3, d=2)
    else: ofa.sample_active_subnet()
    return ofa.get_active_subnet(preserve_weight=True).cuda()

def bn_recal(net, n=20):
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None
    net.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(recal):
            net(x.cuda())
            if i + 1 >= n: break
    net.eval()

def top1(net):
    net.eval(); c = t = 0
    with torch.no_grad():
        for x, y in evl:
            p = net(x.cuda()).argmax(1).cpu()
            c += (p == y).sum().item(); t += len(y)
    return 100.0 * c / t

torch.manual_seed(0)
for kind in ["MAX", "RAND1", "RAND2", "MIN"]:
    net = subnet(kind)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    flops = (count_net_flops(net, (1, 3, 224, 224)) / 1e6) if count_net_flops else -1
    bn_recal(net)
    acc = top1(net)
    print(f"OFA {kind:6s} params={params:.2f}M flops={flops:.0f}M top1={acc:.2f}%", flush=True)
    del net; torch.cuda.empty_cache()
print("OFA_DONE", flush=True)
