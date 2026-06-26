import torch, sys, os, io, contextlib, re, importlib
_orig = torch.load
def _load(*a, **k):
    k.setdefault("weights_only", False); return _orig(*a, **k)
torch.load = _load
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.makedirs("./cache", exist_ok=True)
from only_train_once import OTO
OTO.visualize = lambda self, *a, **k: None
t = sys.argv[1]
buf = io.StringIO(); status = "PASS"
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mod = importlib.import_module(t)
        cls = next(getattr(mod, n) for n in dir(mod)
                   if n.startswith("Test") and isinstance(getattr(mod, n), type))
        cls().test_sanity()
except AssertionError:
    status = "FAIL_DIFF"
except Exception as e:
    status = f"ERR:{type(e).__name__}:{str(e)[:55]}"
out = buf.getvalue()
diff = re.search(r"Maximum output difference\s+([0-9.eE+-]+)", out)
sizes = re.findall(r"Size of (?:full|compress) model[^\n]*?([0-9.eE+-]+)\s*GB", out)
red = ""
if len(sizes) >= 2 and float(sizes[0]) > 0:
    red = f"size -{100*(1-float(sizes[1])/float(sizes[0])):.0f}%"
print(f"SANITY {t:28s} {status:16s} diff={diff.group(1) if diff else '?'} {red}", flush=True)
