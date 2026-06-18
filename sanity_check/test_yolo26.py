import torch, os, unittest
from only_train_once import OTO
from ultralytics import YOLO

OUT_DIR = "./cache"

# YOLO26n detection head = model.23 (Detect). Unlike v8 there is NO dfl (DFL removed,
# box reg outputs 4 channels directly), and there are extra NMS-free one2one branches.
# The final 1x1 output convs (".2.weight", out-channels fixed at 4 box / 80 class)
# must stay unprunable. Discovered via experiments/geta_yolo26/discover_head.py.
YOLO26_HEAD_UNPRUNABLE = [
    "model.23.cv2.0.2.weight", "model.23.cv2.1.2.weight", "model.23.cv2.2.2.weight",
    "model.23.cv3.0.2.weight", "model.23.cv3.1.2.weight", "model.23.cv3.2.2.weight",
    "model.23.one2one_cv2.0.2.weight", "model.23.one2one_cv2.1.2.weight", "model.23.one2one_cv2.2.2.weight",
    "model.23.one2one_cv3.0.2.weight", "model.23.one2one_cv3.1.2.weight", "model.23.one2one_cv3.2.2.weight",
]

# C2PSA attention blocks. The chunk/split fix makes their num_groups consistent, but
# pruning their cv1 split dim cascades into multi-head attention internals (qkv reshape,
# num_heads, key_dim) and Python attributes (self.c) that GETA's construct_subnet cannot
# rewire, so the pruned module's forward breaks. Exclude both blocks from pruning; the
# conv backbone/neck and all C2f/C3k2 blocks still prune.
C2PSA_BLOCK_PREFIXES = ("model.10.", "model.22.")


def yolo26_unprunable_names(model):
    """Full unprunable param-name list for GETA on any yolo26 size (n..x).

    Generalizes across model sizes by detecting structure instead of hardcoding
    indices:
      * Detection head = last module in model.model; its output convs are the
        leaf 1x1s ending in '.2.weight' (cv2/cv3/one2one_* -> 4 box / 80 class).
      * C2PSA attention blocks = any top-level block containing an '.attn.qkv.'
        param; exclude every weight in those blocks (see module-doc rationale).
    """
    head = f"model.{len(model.model) - 1}."
    c2psa_prefixes = set()
    for n, _ in model.named_parameters():
        if ".attn.qkv." in n:
            c2psa_prefixes.add(".".join(n.split(".")[:2]) + ".")  # e.g. 'model.10.'
    c2psa_prefixes = tuple(c2psa_prefixes)
    names = []
    for n, _ in model.named_parameters():
        if n.startswith(head) and n.endswith(".2.weight"):
            names.append(n)
        elif c2psa_prefixes and n.endswith(".weight") and n.startswith(c2psa_prefixes):
            names.append(n)
    return names


def _max_diff(a, b):
    """Max abs elementwise diff over all matching tensors in (possibly nested) outputs."""
    def flat(x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            return [t for e in x for t in flat(e)]
        if isinstance(x, dict):
            return [t for e in x.values() for t in flat(e)]
        return []
    m = 0.0
    for ta, tb in zip(flat(a), flat(b)):
        if ta.shape == tb.shape:
            m = max(m, torch.max(torch.abs(ta - tb)).item())
    return m


class TestYolo26(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 640, 640)):
        model = YOLO("yolo26n.pt").model
        for name, param in model.named_parameters():
            if "running_mean" not in name:
                param.requires_grad = True
        oto = OTO(model, dummy_input)
        oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        try:
            oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        except Exception as e:
            print(f"[visualize skipped: {e}]")  # graphviz is diagnostic, not part of the gate
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full = torch.load(oto.full_group_sparse_model_path, weights_only=False)
        compressed = torch.load(oto.compressed_model_path, weights_only=False)
        with torch.no_grad():
            full_out = full(dummy_input)
            comp_out = compressed(dummy_input)
        diff = _max_diff(full_out, comp_out)
        print("Maximum output difference", diff)
        self.assertLessEqual(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
