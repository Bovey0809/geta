import torch, os, unittest
from only_train_once import OTO
from ultralytics import YOLO

OUT_DIR = "./cache"

# YOLO26n detection head = model.23 (Detect). Unlike v8 there is NO dfl (DFL removed,
# box reg outputs 4 channels directly), and there are extra NMS-free one2one branches.
# The final 1x1 output convs (".2.weight", out-channels fixed at 4 box / 80 class)
# must stay unprunable. Discovered via experiments/geta_yolo26/discover_head.py.
YOLO26_UNPRUNABLE = [
    "model.23.cv2.0.2.weight", "model.23.cv2.1.2.weight", "model.23.cv2.2.2.weight",
    "model.23.cv3.0.2.weight", "model.23.cv3.1.2.weight", "model.23.cv3.2.2.weight",
    "model.23.one2one_cv2.0.2.weight", "model.23.one2one_cv2.1.2.weight", "model.23.one2one_cv2.2.2.weight",
    "model.23.one2one_cv3.0.2.weight", "model.23.one2one_cv3.1.2.weight", "model.23.one2one_cv3.2.2.weight",
]


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
        oto.mark_unprunable_by_param_names(YOLO26_UNPRUNABLE)
        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full = torch.load(oto.full_group_sparse_model_path)
        compressed = torch.load(oto.compressed_model_path)
        with torch.no_grad():
            full_out = full(dummy_input)
            comp_out = compressed(dummy_input)
        diff = _max_diff(full_out, comp_out)
        print("Maximum output difference", diff)
        self.assertLessEqual(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
