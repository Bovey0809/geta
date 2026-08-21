"""Free predictor for Gate B: how much importance mass does sorting retain?

Gate B costs GPU time; this costs nothing and needs no data. For each planned
conv, compare the importance mass kept by taking the first k channels
(what Gate A did) against the mass kept by taking the top k (what sorting
does), where importance is |gamma|/sqrt(running_var+eps).

    retention = sum(kept importance) / sum(all importance)

If unsorted retention is already ~= sorted retention, sorting cannot help and
Gate B is not worth running. A large gap is a necessary — though not
sufficient — condition for sorting to rescue accuracy: it says the channels
being discarded were carrying real weight.

Run: python experiments/ofa/tests/predict_gate_b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from channel_plan import _OUT_PLAN, install_elastic_conv  # noqa: E402
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_model  # noqa: E402
from sorter import importance  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.conv import Conv  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

install_elastic_conv()
install_elastic_attention()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="pretrained .pt; REQUIRED for a meaningful answer -- "
                         "freshly-initialised BN has gamma=1 and var=1 for every "
                         "channel, so first-k and top-k tie trivially")
    a = ap.parse_args()

    torch.manual_seed(0)
    if a.weights:
        model = YOLO(a.weights).model.eval()
        print(f"weights: {a.weights} (pretrained)")
    else:
        model = DetectionModel("yolo26s.yaml", ch=3, nc=80, verbose=False).eval()
        print("weights: RANDOM yaml init -- expect a zero gap, which is an "
              "artifact, not a result")
    plan_model(model)

    convs = [m for m in model.modules()
             if isinstance(m, Conv) and hasattr(m, _OUT_PLAN)]
    print(f"planned convs: {len(convs)}\n")
    print(f"{'w':>7} {'first-k':>9} {'top-k':>9} {'gap':>8} "
          f"{'worst-layer first-k':>20}")

    for w in (0.875, 0.75, 0.625, 0.5):
        tot = kept_first = kept_top = 0.0
        worst = (1.0, None)
        for c in convs:
            plan = getattr(c, _OUT_PLAN)
            score = importance(c).double()
            for off, g, el in zip(plan.offsets(), plan.groups, plan.elastic):
                if not el:
                    continue
                seg = score[off:off + g]
                k = max(1, int(round(g * w)))
                s_all = seg.sum().item()
                s_first = seg[:k].sum().item()
                s_top = torch.topk(seg, k).values.sum().item()
                tot += s_all
                kept_first += s_first
                kept_top += s_top
                if s_all > 0:
                    r = s_first / s_all
                    if r < worst[0]:
                        worst = (r, f"{tuple(c.conv.weight.shape[:2])}")
        rf, rt = kept_first / tot, kept_top / tot
        print(f"{w:>7.3f} {rf:>9.4f} {rt:>9.4f} {rt - rf:>+8.4f} "
              f"{worst[0]:>10.4f} {worst[1] or '':>9}")

    print("\nA large first-k vs top-k gap is NECESSARY (not sufficient) for "
          "sorting\nto rescue accuracy: it says the discarded channels were "
          "carrying real\nweight. A near-zero gap on pretrained weights would "
          "mean sorting cannot\nhelp and Gate B is not worth GPU time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
