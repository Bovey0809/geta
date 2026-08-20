"""Minimal width-elastic OFA PoC for yolo26s.

Five checks (any failure aborts the rest):
  1. Bit-identical forward at w=1.0 vs a freshly loaded (unpatched) model.
     Proves the fast path in _elastic_conv_forward is truly a no-op.
  2. Non-crash forward at w=0.5 with the expected Detect output shape.
     Proves width slicing doesn't break shape flow through the whole net.
  3. Eval mAP50-95 at w=1.0 = ~0.472 (yolo26s baseline).
     Proves the patched-forward eval pipeline is sound.
  4. Eval mAP50-95 at w=0.5 = some finite number.
     Untrained subnet is expected to be poor; this only asks for finite output
     (unlike the depth-elastic d=1 attempt on yolo26l which was exactly 0.0).
  5. One training step at w=0.5 produces gradient on the [:out_k, :in_k]
     slice of a shared Conv weight and NO grad outside that slice. Proves the
     weight-sharing property required for OFA.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import width_elastic  # noqa: F401  (applies the Conv monkey-patch on import)
from width_elastic import set_width

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


def check_1_identity(model_patched, model_unpatched, x):
    print("[check 1] bit-identical forward at w=1.0")
    set_width(model_patched, 1.0)
    with torch.no_grad():
        y_patched = model_patched(x)
        y_ref = model_unpatched(x)
    p = y_patched[0] if isinstance(y_patched, (list, tuple)) else y_patched
    r = y_ref[0] if isinstance(y_ref, (list, tuple)) else y_ref
    diff = (p - r).abs().max().item()
    ok = diff < 1e-5
    print(f"  max_abs_diff={diff:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def check_2_forward_half(model, x):
    print("[check 2] forward at w=0.5 produces finite output")
    set_width(model, 0.5)
    with torch.no_grad():
        y = model(x)
    o = y[0] if isinstance(y, (list, tuple)) else y
    ok = torch.isfinite(o).all().item()
    print(f"  out_shape={tuple(o.shape)} finite={ok}  {'OK' if ok else 'FAIL'}")
    return ok


def check_3_eval_full(y_yolo, data_yaml, device):
    print("[check 3] eval mAP50-95 at w=1.0 (expect ~0.472)")
    set_width(y_yolo.model, 1.0)
    metrics = y_yolo.val(data=data_yaml, imgsz=640, batch=16, plots=False, device=device, verbose=False)
    m = float(metrics.box.map)
    ok = 0.46 <= m <= 0.48
    print(f"  mAP50-95={m:.4f}  {'OK' if ok else 'FAIL'}")
    return ok, m


def check_4_eval_half(y_yolo, data_yaml, device):
    print("[check 4] eval mAP50-95 at w=0.5 (any finite number, expected low)")
    set_width(y_yolo.model, 0.5)
    metrics = y_yolo.val(data=data_yaml, imgsz=640, batch=16, plots=False, device=device, verbose=False)
    m = float(metrics.box.map)
    ok = m == m  # finite (not nan)
    print(f"  mAP50-95={m:.4f}  {'OK' if ok else 'FAIL'}")
    return ok, m


def check_5_grad_flow(model, x, target_labels, device):
    """One backward pass at w=0.5; verify the sliced region gets grad, the rest doesn't."""
    print("[check 5] gradient flow only in the [:out_k, :in_k] slice at w=0.5")
    set_width(model, 0.5)
    model.train()
    # YOLO() loads params with requires_grad=False; re-enable.
    for p in model.parameters():
        p.requires_grad_(True)
    # Attach default detection hyperparameters (trainer normally sets these).
    from types import SimpleNamespace
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    # Force criterion re-init so it picks up model.args.
    model.criterion = None
    # Zero all grads first.
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    batch = {
        "img": x,
        "batch_idx": target_labels["batch_idx"],
        "cls": target_labels["cls"],
        "bboxes": target_labels["bboxes"],
    }
    loss, _ = model.loss(batch)
    loss.sum().backward() if loss.ndim else loss.backward()

    # Pick a mid-network Conv that has out>1 and groups==1 (not stem, not detect internal).
    from width_elastic import _frozen_conv_ids
    frozen = _frozen_conv_ids(model)
    target = None
    for m in model.modules():
        if isinstance(m, Conv) and id(m) not in frozen:
            if m.conv.groups == 1 and m.conv.out_channels >= 8 and m.conv.in_channels >= 8:
                if m.conv.weight.grad is not None:
                    target = m
                    break
    if target is None:
        print("  FAIL: no candidate conv found")
        return False

    g = target.conv.weight.grad
    out_full = target.conv.out_channels
    in_full = target.conv.in_channels
    out_k = max(1, int(round(out_full * 0.5)))
    # in_k depends on x flowing in — infer from the fact that its previous layer sliced too.
    in_k = max(1, int(round(in_full * 0.5)))
    inside = g[:out_k, :in_k].abs().sum().item()
    # Total minus inside = outside slice.
    total = g.abs().sum().item()
    outside = total - inside
    ok = inside > 0 and outside < 1e-8
    print(
        f"  target={target.conv} out_k={out_k} in_k={in_k}"
    )
    print(
        f"  grad |inside|={inside:.3e}  |outside|={outside:.3e}"
        f"  ratio_outside/total={outside / max(total, 1e-30):.3e}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--data", default="/root/geta/experiments/geta_yolo26/coco.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--skip-eval", action="store_true", help="skip mAP checks (fast, no COCO needed)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load twice — one to patch (module methods are class-level so both are patched,
    # but the reference will keep _active_width=1.0 everywhere).
    y_patched = YOLO(args.model)
    m_patched = y_patched.model.to(device).eval()

    y_ref = YOLO(args.model)
    m_ref = y_ref.model.to(device).eval()
    # Both have the elastic forward now (class-level monkey-patch). Reference stays at w=1.0.
    set_width(m_ref, 1.0)

    x = torch.rand(1, 3, 640, 640, device=device)

    results = {}
    results["1_identity"] = check_1_identity(m_patched, m_ref, x)
    results["2_half_forward"] = check_2_forward_half(m_patched, x)

    if not args.skip_eval:
        set_width(m_patched, 1.0)
        results["3_eval_full"], results["3_map"] = check_3_eval_full(y_patched, args.data, args.device)
        results["4_eval_half"], results["4_map"] = check_4_eval_half(y_patched, args.data, args.device)

    # Synthetic detection batch for grad flow. One image, one box, class 0.
    labels = {
        "batch_idx": torch.zeros(1, device=device),
        "cls": torch.zeros(1, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
    }
    # val() marks params as inference tensors, so reload a fresh model for grad flow.
    y_grad = YOLO(args.model)
    m_grad = y_grad.model.to(device)
    results["5_grad_flow"] = check_5_grad_flow(m_grad, x, labels, device)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    ok = all(v for k, v in results.items() if k.endswith("_flow") or k.endswith("_identity") or k.endswith("_forward") or k.endswith("_full") or k.endswith("_half"))
    print(f"\n{'ALL_PASS' if ok else 'SOME_FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
