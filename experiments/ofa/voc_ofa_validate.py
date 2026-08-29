"""Does our OFA pipeline work AT ALL? Settled on VOC, where redundancy exists.

THE QUESTION THIS ANSWERS
-------------------------
Every OFA result in this study is negative, and all of them are on COCO. That
is consistent with two very different causes:
  (a) the pipeline is broken, or
  (b) yolo26-on-COCO has no redundant capacity to exploit.
187 correctness tests prove the *slicing* is exact. They say nothing about
whether sandwich training can ever produce a usable sub-net.

VOC discriminates. 16.5k images / 20 classes leaves yolo26s heavily
OVER-parameterised, so redundancy certainly exists. It is also cheap enough to
run TRUE OFA -- a supernet trained as a supernet from random init, sandwich-
sampled throughout. Every elastic failure so far was *post-hoc* elasticity on an
already-converged checkpoint, which is not what OFA prescribes.

THE MEASUREMENT: "OFA TAX"
--------------------------
    tax(w) = mAP(model of width w trained ALONE) - mAP(supernet's width-w sub-net)

That is the actual OFA promise: one training run, many deployable points, each
close to its individually-trained twin. A small tax means the machinery works.
A large tax on VOC -- where capacity is not the constraint -- means the approach
itself is at fault, and no amount of COCO compute would have rescued it.

Both arms get identical epochs, data, recipe and image size. Sub-nets are
BN-recalibrated per width before evaluation, which this study established is
mandatory (skipping it is what produced two retracted conclusions).

Usage:
  # gold baselines, one model per width, trained independently
  python experiments/ofa/voc_ofa_validate.py baselines --widths 0.50 0.375 0.25 --epochs 100
  # one supernet over the same widths (0.50 == yolo26s, 0.25 == yolo26n)
  python experiments/ofa/voc_ofa_validate.py supernet  --widths 0.50 0.375 0.25 --epochs 100
  # compare
  python experiments/ofa/voc_ofa_validate.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import pathlib
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from channel_plan import (  # noqa: E402
    bn_stats_coverage,
    count_active_params,
    disable_fuse,
    install_elastic_conv,
    recalibrate,
    set_width,
)
from elastic_attn import install_elastic_attention  # noqa: E402
from plan_builder import plan_model  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

RESULTS = Path("/root/voc_ofa_results.json")  # overridable via --results

# Modest, standard-ish recipe. Identical for BOTH arms -- the comparison is
# between arms, so absolute values matter less than using one recipe throughout.
RECIPE = dict(
    optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, box=7.5, cls=0.5, dfl=1.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0, scale=0.5, translate=0.1,
    fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, close_mosaic=10,
    imgsz=640, amp=True, val=False, plots=False,
)


# Keys in a released checkpoint's train_args that describe THAT run rather than
# the recipe, or that we set ourselves. Everything else is passed through, so the
# recipe cannot silently drift from what the released model actually used.
_NOT_RECIPE = {
    "model", "data", "name", "project", "save_dir", "exist_ok", "device",
    "resume", "workers", "val", "plots", "save_json", "save_period", "task",
    "mode", "verbose", "seed", "deterministic", "time", "patience", "fraction",
    "pretrained", "freeze", "cfg", "source", "distill_model",
}


def official_recipe(ckpt_path: str) -> dict:
    """Lift the COCO-stage recipe verbatim out of a released yolo26 checkpoint.

    The released models are trained Objects365 -> COCO; `train_args` records the
    COCO stage. Hand-copying it is how recipes drift, so read it instead. Returns
    a dict ready to splat into `.train()`, including its own epochs/batch --
    those ARE part of the official recipe and must not be overridden casually.
    """
    import torch as _t
    ck = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = dict(ck.get("train_args") or {})
    if not ta:
        raise SystemExit(f"{ckpt_path} has no train_args -- cannot derive recipe")
    rec = {k: v for k, v in ta.items() if k not in _NOT_RECIPE and v is not None}
    rec.update(val=False, plots=False)          # we val separately, per-width
    print(f"[recipe] lifted from {ckpt_path}: "
          f"optimizer={rec.get('optimizer')} epochs={rec.get('epochs')} "
          f"batch={rec.get('batch')} lr0={rec.get('lr0')} lrf={rec.get('lrf')} "
          f"nbs={rec.get('nbs')} close_mosaic={rec.get('close_mosaic')}", flush=True)
    return rec


def apply_init(y, init: str) -> None:
    """Transfer a pretrained checkpoint into a freshly-built model.

    Used to start the supernet from the public Objects365 checkpoint, which is
    what the official pipeline does. The checkpoint has nc=365 and our cfg has
    nc=80, so ultralytics' shape-intersecting load drops the classification head
    and keeps everything else -- exactly the official 365->80 head reinit. We
    verify the transfer actually happened rather than trusting it: a silent
    no-match here would look like a from-scratch run wearing an official label.
    """
    from ultralytics.utils.torch_utils import intersect_dicts
    import torch as _t
    before = {k: v.clone() for k, v in y.model.state_dict().items()}
    ck = _t.load(init, map_location="cpu", weights_only=False)
    src = (ck["model"] if isinstance(ck, dict) else ck).float().state_dict()
    inter = intersect_dicts(src, y.model.state_dict())
    y.model.load_state_dict(inter, strict=False)
    changed = sum(1 for k, v in y.model.state_dict().items()
                  if k in before and not _t.equal(before[k], v))
    total = len(y.model.state_dict())
    print(f"[init] {init}: {len(inter)}/{len(src)} src tensors matched by "
          f"name+shape, {changed}/{total} model tensors changed", flush=True)
    if changed == 0:
        raise SystemExit("ABORT: init transferred NOTHING -- this would be a "
                         "from-scratch run mislabelled as official-pipeline")
    skipped = [k for k in src if k not in inter]
    print(f"[init] skipped {len(skipped)} (expect head cls layers, nc 365->80): "
          f"{skipped[:6]}{' ...' if len(skipped) > 6 else ''}", flush=True)


def width_cfg(width: float) -> str:
    """Write a yolo26 yaml whose 's' scale carries the requested width.

    NOTE: `width` is the ABSOLUTE yolo26 width multiplier, not a fraction of s.
    With max_channels 1024 (what n and s use):
        0.50  == yolo26s  (10.0M)
        0.375 == the intermediate size (5.66M)
        0.25  == yolo26n  (2.57M)
    Passing 1.0 here builds a ~39.8M model, not s -- an easy and expensive
    mistake, caught by a 1-epoch smoke.

    ultralytics only parses scale letters [nslmx], and rewrites
    yolo26<letter>-<suffix>.yaml -> yolo26-<suffix>.yaml to find the shared
    architecture file, so we hijack 's'.
    """
    import yaml as _y
    base = _y.safe_load(open(_HERE / "yolo26-w375.yaml"))
    base["scales"] = dict(base["scales"])
    base["scales"]["s"] = [0.50, float(width), 1024]
    tag = f"w{int(round(width * 1000)):04d}"
    out_dir = Path("/root/voc_cfgs")
    out_dir.mkdir(exist_ok=True)
    shared = out_dir / f"yolo26-{tag}.yaml"
    _y.safe_dump(base, open(shared, "w"), sort_keys=False)
    handle = out_dir / f"yolo26s-{tag}.yaml"
    _y.safe_dump(base, open(handle, "w"), sort_keys=False)
    return str(handle)


def load(path=None) -> dict:
    """Read results. NOTE: resolves RESULTS at CALL time, not def time.

    Using `path=RESULTS` as a default binds the module-level value when the
    function is DEFINED, so a later `RESULTS = Path(args.results)` in main()
    silently has no effect and everything writes to the original filename.
    That is exactly what happened on the first COCO run.
    """
    p = pathlib.Path(path) if path else RESULTS
    return json.loads(p.read_text()) if p.exists() else {}


def save(d: dict, path=None) -> None:
    p = pathlib.Path(path) if path else RESULTS
    p.write_text(json.dumps(d, indent=2))


def calib_batches(data: str, batch: int, n: int):
    """Re-iterable stream of TRAIN images for BN recalibration."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(data)
    cfg = get_cfg()
    cfg.imgsz, cfg.rect = 640, False
    ds = build_yolo_dataset(cfg, d["train"], batch, d, mode="val", stride=32)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[: n * batch].tolist()

    class Stream:
        def __len__(self):
            return n

        def __iter__(self):
            dl = torch.utils.data.DataLoader(
                torch.utils.data.Subset(ds, idx), batch_size=batch, shuffle=False,
                num_workers=8, collate_fn=getattr(ds, "collate_fn", None),
                drop_last=True)
            for b in dl:
                yield b["img"].float() / 255.0
    return Stream()


def resolve_recipe(a):
    """Return (train_kwargs, epochs, batch) for this run.

    With --recipe-from, epochs and batch come from the released checkpoint's own
    train_args unless explicitly overridden on the command line, because they are
    part of the recipe -- silently keeping our --epochs default of 100 against an
    official 70-epoch schedule would make the run neither one thing nor the other.
    """
    if not a.recipe_from:
        return dict(RECIPE), a.epochs, a.batch
    rec = official_recipe(a.recipe_from)
    ep = rec.pop("epochs", a.epochs)
    bs = rec.pop("batch", a.batch)
    if a.epochs_override is not None:
        print(f"[recipe] epochs {ep} -> {a.epochs_override} (explicit override)", flush=True)
        ep = a.epochs_override
    if a.batch_override is not None:
        print(f"[recipe] batch {bs} -> {a.batch_override} (explicit override)", flush=True)
        bs = a.batch_override
    return rec, ep, bs


def cmd_released(a):
    """Evaluate the RELEASED checkpoints as baselines, under our own val call.

    For the official-pipeline experiment the comparison targets are the shipped
    yolo26s/yolo26n, not anything we train. Re-measuring them here rather than
    quoting published numbers keeps imgsz, batch, device and val code identical
    to the sub-net evaluation -- otherwise the tax absorbs a protocol difference.
    """
    res = load()
    res.setdefault("released", {})
    for spec in a.released:
        w, path = spec.split("=", 1)
        print(f"\n=== released baseline {path} (abs width {w}) ===", flush=True)
        y = YOLO(path)
        params = sum(p.numel() for p in y.model.parameters()) / 1e6
        m = y.val(data=a.data, imgsz=640, batch=a.batch, device=a.device,
                  plots=False, verbose=False)
        res["released"][str(float(w))] = {
            "ckpt": path, "params_M": params,
            "map5095": float(m.box.map), "map50": float(m.box.map50)}
        print(f"  {path}: {params:.3f}M  mAP50-95={float(m.box.map):.4f} "
              f"mAP50={float(m.box.map50):.4f}", flush=True)
        save(res)
        del y
        torch.cuda.empty_cache()
    return 0


def cmd_baselines(a):
    res = load()
    res.setdefault("baselines", {})
    for w in a.widths:
        cfg = width_cfg(w)
        name = f"{a.tag}_base_w{w}"
        print(f"\n=== baseline width={w} ({cfg}) ===", flush=True)
        y = YOLO(cfg)
        if a.init:
            apply_init(y, a.init)
        params = sum(p.numel() for p in y.model.parameters()) / 1e6
        rec, ep, bs = resolve_recipe(a)
        y.train(data=a.data, epochs=ep, batch=bs, name=name,
                device=a.device, **rec)
        m = y.val(data=a.data, imgsz=640, batch=a.batch, device=a.device,
                  plots=False, verbose=False)
        res["baselines"][str(w)] = {"params_M": params,
                                    "map5095": float(m.box.map),
                                    "map50": float(m.box.map50)}
        print(f"  width={w}: {params:.3f}M  mAP50-95={float(m.box.map):.4f} "
              f"mAP50={float(m.box.map50):.4f}", flush=True)
        save(res)
        del y
        torch.cuda.empty_cache()
    return 0


def patch_sandwich(trainer, widths, kd_lambda):
    """Sandwich loss: max width takes the real loss and feeds KD to the rest."""
    m = trainer.model
    plan_model(m)
    if getattr(trainer, "ema", None) is not None:
        trainer.ema.enabled = False
    cap = {}
    m.model[-1].register_forward_pre_hook(
        lambda _mod, args: cap.__setitem__("f", args[0]))
    orig = m.loss
    ctr = {"n": 0}

    def sloss(batch, preds=None):
        set_width(m, widths[0])
        l_max, items = orig(batch) if preds is None else orig(batch, preds)
        tfeat = [f.detach() for f in cap["f"]]
        total = l_max
        for w in widths[1:]:
            set_width(m, w)
            l_s, _ = orig(batch) if preds is None else orig(batch, preds)
            kd = 0
            for s, t in zip(cap["f"], tfeat):
                kd = kd + F.mse_loss(s, t[:, : s.shape[1]])
            total = total + l_s + kd_lambda * kd
        set_width(m, widths[0])
        ctr["n"] += 1
        if ctr["n"] <= 2 or ctr["n"] % 500 == 0:
            print(f"[sw {ctr['n']}] l_max={l_max.sum().item():.3f} "
                  f"l_small={l_s.sum().item():.3f} kd={float(kd):.4f}", flush=True)
        return total, items

    m.loss = sloss
    print(f"[sandwich] widths={widths} kd={kd_lambda}", flush=True)


def cmd_supernet(a):
    """Train ONE supernet spanning the same ABSOLUTE widths as the baselines.

    CRITICAL AND EASY TO GET WRONG: `width_cfg(w)` builds an architecture whose
    absolute yolo26 width multiplier is w, but `set_width(model, f)` applies an
    elastic FRACTION of that architecture's own width. They are different
    quantities. The supernet is built at the LARGEST absolute width, so to reach
    absolute width w its fraction must be w / max(widths):

        widths [0.50, 0.375, 0.25]  ->  fractions [1.0, 0.75, 0.5]

    Passing the absolute widths straight to set_width (as an earlier version
    did) builds sub-nets at absolute 0.25/0.1875/0.125 -- roughly 3.5x smaller
    than the baselines they are compared against -- AND makes the sandwich's
    "max" arm fraction 0.50, so the full supernet is never trained and the KD
    teacher is itself a sub-net. The whole comparison is then meaningless.
    """
    res = load()
    sup_w = max(a.widths)
    cfg = width_cfg(sup_w)
    fracs = [w / sup_w for w in a.widths]
    name = f"{a.tag}_supernet"
    print(f"\n=== supernet at absolute width {sup_w} ({cfg}) ===")
    print(f"    absolute widths {a.widths} -> elastic fractions {fracs}", flush=True)
    y = YOLO(cfg)
    if a.init:
        apply_init(y, a.init)
    y.add_callback("on_train_start",
                   lambda tr: patch_sandwich(tr, fracs, a.kd))
    y.add_callback("on_train_epoch_end",
                   lambda tr: tr.ema.ema.load_state_dict(tr.model.state_dict())
                   if getattr(tr, "ema", None) is not None else None)
    rec, ep, bs = resolve_recipe(a)
    y.train(data=a.data, epochs=ep, batch=bs, name=name,
            device=a.device, **rec)

    ckpt = sorted(Path("/root/runs/detect").glob(f"{name}*/weights/last.pt"),
                  key=lambda p: p.stat().st_mtime)[-1]
    print(f"supernet ckpt: {ckpt}", flush=True)

    calib = calib_batches(a.data, a.batch, 100)
    res.setdefault("supernet", {})
    for w, f in zip(a.widths, fracs):
        yq = YOLO(str(ckpt))
        mm = yq.model.to(f"cuda:{a.device}")
        plan_model(mm)
        disable_fuse(mm)
        recalibrate(mm, calib, f, device=torch.device(f"cuda:{a.device}"))
        have, need = bn_stats_coverage(mm, f)
        set_width(mm, f)
        r = yq.val(data=a.data, imgsz=640, batch=a.batch, device=a.device,
                   plots=False, verbose=False)
        res["supernet"][str(w)] = {
            "fraction": f,
            "params_M": count_active_params(mm, f) / 1e6,
            "map5095": float(r.box.map), "map50": float(r.box.map50),
            "bn_stats": f"{have}/{need}"}
        print(f"  supernet abs_w={w} (frac {f:.3f}): "
              f"{count_active_params(mm, f)/1e6:.3f}M "
              f"mAP50-95={float(r.box.map):.4f}", flush=True)
        save(res)
        del yq, mm
        torch.cuda.empty_cache()
    return 0


def cmd_report(a):
    res = load()
    b, s = res.get("baselines", {}), res.get("supernet", {})
    if not b or not s:
        print("need both arms; run `baselines` and `supernet` first")
        return 1
    print("\n" + "=" * 72)
    print("OFA TAX on VOC  (trained-alone minus supernet-subnet)")
    print("=" * 72)
    print(f"{'width':>7} {'alone(M)':>9} {'super(M)':>9} {'alone':>9} "
          f"{'supernet':>9} {'tax':>9}")
    taxes, mismatched, oversize, undersize = [], [], [], []
    for w in sorted(b, key=float, reverse=True):
        if w not in s:
            continue
        pa, ps = b[w]["params_M"], s[w]["params_M"]
        ba, su = b[w]["map5095"], s[w]["map5095"]
        tax = ba - su
        # A LARGE gap means the width bookkeeping is broken (the first VOC run
        # had ~3.5x) and the tax is not a tax at all. A small gap is expected
        # and benign: the Detect head does not shrink -- its output dims are
        # fixed and its interior is unplanned -- so the supernet carries a
        # full-width head at every fraction, which at low widths is a few
        # hundred kB of fixed overhead. That makes the sub-net LARGER than its
        # baseline, so a positive tax measured against it is CONSERVATIVE.
        gap = (ps - pa) / max(pa, 1e-9)
        if abs(gap) > 0.25:
            mismatched.append((w, pa, ps))
        elif gap > 0.02:
            oversize.append((w, gap))
        elif gap < -0.02:
            undersize.append((w, gap))
        taxes.append((float(w), tax))
        print(f"{w:>7} {pa:>8.3f}M {ps:>8.3f}M {ba:>9.4f} {su:>9.4f} {tax:>+9.4f}")
    if mismatched:
        print("\nABORT: arms differ by >25% in parameters -- that is broken width")
        print("bookkeeping, not a tax (the first VOC run was off by ~3.5x).")
        for w, pa, ps in mismatched:
            print(f"  width {w}: baseline {pa:.3f}M vs supernet {ps:.3f}M")
        print("Check absolute-width vs elastic-fraction handling.")
        return 1
    if oversize:
        print("\nNote: the supernet sub-net is LARGER than its baseline at "
              + ", ".join(f"w={w} (+{g*100:.1f}%)" for w, g in oversize))
        print("  Cause: the Detect head does not shrink. The tax below is "
              "therefore CONSERVATIVE\n  -- a size-matched sub-net would score "
              "no better, so the true tax is at least this large.")
    if undersize:
        print("\nWARNING: the supernet sub-net is SMALLER than its baseline at "
              + ", ".join(f"w={w} ({g*100:.1f}%)" for w, g in undersize)
              + " -- the tax is FLATTERED and should not be read as an upper bound.")
    if taxes:
        worst = max(t for _, t in taxes)
        print(f"\nworst tax: {worst:+.4f}")
        if worst < 0.02:
            print("VERDICT: pipeline VALIDATED. Sub-nets land within 0.02 of their\n"
                  "individually-trained twins, so the machinery does deliver the OFA\n"
                  "promise. The COCO failure is therefore about yolo26/COCO having no\n"
                  "redundant capacity, not about broken code.")
        elif worst < 0.05:
            print("VERDICT: partial. The machinery works but pays a real cost; the COCO\n"
                  "result is then a mix of that cost and genuine lack of redundancy.")
        else:
            print("VERDICT: pipeline does NOT deliver even where redundancy certainly\n"
                  "exists. The approach itself is at fault -- no amount of COCO compute\n"
                  "would have rescued it, and the earlier negatives are not evidence\n"
                  "about yolo26's capacity.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "supernet", "released", "report"])
    ap.add_argument("--data", default="/root/autodl-tmp/VOC/VOC.yaml")
    ap.add_argument("--widths", type=float, nargs="+", default=[0.50, 0.375, 0.25],
                    help="ABSOLUTE yolo26 width multipliers: 0.50=s, 0.25=n")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--results", default=None,
                    help="results json path; lets the two arms run on separate "
                         "boxes and be merged afterwards")
    ap.add_argument("--device", default="0")
    ap.add_argument("--init", default=None,
                    help="pretrained checkpoint to transfer into the freshly-built "
                         "model (e.g. the public Objects365 yolo26s), matched by "
                         "name+shape so a differing nc reinitialises the head")
    ap.add_argument("--recipe-from", default=None,
                    help="released .pt whose train_args ARE the recipe; brings its "
                         "own epochs/batch unless --epochs-override/--batch-override")
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--batch-override", type=int, default=None)
    ap.add_argument("--released", nargs="+", default=[],
                    help="ABSWIDTH=PATH pairs to evaluate as released baselines, "
                         "e.g. 0.50=/root/yolo26s.pt 0.25=/root/yolo26n.pt")
    ap.add_argument("--tag", default="voc",
                    help="run-name prefix, keeps separate experiments from colliding")
    ap.add_argument("--done-sentinel", default=None,
                    help="file to touch on success; watch THIS, not a log filename "
                         "(a notifier once waited on a log path that never existed)")
    a = ap.parse_args()

    install_elastic_conv()
    install_elastic_attention()
    if a.results:
        global RESULTS
        RESULTS = Path(a.results)
    a.widths = sorted(a.widths, reverse=True)
    rc = {"baselines": cmd_baselines, "supernet": cmd_supernet,
          "released": cmd_released, "report": cmd_report}[a.cmd](a)
    if rc == 0 and a.done_sentinel:
        pathlib.Path(a.done_sentinel).write_text(f"{a.cmd} ok\n")
        print(f"[done] wrote sentinel {a.done_sentinel}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
