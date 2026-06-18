# GETA × YOLO26 Structured Pruning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Structurally prune `yolo26n` with GETA so it is smaller + faster while COCO mAP50-95 stays at parity, and produce an ONNX speed profile vs. the stock model.

**Architecture:** Develop locally → push to fork `Bovey0809/geta` → `git pull` + run on a single-RTX-3080 AutoDL box (user executes, agent hands off). GETA's dependency-graph tracer prunes the un-quantized `yolo26n` `nn.Module`; its sparsity-inducing optimizer is driven *inside* a subclassed Ultralytics `DetectionTrainer` (Approach A) so YOLO26's validated COCO mAP/data/loss pipeline is reused. A Phase-2 graph-trace gate (`construct_subnet` output diff < 1e-4) must pass before any COCO compute is spent.

**Tech Stack:** PyTorch ≥ 2.0, `only_train_once` (GETA/OTO v3, this repo), `ultralytics` (YOLO26-capable), ONNX + onnxruntime-gpu, AutoDL (CUDA), COCO from AutoDL public dataset mount.

## Global Constraints

- **Scope (phase 1): pruning only** — do NOT pass the model through `model_to_quantize_model`; quantization is out of scope.
- **Variant: `yolo26n` only.**
- **Single RTX 3080** (~10–12 GB): batch sizes small; schedules must fit one GPU.
- **User runs all server commands**; every server step is a copy-pasteable command with an expected-output check. Agent has no server shell.
- **Code sync via fork** `Bovey0809/geta` (origin `microsoft/geta` is not pushable). Server pulls from `fork`.
- **COCO** lives read-only in the AutoDL public dataset area (e.g. `/root/autodl-pub/...`); symlink it, do not re-download. COCO8/COCO128 via Ultralytics auto-download.
- **mAP parity target:** pruned mAP50-95 ≥ stock − 0.3 pts.
- **Branch:** `geta-yolo26-pruning`. Commit after each task; push to `fork`.
- **ONNX profiling protocol (fixed):** input 640×640, batch=1, opset pinned (set once in Task 10, document it), 20 warmup + 100 timed iters, report mean/p50/p95; run on CUDAExecutionProvider (3080) and CPUExecutionProvider.

---

### Task 0: Fork + remote sync setup

**Files:**
- Create: `scripts/server/README.md` (sync + run notes)

**Interfaces:**
- Produces: a pushable `fork` remote (`https://github.com/Bovey0809/geta`) and the documented edit→push→pull loop used by every later task.

- [ ] **Step 1: Create the fork (local, agent or user — needs `gh` auth as Bovey0809)**

Run:
```bash
gh repo fork microsoft/geta --clone=false --remote=false
```
Expected: `✓ Created fork Bovey0809/geta` (or "already exists" — fine).

- [ ] **Step 2: Add the fork remote and push the current branch**

Run:
```bash
cd /home/rick/geta
git remote add fork https://github.com/Bovey0809/geta.git 2>/dev/null || git remote set-url fork https://github.com/Bovey0809/geta.git
git push -u fork geta-yolo26-pruning
```
Expected: branch `geta-yolo26-pruning` appears on `Bovey0809/geta`; no 403.

- [ ] **Step 3: Write the sync/run notes**

Create `scripts/server/README.md`:
```markdown
# Server sync & run (AutoDL, single RTX 3080)

Dev loop: edit locally in /home/rick/geta -> commit -> `git push fork geta-yolo26-pruning`
On server: `git -C ~/geta pull fork geta-yolo26-pruning`

First-time server clone:
    cd ~ && git clone -b geta-yolo26-pruning https://github.com/Bovey0809/geta.git
    cd ~/geta

All experiment entrypoints live under experiments/geta_yolo26/ and scripts/server/.
Run order: setup_env.sh -> link_coco.sh -> baseline -> sanity (test_yolo26) -> smoke -> full -> profile.
```

- [ ] **Step 4: Verify the server can pull (user runs on AutoDL)**

Hand off to user:
```bash
cd ~ && git clone -b geta-yolo26-pruning https://github.com/Bovey0809/geta.git || (cd ~/geta && git pull fork geta-yolo26-pruning)
ls ~/geta/scripts/server/README.md
```
Expected: file path prints; repo present at `~/geta`.

- [ ] **Step 5: Commit**

```bash
git add scripts/server/README.md && git commit -m "chore: fork sync workflow + server run notes"
git push fork geta-yolo26-pruning
```

---

### Task 1: Server env bring-up script

**Files:**
- Create: `scripts/server/setup_env.sh`

**Interfaces:**
- Consumes: a fresh AutoDL instance with CUDA + base PyTorch.
- Produces: a Python env where `import torch; import ultralytics; import only_train_once; import onnxruntime` all succeed and `torch.cuda.is_available()` is True. Establishes that `ultralytics` ships YOLO26.

- [ ] **Step 1: Write `scripts/server/setup_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "== GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== Torch =="
python - <<'PY'
import torch
assert torch.__version__ >= "2", f"need torch>=2.0, got {torch.__version__}"
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "== Install deps (GETA + ultralytics + onnx tooling) =="
pip install -q "ultralytics>=8.3" onnx onnxruntime-gpu graphviz
# GETA package is used in-place from the repo via PYTHONPATH; install its imports' deps:
pip install -q numpy pillow

echo "== Import checks =="
python - <<'PY'
import torch, ultralytics, onnx, onnxruntime
import sys; sys.path.insert(0, ".")
import only_train_once
print("ultralytics", ultralytics.__version__)
print("onnxruntime providers", onnxruntime.get_available_providers())
# YOLO26 availability probe (does NOT download weights yet)
from ultralytics import YOLO
print("YOLO import OK")
PY
echo "ENV OK"
```

- [ ] **Step 2: Make executable + commit (local)**

```bash
chmod +x scripts/server/setup_env.sh
git add scripts/server/setup_env.sh && git commit -m "feat: AutoDL env bring-up script"
git push fork geta-yolo26-pruning
```

- [ ] **Step 3: Run on server (user) and confirm**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && bash scripts/server/setup_env.sh
```
Expected: `nvidia-smi` shows the 3080; torch ≥ 2.0 + `cuda_available True`; `ultralytics` version prints; `onnxruntime providers` includes `CUDAExecutionProvider`; final line `ENV OK`.
**If `pip install ultralytics>=8.3` does not expose YOLO26**, paste the version back — we pin the correct YOLO26 version before proceeding (blocks Task 3).

---

### Task 2: COCO dataset wiring + validation

**Files:**
- Create: `scripts/server/link_coco.sh`
- Create: `experiments/geta_yolo26/coco.yaml`

**Interfaces:**
- Consumes: COCO mounted read-only under the AutoDL public dataset path (discovered at runtime).
- Produces: an Ultralytics-resolvable `coco.yaml` pointing at `images/train2017`, `images/val2017`, `annotations/instances_val2017.json`, validated by the `ultralytics:dataset-check` skill.

- [ ] **Step 1: Write `scripts/server/link_coco.sh` (discovers + symlinks)**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Find COCO in the AutoDL public mount (path varies per instance).
echo "== Locating COCO under /root/autodl-pub =="
PUB=$(find /root/autodl-pub -maxdepth 3 -type d -iname "*coco*" 2>/dev/null | head -1 || true)
echo "Candidate COCO dir: ${PUB:-<none found>}"
test -n "${PUB}" || { echo "ERROR: set COCO path manually in this script"; exit 1; }

DST=~/datasets/coco
mkdir -p ~/datasets
ln -sfn "${PUB}" "${DST}"
echo "Linked ${DST} -> ${PUB}"
echo "== Layout =="
ls "${DST}" || true
find "${DST}" -maxdepth 2 -iname "instances_val2017.json" 2>/dev/null | head -1
find "${DST}" -maxdepth 2 -type d -name "val2017" 2>/dev/null | head -2
```

- [ ] **Step 2: Write `experiments/geta_yolo26/coco.yaml`**

```yaml
# Resolves against ~/datasets/coco (symlink created by scripts/server/link_coco.sh)
path: /root/datasets/coco
train: images/train2017
val: images/val2017
# Ultralytics standard 80-class COCO names are inherited from its packaged coco.yaml;
# if class names are needed explicitly, copy `names:` from ultralytics/cfg/datasets/coco.yaml.
```

- [ ] **Step 3: Commit + run on server (user)**

```bash
git add scripts/server/link_coco.sh experiments/geta_yolo26/coco.yaml
git commit -m "feat: COCO symlink + dataset yaml" && git push fork geta-yolo26-pruning
```
Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && bash scripts/server/link_coco.sh
```
Expected: prints a real COCO dir, an `instances_val2017.json`, and a `val2017` images dir. If the public layout differs (e.g. labels in YOLO `.txt` vs COCO json), paste `ls -R` of the top two levels back so we adjust `coco.yaml`.

- [ ] **Step 4: Validate the dataset yaml**

Use the **`ultralytics:dataset-check`** skill against `experiments/geta_yolo26/coco.yaml` (paths resolve, val split has images, labels/classes match). Fix `coco.yaml` until it passes. This task is done when dataset-check reports OK.

---

### Task 3: Stock baseline — lock the parity target

**Files:**
- Create: `experiments/geta_yolo26/baseline.py`

**Interfaces:**
- Consumes: env (Task 1), `coco.yaml` (Task 2).
- Produces: `experiments/geta_yolo26/out/baseline_metrics.json` with stock yolo26n `{map5095, map50, params_M, gflops}` — the numbers later tasks must match/beat.

- [ ] **Step 1: Write `experiments/geta_yolo26/baseline.py`**

```python
import json, os
from ultralytics import YOLO

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(os.path.dirname(__file__), "coco.yaml")

def main():
    model = YOLO("yolo26n.pt")  # downloads pretrained weights
    metrics = model.val(data=DATA, imgsz=640, batch=16, device=0)
    # n_params / GFLOPs from the underlying module
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    rec = {
        "model": "yolo26n",
        "map5095": float(metrics.box.map),     # mAP50-95
        "map50": float(metrics.box.map50),
        "params_M": round(n_params, 4),
    }
    with open(os.path.join(OUT, "baseline_metrics.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print("BASELINE", json.dumps(rec))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit + run on server (user)**

```bash
git add experiments/geta_yolo26/baseline.py && git commit -m "feat: stock yolo26n COCO baseline" && git push fork geta-yolo26-pruning
```
Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python experiments/geta_yolo26/baseline.py
```
Expected: a `BASELINE {...}` line with a non-zero `map5095` close to the published yolo26n number; `out/baseline_metrics.json` written. Paste the JSON back — it is the parity target recorded in the spec's §6 table.

---

### Task 4: GETA graph-trace gate for YOLO26 (GO/NO-GO)

**Files:**
- Create: `sanity_check/test_yolo26.py`
- Create: `experiments/geta_yolo26/discover_head.py`

**Interfaces:**
- Consumes: env (Task 1), `OTO`, `mark_unprunable_by_param_names`, `construct_subnet` (this repo, pattern from `sanity_check/test_yolov8.py`).
- Produces: the verified list `YOLO26_UNPRUNABLE` (detection-head output conv param names) reused by Task 6; a passing reconstruction test (output diff < 1e-4).

- [ ] **Step 1: Write `experiments/geta_yolo26/discover_head.py` (prints candidate unprunable params)**

```python
# Prints the detection head's final/output conv param names so we can mark them unprunable.
import torch
from ultralytics import YOLO

m = YOLO("yolo26n.pt").model
# The detect head is the last module in model.model; print its index + leaf conv weights.
head_idx = len(m.model) - 1
print("HEAD_INDEX", head_idx, "type", type(m.model[head_idx]).__name__)
for name, p in m.named_parameters():
    if name.startswith(f"model.{head_idx}.") and name.endswith(".weight") and p.dim() == 4:
        print("HEAD_CONV", name, tuple(p.shape))
```

- [ ] **Step 2: Run discovery on server (user), capture the head conv names**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python experiments/geta_yolo26/discover_head.py
```
Expected: `HEAD_INDEX <n> type <DetectHead-like>` plus a list of `HEAD_CONV model.<n>....weight (shape)`. **Paste this back.** The output-channel conv layers (those whose out-channels == num_classes or == reg dims, analogous to v8's `cv2.*.2`, `cv3.*.2`, and any `dfl`) become `YOLO26_UNPRUNABLE`.

- [ ] **Step 3: Write `sanity_check/test_yolo26.py` using the discovered names**

```python
import torch, os, unittest
from only_train_once import OTO
from ultralytics import YOLO

OUT_DIR = "./cache"

# Filled from experiments/geta_yolo26/discover_head.py output (Step 2).
# Replace <N> and the leaf names with the actual head output convs.
YOLO26_UNPRUNABLE = [
    # e.g. "model.<N>.cv2.0.2.weight", "model.<N>.cv3.0.2.weight", ...
]

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
        comp = torch.load(oto.compressed_model_path)
        fo, co = full(dummy_input), comp(dummy_input)
        diff = torch.max(torch.abs(fo[0] - co[0]))
        print("Maximum output difference", diff.item())
        self.assertLessEqual(diff, 1e-4)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the gate on server (user)**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python -m unittest sanity_check.test_yolo26 -v
```
Expected: prints `Maximum output difference <= 1e-4` and `OK`.
**This is the GO/NO-GO gate.** If tracing throws or diff > 1e-4: capture the traceback, inspect `cache/*.gv`/visualization, and iterate on `YOLO26_UNPRUNABLE` / `OTO(..., skip_patterns=...)`. If the NMS-free head is fundamentally untraceable, fall back to marking the whole head module unprunable (prune backbone+neck only) and record that as a finding before continuing.

- [ ] **Step 5: Commit the verified test + names**

```bash
git add sanity_check/test_yolo26.py experiments/geta_yolo26/discover_head.py
git commit -m "feat: GETA graph-trace gate for yolo26n (output diff < 1e-4)"
git push fork geta-yolo26-pruning
```

---

### Task 5: GETA-driven DetectionTrainer (Approach A)

**Files:**
- Create: `experiments/geta_yolo26/geta_trainer.py`

**Interfaces:**
- Consumes: `YOLO26_UNPRUNABLE` (Task 4), `oto.geta(...)` signature (`only_train_once/__init__.py:122`), Ultralytics `DetectionTrainer`.
- Produces: `GetaDetectionTrainer` class + `build_geta_trainer(overrides, sparsity, schedule)` factory; an `oto` handle accessible post-train for `construct_subnet`. Consumed by Tasks 6–7.

- [ ] **Step 1: Write `experiments/geta_yolo26/geta_trainer.py`**

```python
"""GETA sparsity optimizer driven inside Ultralytics DetectionTrainer (Approach A)."""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from only_train_once import OTO
from sanity_check.test_yolo26 import YOLO26_UNPRUNABLE


class GetaDetectionTrainer(DetectionTrainer):
    """Replaces the optimizer with oto.geta(...) and drives its step-based schedule.

    Extra overrides keys: geta_sparsity (float), geta_schedule (dict of step counts).
    """
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=None):
        dummy = torch.rand(1, 3, self.args.imgsz, self.args.imgsz, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if "running_mean" not in n:
                p.requires_grad = True
        self.oto = OTO(model, dummy)
        self.oto.mark_unprunable_by_param_names(YOLO26_UNPRUNABLE)
        sch = self.args.geta_schedule  # dict, see factory
        opt = self.oto.geta(
            variant="adam", lr=lr, lr_quant=1e-3, first_momentum=0.9, weight_decay=decay,
            target_group_sparsity=self.args.geta_sparsity,
            start_projection_step=sch["start_projection_step"],
            projection_periods=sch["projection_periods"],
            projection_steps=sch["projection_steps"],
            start_pruning_step=sch["start_pruning_step"],
            pruning_periods=sch["pruning_periods"],
            pruning_steps=sch["pruning_steps"],
        )
        return opt


def build_geta_trainer(overrides, sparsity, steps_per_epoch, epochs):
    """Compute the step-based GETA schedule from dataloader length and wire overrides.

    Schedule (in optimizer steps = batches): warm a few epochs, then project, then prune.
    """
    total = steps_per_epoch * epochs
    schedule = {
        "start_projection_step": int(0.1 * total),
        "projection_periods": 5,
        "projection_steps": int(0.4 * total),
        "start_pruning_step": int(0.5 * total),
        "pruning_periods": 5,
        "pruning_steps": int(0.4 * total),
    }
    overrides = dict(overrides)
    overrides.update({
        "geta_sparsity": sparsity,
        "geta_schedule": schedule,
        "amp": False,   # correctness first; re-enable after parity confirmed
        "optimizer": "auto",
    })
    return GetaDetectionTrainer(overrides=overrides), schedule
```

- [ ] **Step 2: Sanity-import on server (user)**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python -c "import sys; sys.path.insert(0,'.'); from experiments.geta_yolo26.geta_trainer import build_geta_trainer; print('import OK')"
```
Expected: `import OK` (verifies the Ultralytics `build_optimizer` signature matches and custom `args` keys are accepted; if Ultralytics rejects unknown `args` keys, store schedule/sparsity on the class instead — note in output).

- [ ] **Step 3: Commit**

```bash
git add experiments/geta_yolo26/geta_trainer.py && git commit -m "feat: GETA DetectionTrainer (Approach A) + schedule factory"
git push fork geta-yolo26-pruning
```

---

### Task 6: Prune+train entrypoint with EMA/AMP off — COCO8 smoke

**Files:**
- Create: `experiments/geta_yolo26/prune_train.py`

**Interfaces:**
- Consumes: `build_geta_trainer` (Task 5), `coco.yaml` (Task 2).
- Produces: a CLI entrypoint `python prune_train.py --data ... --epochs ... --sparsity ... --construct` that trains under GETA and (with `--construct`) writes a pruned subnet to `out/subnet/`. Consumed by Tasks 7–8.

- [ ] **Step 1: Write `experiments/geta_yolo26/prune_train.py`**

```python
import argparse, os, json
HERE = os.path.dirname(__file__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "coco.yaml"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sparsity", type=float, default=0.3)
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--name", default="geta_smoke")
    args = ap.parse_args()

    import sys; sys.path.insert(0, ".")
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from experiments.geta_yolo26.geta_trainer import build_geta_trainer

    # steps_per_epoch ~= train images / batch
    data = check_det_dataset(args.data)
    n_train = len(open(data["train"]).readlines()) if isinstance(data["train"], str) and data["train"].endswith(".txt") else None
    # Fallback: let Ultralytics compute; approximate with COCO sizes when unknown.
    steps_per_epoch = max(1, (n_train or 1000) // args.batch)

    overrides = dict(model=args.model, data=args.data, epochs=args.epochs,
                     batch=args.batch, imgsz=args.imgsz, device=0, name=args.name,
                     val=True, plots=False)
    trainer, schedule = build_geta_trainer(overrides, args.sparsity, steps_per_epoch, args.epochs)
    print("GETA schedule:", json.dumps(schedule))
    trainer.train()

    if args.construct:
        out = os.path.join(HERE, "out", "subnet")
        os.makedirs(out, exist_ok=True)
        trainer.oto.construct_subnet(out_dir=out)
        print("SUBNET", trainer.oto.compressed_model_path)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run COCO8 smoke on server (user)**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && \
python experiments/geta_yolo26/prune_train.py --data coco8.yaml --epochs 5 --batch 4 --sparsity 0.3 --construct --name geta_coco8
```
Expected: training runs to completion on COCO8 (minutes); `GETA schedule {...}` prints; loss decreases; a `SUBNET <path>` line prints a constructed compressed model. **This validates the optimizer steps + schedule + construct_subnet wiring end-to-end on tiny data.** Paste the tail of the log.

- [ ] **Step 3: COCO128 smoke (confirms validation/mAP path runs under GETA)**

Hand off:
```bash
cd ~/geta && python experiments/geta_yolo26/prune_train.py --data coco128.yaml --epochs 10 --batch 8 --sparsity 0.3 --construct --name geta_coco128
```
Expected: per-epoch Ultralytics `val` mAP line appears (non-crashing), schedule ramps sparsity, subnet constructed. Paste the final mAP + SUBNET lines.

- [ ] **Step 4: Commit**

```bash
git add experiments/geta_yolo26/prune_train.py && git commit -m "feat: GETA prune+train entrypoint; COCO8/128 smoke passing"
git push fork geta-yolo26-pruning
```

---

### Task 7: Full-COCO prune + fine-tune run

**Files:**
- Modify: `experiments/geta_yolo26/prune_train.py` (no code change expected; reused with full-COCO args)

**Interfaces:**
- Consumes: Task 6 entrypoint, full `coco.yaml`, baseline target (Task 3).
- Produces: a trained pruned `yolo26n` subnet at `out/subnet/` at the chosen sparsity. Consumed by Task 8 (val) and Tasks 9–10 (export/profile).

- [ ] **Step 1: Launch full-COCO run on server (user), in a detached session**

Hand off (use tmux/nohup so the SSH session can drop):
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && \
nohup python experiments/geta_yolo26/prune_train.py \
  --data experiments/geta_yolo26/coco.yaml --epochs 60 --batch 16 --imgsz 640 \
  --sparsity 0.3 --construct --name geta_coco_s30 > out_geta_coco_s30.log 2>&1 &
echo "PID $!"; sleep 30; tail -n 30 out_geta_coco_s30.log
```
Expected: training starts; GPU memory fits (if OOM, drop `--batch` to 8 and note it); schedule prints. Monitor with `tail -f out_geta_coco_s30.log`.

- [ ] **Step 2: On completion, confirm subnet exists (user)**

Hand off:
```bash
ls -la ~/geta/experiments/geta_yolo26/out/subnet/ && tail -n 40 ~/geta/out_geta_coco_s30.log
```
Expected: a compressed model file present; final training/val mAP line in the log. Paste both.

- [ ] **Step 3: Commit run config note**

```bash
git commit --allow-empty -m "chore: full-COCO GETA prune run (yolo26n, sparsity=0.3, 60ep, bs16)"
git push fork geta-yolo26-pruning
```
(If sparsity/batch/epochs changed from defaults to fit the 3080, edit the message to the actual values used.)

---

### Task 8: Validate pruned subnet mAP (parity check)

**Files:**
- Create: `experiments/geta_yolo26/val_subnet.py`

**Interfaces:**
- Consumes: `out/subnet/` compressed model (Task 7), `coco.yaml`, `out/baseline_metrics.json` (Task 3).
- Produces: `out/pruned_metrics.json` `{map5095, map50, params_M}` and a PASS/FAIL parity verdict (pruned ≥ baseline − 0.3).

- [ ] **Step 1: Write `experiments/geta_yolo26/val_subnet.py`**

```python
import json, os, glob, torch
from ultralytics import YOLO
HERE = os.path.dirname(__file__)

def load_pruned():
    cand = sorted(glob.glob(os.path.join(HERE, "out", "subnet", "*compress*")))
    assert cand, "no compressed subnet found"
    return torch.load(cand[-1]), cand[-1]

def main():
    pruned_module, path = load_pruned()
    # Wrap the pruned nn.Module back into a YOLO model for the official COCO validator.
    y = YOLO("yolo26n.pt")
    y.model = pruned_module.eval().cuda()
    metrics = y.val(data=os.path.join(HERE, "coco.yaml"), imgsz=640, batch=16, device=0)
    n_params = sum(p.numel() for p in pruned_module.parameters()) / 1e6
    rec = {"path": path, "map5095": float(metrics.box.map),
           "map50": float(metrics.box.map50), "params_M": round(n_params, 4)}
    with open(os.path.join(HERE, "out", "pruned_metrics.json"), "w") as f:
        json.dump(rec, f, indent=2)
    base = json.load(open(os.path.join(HERE, "out", "baseline_metrics.json")))
    verdict = "PASS" if rec["map5095"] >= base["map5095"] - 0.003 else "FAIL"
    print("PRUNED", json.dumps(rec))
    print(f"PARITY {verdict}: pruned {rec['map5095']:.4f} vs baseline {base['map5095']:.4f} "
          f"(params {base['params_M']:.2f}M -> {rec['params_M']:.2f}M)")

if __name__ == "__main__":
    main()
```
(Note: mAP is a 0–1 fraction in Ultralytics, so the 0.3-pt tolerance = 0.003.)

- [ ] **Step 2: Run on server (user)**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python experiments/geta_yolo26/val_subnet.py
```
Expected: `PRUNED {...}` then `PARITY PASS/FAIL ...`. Paste it. If FAIL, options recorded in spec §7-R2/R3: re-enable correctness toggles or extend fine-tune before treating it as a true negative.

- [ ] **Step 3: Commit**

```bash
git add experiments/geta_yolo26/val_subnet.py && git commit -m "feat: pruned subnet COCO val + parity verdict"
git push fork geta-yolo26-pruning
```

---

### Task 9: Export stock + pruned to ONNX

**Files:**
- Create: `experiments/geta_yolo26/export_onnx.py`

**Interfaces:**
- Consumes: `yolo26n.pt` (stock), `out/subnet/` (pruned), fixed opset.
- Produces: `out/yolo26n_stock.onnx` and `out/yolo26n_pruned.onnx` (both 640×640, batch=1, same opset). Consumed by Task 10.

- [ ] **Step 1: Write `experiments/geta_yolo26/export_onnx.py`**

```python
import os, glob, torch
from ultralytics import YOLO
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
OPSET = 17  # pinned; documented in plan Global Constraints

def export_stock():
    YOLO("yolo26n.pt").export(format="onnx", imgsz=640, opset=OPSET, dynamic=False, simplify=True)
    # ultralytics writes yolo26n.onnx next to the weights; move into OUT
    src = "yolo26n.onnx"
    os.replace(src, os.path.join(OUT, "yolo26n_stock.onnx"))

def export_pruned():
    cand = sorted(glob.glob(os.path.join(OUT, "subnet", "*compress*")))
    m = torch.load(cand[-1]).eval()
    dummy = torch.rand(1, 3, 640, 640)
    torch.onnx.export(m, dummy, os.path.join(OUT, "yolo26n_pruned.onnx"),
                      opset_version=OPSET, input_names=["images"], output_names=["output0"])

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    export_stock(); export_pruned()
    for f in ["yolo26n_stock.onnx", "yolo26n_pruned.onnx"]:
        p = os.path.join(OUT, f); print(f, os.path.getsize(p)/1e6, "MB")
```

- [ ] **Step 2: Run on server (user)**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python experiments/geta_yolo26/export_onnx.py
```
Expected: two `.onnx` files written with sizes printed; pruned size < stock size. Paste sizes. (If OTO provides a dedicated ONNX export that preserves the YOLO head better, prefer it for the pruned model and note the swap.)

- [ ] **Step 3: Commit**

```bash
git add experiments/geta_yolo26/export_onnx.py && git commit -m "feat: ONNX export for stock + pruned yolo26n (opset 17)"
git push fork geta-yolo26-pruning
```

---

### Task 10: ONNX speed profile + final comparison table

**Files:**
- Create: `experiments/geta_yolo26/profile_onnx.py`

**Interfaces:**
- Consumes: the two ONNX files (Task 9), `out/baseline_metrics.json`, `out/pruned_metrics.json`.
- Produces: `out/profile.json` + a printed comparison table populating spec §6 (params, GFLOPs, size, latency CUDA+CPU, mAP).

- [ ] **Step 1: Write `experiments/geta_yolo26/profile_onnx.py`**

```python
import os, json, time, glob
import numpy as np, onnxruntime as ort
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out")
WARMUP, ITERS = 20, 100

def bench(path, provider):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(path, sess_options=so, providers=[provider])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(WARMUP): sess.run(None, {name: x})
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter()-t)*1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p50_ms": float(np.percentile(ts,50)), "p95_ms": float(np.percentile(ts,95))}

def main():
    res = {}
    for tag, fn in [("stock","yolo26n_stock.onnx"), ("pruned","yolo26n_pruned.onnx")]:
        p = os.path.join(OUT, fn)
        res[tag] = {"size_MB": round(os.path.getsize(p)/1e6, 3)}
        for prov in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
            try: res[tag][prov] = bench(p, prov)
            except Exception as e: res[tag][prov] = {"error": str(e)}
    base = json.load(open(os.path.join(OUT,"baseline_metrics.json")))
    pruned = json.load(open(os.path.join(OUT,"pruned_metrics.json")))
    res["map5095"] = {"stock": base["map5095"], "pruned": pruned["map5095"]}
    res["params_M"] = {"stock": base["params_M"], "pruned": pruned["params_M"]}
    json.dump(res, open(os.path.join(OUT,"profile.json"),"w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on server (user)**

Hand off:
```bash
cd ~/geta && git pull fork geta-yolo26-pruning && python experiments/geta_yolo26/profile_onnx.py
```
Expected: `profile.json` with stock vs pruned latency (CUDA + CPU), size, params, mAP. Paste the JSON.

- [ ] **Step 3: Fill spec §6 metrics table + write findings**

Update `docs/superpowers/specs/2026-06-18-geta-yolo26-pruning-design.md` §6 table from `profile.json`, and append a short **Results** section: parity verdict, size/GFLOPs/latency deltas, whether the ONNX speedup materialized (per R4), and recommended next step (sparsity sweep or add quantization).

- [ ] **Step 4: Commit**

```bash
git add experiments/geta_yolo26/profile_onnx.py docs/superpowers/specs/2026-06-18-geta-yolo26-pruning-design.md
git commit -m "feat: ONNX speed profile + filled results table"
git push fork geta-yolo26-pruning
```

---

## Self-Review

**Spec coverage:** §1 success criteria → Tasks 3/8 (mAP parity) + 9/10 (smaller+faster ONNX). §2 constraints → Tasks 0–2 (fork sync, AutoDL env, COCO mount). §3 decisions → pruning-only (no `model_to_quantize_model`, Task 5/6), yolo26n only (all tasks), smoke-first (Task 6), Approach A (Task 5). §4 architecture → Task 4 (trace pattern), Task 5 (trainer), 9/10 (profiling). §5 phases → Tasks map 1:1 (Phase 0=T0, 1=T1–2, baseline=T3, 2=T4, 3=T5–6, 4=T7–8, 5=T9–10). §6 metrics → Task 10. §7 risks → R1 gate (T4), R2/R3 (T6/T8 toggles), R4 (T10 CPU+CUDA + honest note), R5 (T4 diff<1e-4), R6 (T2 dataset-check). All covered.

**Placeholder scan:** The one deliberate runtime-discovery hole is `YOLO26_UNPRUNABLE` in Task 4 — it is filled from `discover_head.py` *actual output* (Step 2) before the test runs, with concrete fallback (mark whole head). Not a stray TODO; it cannot be known pre-trace. No "TBD"/"handle edge cases"/"similar to" placeholders elsewhere.

**Type consistency:** `oto.geta(...)` kwargs match `only_train_once/__init__.py:122`. `build_geta_trainer(overrides, sparsity, steps_per_epoch, epochs)` defined in T5, called identically in T6. `YOLO26_UNPRUNABLE` defined T4, imported T5. `metrics.box.map`/`map50`, `out/baseline_metrics.json`/`pruned_metrics.json` keys consistent across T3/T8/T10. mAP tolerance unit clarified (0.003 = 0.3 pts) in T8.

## Known unverified assumptions (resolve during execution, not blockers)
- Ultralytics `DetectionTrainer.build_optimizer` signature and whether custom `args` keys (`geta_sparsity`, `geta_schedule`) pass through `self.args` — verified in Task 5 Step 2; fallback noted there.
- Exact AutoDL COCO path/layout — discovered in Task 2.
- YOLO26 head module index + output conv names — discovered in Task 4.
- Whether `yolo26n.pt` weights download is reachable from the AutoDL box — first hits in Task 3.
