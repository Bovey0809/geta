"""GETA structured pruning inside the Ultralytics DetectionTrainer (Approach A).

The GETA optimizer induces group sparsity *during* training and tracks its own
projection/pruning schedule internally (driven by each optimizer.step()). We:
  - replace build_optimizer with oto.geta(...) on the OTO-traced model,
  - disable EMA (its weight averaging would blur GETA's exact-zero groups so
    construct_subnet could no longer detect pruned channels),
  - rely on the caller to pass amp=False, warmup_epochs=0, nbs==batch (so each
    batch is one optimizer step and the schedule below lines up).
After train(), call self.oto.construct_subnet(out_dir=...) to emit the pruned net.
"""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from only_train_once import OTO
from sanity_check.test_yolo26 import yolo26_unprunable_names


class GetaDetectionTrainer(DetectionTrainer):
    geta_sparsity = 0.5  # set on the class before train()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oto = None
        self.add_callback("on_train_start", self._disable_ema)

    @staticmethod
    def _disable_ema(trainer):
        if getattr(trainer, "ema", None) is not None:
            trainer.ema.enabled = False

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        dev = next(model.parameters()).device
        for n, p in model.named_parameters():
            if "running_mean" not in n:
                p.requires_grad = True
        dummy = torch.rand(1, 3, self.args.imgsz, self.args.imgsz, device=dev)
        self.oto = OTO(model, dummy)
        self.oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        it = max(int(iterations), 100)
        opt = self.oto.geta(
            variant="adam", lr=lr, lr_quant=1e-3, first_momentum=0.9, weight_decay=decay,
            target_group_sparsity=self.geta_sparsity,
            start_projection_step=int(0.10 * it), projection_periods=5, projection_steps=int(0.40 * it),
            start_pruning_step=int(0.50 * it), pruning_periods=5, pruning_steps=int(0.40 * it),
        )
        print(f"[GETA] sparsity={self.geta_sparsity} iterations={it} "
              f"proj@{int(0.10*it)} prune@{int(0.50*it)}..{int(0.90*it)}")
        return opt
