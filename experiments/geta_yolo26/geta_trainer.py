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
    geta_lr = None       # if set, overrides Ultralytics' auto LR (use a fine-tune LR)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oto = None
        self.add_callback("on_train_start", self._disable_ema)

    @staticmethod
    def _disable_ema(trainer):
        if getattr(trainer, "ema", None) is not None:
            trainer.ema.enabled = False

    # GETA's optimizer has a non-standard state_dict() (no 'state' key), which breaks
    # Ultralytics' fp16 checkpoint save. We construct the pruned subnet directly from
    # the trained model, so skip Ultralytics' checkpointing / best-weight reload.
    def save_model(self):
        pass

    def final_eval(self):
        pass

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        dev = next(model.parameters()).device
        for n, p in model.named_parameters():
            if "running_mean" not in n:
                p.requires_grad = True
        dummy = torch.rand(1, 3, self.args.imgsz, self.args.imgsz, device=dev)
        self.oto = OTO(model, dummy)
        self.oto.mark_unprunable_by_param_names(yolo26_unprunable_names(model))
        if self.geta_lr is not None:
            lr = self.geta_lr  # fine-tune LR (Ultralytics 'auto' otherwise forces lr0=0.01)
        it = max(int(iterations), 100)
        # Pruning only -> use HESSO (oto.geta() is the joint prune+quant optimizer and
        # assumes quantization layers exist, which we don't insert).
        opt = self.oto.hesso(
            variant="sgd", lr=lr, weight_decay=decay, first_momentum=momentum,
            target_group_sparsity=self.geta_sparsity,
            start_pruning_step=int(0.20 * it), pruning_periods=10, pruning_steps=int(0.50 * it),
            group_divisible=1,
        )
        print(f"[GETA/HESSO] sparsity={self.geta_sparsity} iterations={it} "
              f"prune@{int(0.20*it)}..{int(0.70*it)}")
        return opt
