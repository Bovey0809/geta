"""Improved knowledge distillation for Ultralytics YOLO, as a drop-in subclass of the
built-in DistillationModel. Adds three toggleable terms the stock method lacks:

  logit : response/logit KD on the Detect head class-scores (one2many+one2one),
          soft-BCE with the teacher's sigmoid probabilities as targets. The stock method
          only uses head scores for spatial weighting, never distills the decisions.
  cwd   : channel-wise distillation (Shu 2021) -- per-channel softmax over spatial dims
          + KL divergence, replacing the stock raw score-weighted L2 on neck features.
  fgd   : Focal-and-Global-style global term -- spatial + channel attention MSE and a
          parameter-free global-context relation, complementing the focal weighting.

Each term has its own weight; with all off it falls back to the stock score-weighted L2,
so it A/B's cleanly against the built-in. Enable via ImprovedDistillationModel.CFG before
training; the runner monkeypatches the trainer to build this class.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from ultralytics.nn.distill_model import DistillationModel


class ImprovedDistillationModel(DistillationModel):
    # set before train(); read per-step in loss()
    CFG = {"logit": False, "cwd": False, "fgd": False,
           "logit_w": 1.0, "cwd_w": 4.0, "fgd_w": 2e-4, "cwd_T": 4.0}

    def cwd_loss(self, s, t, T):
        """Channel-wise KL: per-channel softmax over spatial positions, KL(teacher||student)."""
        n, c = s.shape[:2]
        s = s.view(n, c, -1)
        t = t.view(n, c, -1)
        log_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        return F.kl_div(log_s, p_t, reduction="batchmean") * (T * T) / c

    def fgd_global_loss(self, s, t):
        """Spatial + channel attention MSE and a parameter-free global-context relation."""
        def spatial_att(f):
            return f.abs().mean(1).flatten(1).softmax(-1)          # (N, H*W)

        def channel_att(f):
            return f.abs().mean((2, 3)).softmax(-1)                # (N, C)

        def global_ctx(f):
            n, c, h, w = f.shape
            att = f.abs().mean(1).flatten(1).softmax(-1).view(n, 1, h * w)
            return (f.view(n, c, h * w) * att).sum(-1)             # (N, C) attention-pooled

        return (F.mse_loss(spatial_att(s), spatial_att(t))
                + F.mse_loss(channel_att(s), channel_att(t))
                + F.mse_loss(global_ctx(s), global_ctx(t)))

    def logit_kd(self, student_head, teacher_head):
        """Soft-BCE distillation of head class-scores over both branches. Returns a scalar
        tensor, or None if the head scores aren't available in a comparable format."""
        loss = 0.0
        n = 0
        for br in ("one2many", "one2one"):
            ts = self.decouple_outputs(teacher_head, br)
            ss = self.decouple_outputs(student_head, br)
            if not (isinstance(ts, dict) and isinstance(ss, dict) and "scores" in ts and "scores" in ss):
                continue
            t_scores = ts["scores"].detach()
            s_scores = ss["scores"]
            if s_scores.shape != t_scores.shape:
                continue
            loss = loss + F.binary_cross_entropy_with_logits(s_scores, t_scores.sigmoid())
            n += 1
        return (loss / n) if n else None

    def loss(self, batch, preds=None):
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
            return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, loss_distill])

        cfg = self.CFG
        self._teacher_feats.clear()
        self._student_feats.clear()
        with torch.no_grad():
            self.teacher_model(batch["img"])
        preds = self.student_model(batch["img"])
        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)

        # focal spatial weights from teacher head (as in the stock method)
        teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        neck_feats = [self._teacher_feats[idx] for idx in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
        teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)

        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            tf = self.decouple_outputs(self._teacher_feats[feat_idx])
            sf = self.projector[i](self.decouple_outputs(self._student_feats[feat_idx]))
            if cfg["cwd"]:
                loss_distill += self.cwd_loss(sf, tf, cfg["cwd_T"]) * cfg["cwd_w"]
            else:
                loss_distill += self.loss_sl2(sf, tf, feat_idx=i, teacher_scores=teacher_scores) * self.dis
            if cfg["fgd"]:
                loss_distill += self.fgd_global_loss(sf, tf) * cfg["fgd_w"]

        if cfg["logit"]:
            lk = self.logit_kd(self._student_feats[self.feats_idx[-1]], teacher_head_feat)
            if torch.is_tensor(lk):
                loss_distill += lk * cfg["logit_w"]

        distill_loss_detach = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])
