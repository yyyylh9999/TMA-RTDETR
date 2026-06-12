import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideVarifocalLoss(nn.Module):
    """Varifocal loss with slide reweighting around the dynamic IoU threshold."""

    def forward(self, pred, true, one_hot, auto_iou=0.5):
        loss = self.loss_fcn(pred, true, one_hot)
        auto_iou = max(float(auto_iou), 0.2)
        b1 = true <= auto_iou - 0.1
        b2 = (true > auto_iou - 0.1) & (true < auto_iou)
        b3 = true >= auto_iou
        modulating_weight = 1.0 * b1 + math.exp(1.0 - auto_iou) * b2 + torch.exp(-(true - 1.0)) * b3
        return (loss * modulating_weight).mean(1).sum()

    @staticmethod
    def loss_fcn(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
        return loss


class LesionAdaptiveSlideVarifocalLoss(SlideVarifocalLoss):
    """
    LA-SVFL adds class-prior and lesion-scale modulation to SlideVarifocalLoss.

    Args:
        class_counts: per-class instance counts used for long-tail reweighting.
        rho: strength of class-prior reweighting.
        lambda_s: strength of small-lesion scale weighting.
    """

    def __init__(self, class_counts=None, rho=0.5, alpha_min=0.5, alpha_max=2.0, lambda_s=0.5, eps=1e-6):
        super().__init__()
        if isinstance(class_counts, dict):
            class_counts = [class_counts[k] for k in sorted(class_counts)]
        counts = torch.as_tensor(class_counts or [], dtype=torch.float32)
        self.register_buffer("class_counts", counts, persistent=False)
        self.rho = rho
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda_s = lambda_s
        self.eps = eps

    def _class_prior(self, targets, nc):
        if self.class_counts.numel() == 0 or self.rho <= 0:
            return torch.ones_like(targets, dtype=torch.float32, device=targets.device)
        counts = self.class_counts.to(targets.device).clamp_min(self.eps)
        total = counts.sum().clamp_min(self.eps)
        prior = (total / (counts.numel() * counts)).pow(self.rho).clamp(self.alpha_min, self.alpha_max)
        safe_targets = targets.clamp(0, counts.numel() - 1).long()
        weight = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
        pos_mask = (targets < nc) & (targets < counts.numel())
        weight[pos_mask] = prior[safe_targets[pos_mask]]
        return weight

    def _scale_prior(self, targets, box_areas, nc):
        weight = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
        if box_areas is None or self.lambda_s <= 0:
            return weight
        pos_mask = targets < nc
        areas = box_areas.to(targets.device).float().clamp(self.eps, 1.0)
        scale_weight = 1.0 + self.lambda_s * (1.0 - torch.sqrt(areas))
        weight[pos_mask] = scale_weight[pos_mask].clamp(1.0, 1.0 + self.lambda_s)
        return weight

    def forward(self, pred, true, one_hot, auto_iou=0.5, targets=None, box_areas=None):
        loss = self.loss_fcn(pred, true, one_hot)
        auto_iou = max(float(auto_iou), 0.2)
        b1 = true <= auto_iou - 0.1
        b2 = (true > auto_iou - 0.1) & (true < auto_iou)
        b3 = true >= auto_iou
        loss *= 1.0 * b1 + math.exp(1.0 - auto_iou) * b2 + torch.exp(-(true - 1.0)) * b3

        if targets is not None:
            nc = pred.shape[-1]
            class_weight = self._class_prior(targets, nc)
            scale_weight = self._scale_prior(targets, box_areas, nc)
            adaptive_weight = (class_weight * scale_weight).unsqueeze(-1)
            loss *= torch.where(one_hot.bool(), adaptive_weight, torch.ones_like(loss))

        return loss.mean(1).sum()
