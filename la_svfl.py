import torch
import torch.nn as nn
import torch.nn.functional as F


def positive_quality_threshold(target_scores, min_tau=0.2):
    """tau = max(mean({q_i,c | q_i,c > 0}), min_tau) for the current batch."""

    positive_scores = target_scores[target_scores > 0]
    if positive_scores.numel() == 0:
        return target_scores.new_tensor(float(min_tau))
    return positive_scores.mean().clamp_min(float(min_tau))


def normalized_xyxy_box_areas(boxes, image_size, eps=1e-6):
    """
    Convert matched xyxy boxes to normalized areas A_i / A_img.

    Args:
        boxes: tensor shaped (..., 4), in pixel xyxy format.
        image_size: (height, width) of the training image.
    """

    h, w = image_size
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp_min(0)
    area = wh[..., 0] * wh[..., 1]
    return (area / float(h * w)).clamp(eps, 1.0)


def _reduce_query_class_loss(loss):
    """Apply 1 / N_q normalization from the paper and average over batch."""

    if loss.ndim < 2:
        raise ValueError("Expected loss to contain query and class dimensions.")
    nq = loss.shape[-2]
    reduced = loss.sum(dim=(-2, -1)) / max(int(nq), 1)
    return reduced.mean()


class SlideVarifocalLoss(nn.Module):
    """Varifocal loss with slide reweighting around the dynamic IoU threshold."""

    def forward(self, pred, true, one_hot, auto_iou=None):
        loss = self.loss_fcn(pred, true, one_hot)
        tau = positive_quality_threshold(true) if auto_iou is None else true.new_tensor(float(auto_iou)).clamp_min(0.2)
        b1 = true <= tau - 0.1
        b2 = (true > tau - 0.1) & (true < tau)
        b3 = true >= tau
        modulating_weight = 1.0 * b1 + torch.exp(1.0 - tau) * b2 + torch.exp(1.0 - true) * b3
        return _reduce_query_class_loss(loss * modulating_weight)

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

    def __init__(self, class_counts, rho=0.5, alpha_min=0.5, alpha_max=2.0, lambda_s=0.5, eps=1e-6):
        super().__init__()
        if isinstance(class_counts, dict):
            class_counts = [class_counts[k] for k in sorted(class_counts)]
        if class_counts is None or len(class_counts) == 0:
            raise ValueError("LA-SVFL requires the fixed training-set class_counts vector.")
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if torch.any(counts <= 0):
            raise ValueError("class_counts must contain positive instance counts for every class.")
        self.register_buffer("class_counts", counts, persistent=False)
        self.rho = rho
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda_s = lambda_s
        self.eps = eps

    def _class_prior_vector(self, device, nc):
        counts = self.class_counts.to(device).clamp_min(self.eps)
        if counts.numel() != nc:
            raise ValueError(f"class_counts has {counts.numel()} entries, but prediction has {nc} classes.")
        if self.rho <= 0:
            return torch.ones(nc, dtype=torch.float32, device=device)
        total = counts.sum().clamp_min(self.eps)
        return (total / (counts.numel() * counts)).pow(self.rho).clamp(self.alpha_min, self.alpha_max)

    def _scale_prior(self, box_areas, query_shape, device):
        if self.lambda_s <= 0:
            return torch.ones(query_shape, dtype=torch.float32, device=device)
        if box_areas is None:
            raise ValueError("LA-SVFL requires normalized box_areas when lambda_s > 0.")
        if tuple(box_areas.shape) != tuple(query_shape):
            raise ValueError(f"box_areas must have shape {tuple(query_shape)}, got {tuple(box_areas.shape)}.")
        areas = box_areas.to(device).float().clamp(self.eps, 1.0)
        scale_weight = 1.0 + self.lambda_s * (1.0 - torch.sqrt(areas))
        return scale_weight.clamp(1.0, 1.0 + self.lambda_s)

    def forward(self, pred, true, one_hot, auto_iou=None, box_areas=None):
        """
        Args:
            pred: logits shaped (..., N_q, C).
            true: IoU-aware target scores q_i,c with the same shape as pred.
            one_hot: binary class labels y_i,c with the same shape as pred.
            auto_iou: optional externally logged tau. If None, tau is computed
                from current-batch positive target scores exactly as in Eq. (11).
            box_areas: normalized matched-box areas shaped (..., N_q). Unmatched
                queries may contain any value; eta is applied only where one_hot=1.
        """

        if pred.shape != true.shape or pred.shape != one_hot.shape:
            raise ValueError("pred, true, and one_hot must have the same shape (..., N_q, C).")
        if pred.ndim < 2:
            raise ValueError("pred must contain query and class dimensions (..., N_q, C).")

        loss = self.loss_fcn(pred, true, one_hot)
        tau = positive_quality_threshold(true) if auto_iou is None else true.new_tensor(float(auto_iou)).clamp_min(0.2)
        b1 = true <= tau - 0.1
        b2 = (true > tau - 0.1) & (true < tau)
        b3 = true >= tau
        slide_weight = 1.0 * b1 + torch.exp(1.0 - tau) * b2 + torch.exp(1.0 - true) * b3

        nc = pred.shape[-1]
        query_shape = pred.shape[:-1]
        class_weight = self._class_prior_vector(pred.device, nc)
        class_shape = (1,) * (pred.ndim - 1) + (nc,)
        scale_weight = self._scale_prior(box_areas, query_shape, pred.device).unsqueeze(-1)
        eta = class_weight.view(class_shape) * scale_weight
        adaptive_weight = torch.where(one_hot.bool(), eta.expand_as(loss), torch.ones_like(loss))

        return _reduce_query_class_loss(loss * slide_weight * adaptive_weight)
