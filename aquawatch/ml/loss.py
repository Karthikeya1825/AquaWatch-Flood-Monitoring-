"""
AquaWatch - Masked Loss Functions and Metrics for Satellite Water Segmentation

Sen1Floods11 Ground Truth values:
  1: Water
  0: Land / Not Water
 -1: Cloud / Shadow / Invalid (MUST be masked out in loss and metric computation)

Loss formulation:
  L = L_BCE + L_Dice (both calculated strictly over valid pixels where target != -1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MaskedBCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss with invalid pixel masking.
    Masks out pixels where target == -1 (clouds / no data).
    """
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float = 1.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (B, 1, H, W) raw model output
            targets: (B, 1, H, W) ground truth with values {1, 0, -1}
        Returns:
            total_loss, bce_loss, dice_loss
        """
        valid_mask = (targets != -1)
        if not valid_mask.any():
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return zero, zero, zero

        # 1. Masked Binary Cross-Entropy
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        pos_w = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=pos_w
        )

        # 2. Masked Soft Dice Loss
        probs = torch.sigmoid(logits)
        valid_probs = probs[valid_mask]

        intersection = torch.sum(valid_probs * valid_targets)
        cardinality = torch.sum(valid_probs) + torch.sum(valid_targets)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_score

        # Total combined loss
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return total_loss, bce, dice_loss


def calculate_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Compute IoU (Jaccard), F1 (Dice), Precision, and Recall for a batch,
    masking out any pixels where target == -1.
    """
    with torch.no_grad():
        valid_mask = (targets != -1)
        if not valid_mask.any():
            return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        probs = torch.sigmoid(logits)
        preds = (probs > threshold)

        valid_preds = preds[valid_mask]
        valid_targets = (targets[valid_mask] == 1)

        tp = (valid_preds & valid_targets).sum().float().item()
        fp = (valid_preds & (~valid_targets)).sum().float().item()
        fn = ((~valid_preds) & valid_targets).sum().float().item()
        tn = ((~valid_preds) & (~valid_targets)).sum().float().item()

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp) / denom_f1 if denom_f1 > 0 else 0.0

        denom_p = tp + fp
        precision = tp / denom_p if denom_p > 0 else 0.0

        denom_r = tp + fn
        recall = tp / denom_r if denom_r > 0 else 0.0

        return {
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
