from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


KD_LOSS = "bce_posweight"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


class RescoreHead(nn.Module):
    """Forward-KD head and BiKD student head."""

    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.cls_emb = nn.Embedding(num_classes, 16)
        self.mlp = nn.Sequential(
            nn.Linear(5 + 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, conf, sal_e, cx, cy, area, cls_idx):
        emb = self.cls_emb(cls_idx)
        x = torch.stack([conf, sal_e, cx, cy, area], dim=1)
        x = torch.cat([x, emb], dim=1)
        logits = self.mlp(x).squeeze(1)
        score = torch.sigmoid(logits)
        return score, logits


class TaskBridgeAuxHead(nn.Module):
    """Teacher-anchored auxiliary head used by BiKD / Rev-KD."""

    def __init__(
        self,
        num_classes: int,
        hidden: int = 128,
        init_beta: float = 0.5,
        use_tanh_residual: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_emb = nn.Embedding(num_classes, 16)
        self.use_tanh_residual = use_tanh_residual
        self.mlp = nn.Sequential(
            nn.Linear(4 + 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, conf, sal_e, cx, cy, area, cls_idx, eps: float = 1e-4):
        emb = self.cls_emb(cls_idx)
        sal_e_clamped = torch.clamp(sal_e, eps, 1.0 - eps)
        base_logit = torch.log(sal_e_clamped / (1.0 - sal_e_clamped))
        x = torch.stack([conf, cx, cy, area], dim=1)
        x = torch.cat([x, emb], dim=1)
        delta = self.mlp(x).squeeze(1)
        if self.use_tanh_residual:
            delta = torch.tanh(delta)
        logits = base_logit + self.beta * delta
        score = torch.sigmoid(logits)
        return score, logits


def binary_focal_loss_with_logits(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = w * (1 - pt).pow(gamma) * bce
    return loss.mean()


def supervised_loss(logits, targets, pos_weight, loss_mode: str = KD_LOSS):
    if loss_mode == "bce_posweight":
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    if loss_mode == "focal":
        return binary_focal_loss_with_logits(logits, targets)
    raise ValueError(f"Unknown loss_mode: {loss_mode}")

