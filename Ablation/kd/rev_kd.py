from __future__ import annotations

import torch


LAMBDA_REV_KL_MAX = 0.01
REV_WARMUP_EPOCHS = 2
REV_RAMP_EPOCHS = 3
KD_TEMPERATURE = 2.0
REV_CONF_THRESH = 0.20
REV_SAL_THRESH = 0.15


def sigmoid_with_temperature(logits: torch.Tensor, temperature: float = KD_TEMPERATURE) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return torch.sigmoid(logits / t)


def masked_bernoulli_kl_from_probs(
    p: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)
    kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    if mask is None:
        return kl.mean()
    mask = mask.float()
    denom = mask.sum()
    if denom.item() < 1.0:
        return kl.new_tensor(0.0)
    return (kl * mask).sum() / (denom + 1e-9)


def build_informative_mask(
    conf: torch.Tensor,
    sal_e: torch.Tensor,
    y_tp: torch.Tensor,
    conf_thr: float = REV_CONF_THRESH,
    sal_thr: float = REV_SAL_THRESH,
) -> torch.Tensor:
    return (y_tp > 0.5) | (conf >= conf_thr) | (sal_e >= sal_thr)


def current_rev_lambda(epoch_idx_1based: int) -> float:
    if epoch_idx_1based <= REV_WARMUP_EPOCHS:
        return 0.0
    ramp_pos = epoch_idx_1based - REV_WARMUP_EPOCHS
    frac = min(1.0, ramp_pos / max(1, REV_RAMP_EPOCHS))
    return LAMBDA_REV_KL_MAX * frac

