import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import timm


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    # Modes:
    #   "train_and_test" -> train on 01..38 with val split, then test on 39..74
    #   "test_only"      -> only evaluate TEST_CKPT on 39..74
    RUN_MODE: str = "train_and_test"

    DREYEVE_ROOT: str = "/home/aixlab/DrEYEve"

    # Protocol
    TRAIN_SEQ_START: int = 1
    TRAIN_SEQ_END: int = 38
    TEST_SEQ_START: int = 39
    TEST_SEQ_END: int = 74
    VAL_MIDDLE_FRAMES_PER_SEQ: int = 500

    # Model
    BACKBONE_NAME: str = "vit_base_patch16_224.dino"
    PATCH_SIZE: int = 16
    IMG_SIZE: int = 224  
    PRETRAINED_BACKBONE: bool = True

    # Partial init from your previous checkpoint (backbone keys will still load)
    INIT_CKPT: str = ".../teacher_saliency_dino_224_dreyeve.pth"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 123

    # Training
    EPOCHS: int = 10
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    AMP: bool = True
    GRAD_CLIP_NORM: float = 1.0

    # Fine-tuning strategy
    UNFREEZE_LAST_N_BLOCKS: int = 2
    LR_DECODER: float = 2e-4
    LR_BACKBONE: float = 2e-5
    WEIGHT_DECAY: float = 1e-5

    # Loss weights (aligned for better CC / lower KLD)
    W_KL: float = 0.40
    W_CC: float = 0.25
    W_SIM: float = 0.20
    W_WBCE: float = 0.12
    W_TV: float = 0.03

    # Weighted BCE emphasis on salient pixels
    WBCE_POS_BOOST: float = 4.0

    # Eval-time post-processing (helps CC/KLD)
    EVAL_TTA_HFLIP: bool = True
    EVAL_GAUSSIAN_BLUR: bool = True
    EVAL_BLUR_KERNEL: int = 11
    EVAL_BLUR_SIGMA: float = 3.0

    # Output / checkpoints
    OUT_DIR: str = os.path.join(DREYEVE_ROOT, "teacher_finetune_stronger_decoder_dreyeve")
    BEST_CKPT: str = os.path.join(OUT_DIR, "best.pth")
    LAST_CKPT: str = os.path.join(OUT_DIR, "last.pth")
    SUMMARY_TXT: str = os.path.join(OUT_DIR, "test_summary.txt")

    # Used only in test_only mode
    TEST_CKPT: str = os.path.join(DREYEVE_ROOT, "teacher_finetune_stronger_decoder_dreyeve", "best.pth")


cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# Repro
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Split builder (DR(eye)VE protocol)
# ============================================================
def z2(i: int) -> str:
    return str(i).zfill(2)


def list_sequence_frames(seq_dir: str) -> List[str]:
    frames_dir = os.path.join(seq_dir, "garmin_frames")
    if not os.path.isdir(frames_dir):
        return []

    files = []
    for f in os.listdir(frames_dir):
        p = os.path.join(frames_dir, f)
        if os.path.isfile(p) and os.path.splitext(f.lower())[1] in IMG_EXTS:
            files.append(p)

    def key_fn(p: str):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return (0, int(base))
        except Exception:
            return (1, base)

    files.sort(key=key_fn)
    return files


def paired_paths_for_frame(img_path: str) -> Tuple[str, str]:
    seq_dir = os.path.dirname(os.path.dirname(img_path))  # .../XX
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]

    sal_path = os.path.join(seq_dir, "saliency_frames", fname)
    if not os.path.isfile(sal_path):
        alt_png = os.path.join(seq_dir, "saliency_frames", stem + ".png")
        alt_jpg = os.path.join(seq_dir, "saliency_frames", stem + ".jpg")
        if os.path.isfile(alt_png):
            sal_path = alt_png
        elif os.path.isfile(alt_jpg):
            sal_path = alt_jpg

    return img_path, sal_path


def build_dreyeve_split() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_pairs: List[Tuple[str, str]] = []
    val_pairs: List[Tuple[str, str]] = []
    test_pairs: List[Tuple[str, str]] = []

    # TRAIN / VAL
    for s in range(cfg.TRAIN_SEQ_START, cfg.TRAIN_SEQ_END + 1):
        seq_dir = os.path.join(cfg.DREYEVE_ROOT, z2(s))
        frames = list_sequence_frames(seq_dir)
        if not frames:
            print(f"[WARN] No frames found for train seq {z2(s)} at {seq_dir}")
            continue

        n = len(frames)
        mid = n // 2
        k = cfg.VAL_MIDDLE_FRAMES_PER_SEQ
        half = k // 2

        start = max(0, mid - half)
        end = min(n, start + k)
        start = max(0, end - k)
        val_set = set(frames[start:end])

        for fp in frames:
            img_p, sal_p = paired_paths_for_frame(fp)
            if not (os.path.isfile(img_p) and os.path.isfile(sal_p)):
                continue
            if fp in val_set:
                val_pairs.append((img_p, sal_p))
            else:
                train_pairs.append((img_p, sal_p))

    # TEST
    for s in range(cfg.TEST_SEQ_START, cfg.TEST_SEQ_END + 1):
        seq_dir = os.path.join(cfg.DREYEVE_ROOT, z2(s))
        frames = list_sequence_frames(seq_dir)
        if not frames:
            print(f"[WARN] No frames found for test seq {z2(s)} at {seq_dir}")
            continue

        for fp in frames:
            img_p, sal_p = paired_paths_for_frame(fp)
            if not (os.path.isfile(img_p) and os.path.isfile(sal_p)):
                continue
            test_pairs.append((img_p, sal_p))

    print(f"[Split] train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


def build_dreyeve_test_pairs_only() -> List[Tuple[str, str]]:
    test_pairs: List[Tuple[str, str]] = []
    missing_sal = 0

    for s in range(cfg.TEST_SEQ_START, cfg.TEST_SEQ_END + 1):
        seq_dir = os.path.join(cfg.DREYEVE_ROOT, z2(s))
        frames = list_sequence_frames(seq_dir)
        if not frames:
            print(f"[WARN] No frames found for test seq {z2(s)} at {seq_dir}")
            continue

        for fp in frames:
            img_p, sal_p = paired_paths_for_frame(fp)
            if not os.path.isfile(img_p):
                continue
            if not os.path.isfile(sal_p):
                missing_sal += 1
                continue
            test_pairs.append((img_p, sal_p))

    print(f"[TestPairs] pairs={len(test_pairs)} | missing_saliency={missing_sal}")
    return test_pairs


# ============================================================
# Paired transforms
# ============================================================
class TrainPairTransform:
    """
    Kept conservative:
    - no horizontal flip (driving saliency has side-specific priors)
    - mild photometric jitter only
    """
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.color = transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.01
        )

    def __call__(self, img: Image.Image, gt: Image.Image):
        img = self.color(img)

        img = TF.resize(img, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)
        gt = TF.resize(gt, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(
            img_t,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        gt_t = TF.to_tensor(gt)  # [1,H,W] in [0,1]
        return img_t, gt_t


class EvalPairTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(self, img: Image.Image, gt: Image.Image):
        img = TF.resize(img, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)
        gt = TF.resize(gt, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(
            img_t,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        gt_t = TF.to_tensor(gt)
        return img_t, gt_t


# ============================================================
# Dataset
# ============================================================
class DreyeveSaliencyDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], pair_tf):
        super().__init__()
        filtered = []
        for img_p, gt_p in pairs:
            if os.path.isfile(img_p) and os.path.isfile(gt_p):
                filtered.append((img_p, gt_p))
        self.pairs = filtered
        self.pair_tf = pair_tf

        if not self.pairs:
            raise RuntimeError("No valid (image, gt) pairs found.")
        print(f"[Dataset] pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.pairs[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("L")
        except FileNotFoundError:
            return None

        img_t, gt_t = self.pair_tf(img, gt)
        return img_t, gt_t, img_path, gt_path


def dreyeve_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, gts, img_paths, gt_paths = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(gts, 0), list(img_paths), list(gt_paths)


# ============================================================
# Model
# ============================================================
class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, 1),
            ConvBNReLU(cout, cout, 3, 1, 1),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return x


class StrongSaliencyNet(nn.Module):
    def __init__(self, backbone_name: str, patch_size: int, img_size: int, pretrained_backbone: bool = True):
        super().__init__()
        assert img_size % patch_size == 0, "IMG_SIZE must be divisible by PATCH_SIZE"

        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool=""
        )
        embed_dim = self.backbone.num_features

        # Stronger progressive decoder than the original 2-layer head
        self.proj = ConvBNReLU(embed_dim, 256, 1, 1, 0)
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

        # Learnable center-bias prior (classic and effective for saliency maps)
        self.center_bias = nn.Parameter(torch.zeros(1, 1, self.grid_size, self.grid_size))

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                tokens = feats["x_norm_patchtokens"]
            elif "x" in feats:
                tokens = feats["x"]
            else:
                raise ValueError(f"Unsupported backbone feature dict keys: {list(feats.keys())}")
        elif isinstance(feats, torch.Tensor):
            tokens = feats
        else:
            raise ValueError(f"Unexpected forward_features output: {type(feats)}")

        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = self._extract_tokens(x)  # [B, N or N+1, C]

        gh, gw = H // self.patch_size, W // self.patch_size
        expected = gh * gw

        # Remove CLS token if present
        if tokens.dim() != 3:
            raise ValueError(f"Unexpected token shape: {tokens.shape}")
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:, :]
        if tokens.shape[1] != expected:
            raise ValueError(f"Token count mismatch: got {tokens.shape[1]} expected {expected}")

        feat = tokens.transpose(1, 2).reshape(B, tokens.shape[2], gh, gw)

        x = self.proj(feat)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.head(x)

        cb = F.interpolate(self.center_bias, size=(H, W), mode="bilinear", align_corners=False)
        logits = logits + cb
        return logits


def load_partial_ckpt(model: nn.Module, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("[Init] No init checkpoint found; using pretrained backbone + random decoder.")
        return

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model_sd = model.state_dict()
    loadable = {}
    skipped = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped += 1

    msg = model.load_state_dict(loadable, strict=False)
    print(f"[Init] Loaded {len(loadable)} tensors from {ckpt_path} | skipped={skipped}")
    print(f"[Init] Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")


def set_trainable_params(model: StrongSaliencyNet):
    # Freeze everything first
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Unfreeze last N transformer blocks + norm
    n = max(0, int(cfg.UNFREEZE_LAST_N_BLOCKS))
    if n > 0 and hasattr(model.backbone, "blocks"):
        blocks = list(model.backbone.blocks)
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(model.backbone, "norm"):
            for p in model.backbone.norm.parameters():
                p.requires_grad = True

    # Decoder + center bias always trainable
    for p in model.proj.parameters():
        p.requires_grad = True
    for p in model.up1.parameters():
        p.requires_grad = True
    for p in model.up2.parameters():
        p.requires_grad = True
    for p in model.up3.parameters():
        p.requires_grad = True
    for p in model.up4.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    model.center_bias.requires_grad = True


def build_model(device: torch.device) -> StrongSaliencyNet:
    model = StrongSaliencyNet(
        backbone_name=cfg.BACKBONE_NAME,
        patch_size=cfg.PATCH_SIZE,
        img_size=cfg.IMG_SIZE,
        pretrained_backbone=cfg.PRETRAINED_BACKBONE
    )
    load_partial_ckpt(model, cfg.INIT_CKPT)
    set_trainable_params(model)
    model.to(device)
    return model


def build_optimizer(model: StrongSaliencyNet):
    backbone_params = []
    decoder_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            decoder_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_params, "lr": cfg.LR_DECODER},
            {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        ],
        weight_decay=cfg.WEIGHT_DECAY
    )
    return optimizer


# ============================================================
# Metrics
# ============================================================
def norm_minmax_np(m: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    m = m.astype(np.float32)
    m = m - m.min()
    m = m / (m.max() + eps)
    return m


def norm_prob_np(m: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    m = norm_minmax_np(m, eps=eps)
    s = float(m.sum())
    if s < eps:
        return np.full_like(m, 1.0 / (m.size + eps), dtype=np.float32)
    return (m / s).astype(np.float32)


def cc_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> float:
    p = pred.astype(np.float32).reshape(-1)
    g = gt.astype(np.float32).reshape(-1)
    p = (p - p.mean()) / (p.std() + eps)
    g = (g - g.mean()) / (g.std() + eps)
    return float((p * g).mean())


def kld_np(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    P = norm_prob_np(pred, eps=eps)
    Q = norm_prob_np(gt, eps=eps)
    return float((Q * np.log((Q + eps) / (P + eps))).sum())


def sim_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> float:
    P = norm_prob_np(pred, eps=eps)
    Q = norm_prob_np(gt, eps=eps)
    return float(np.minimum(P, Q).sum())


def val_score_from_metrics(m: Dict[str, float]) -> float:
    # Higher is better. Tuned to favor CC up and KLD down.
    return float(m["mean_CC"] + 0.35 * m["mean_SIM"] - 0.15 * m["mean_KLD"])


# ============================================================
# Losses
# ============================================================
def prob_norm_torch(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    x = x.float()
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + eps)
    s = x.sum(dim=(-2, -1), keepdim=True)
    return x / (s + eps)


def kl_torch(gt01: torch.Tensor, pred01: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    Q = prob_norm_torch(gt01, eps=eps)
    P = prob_norm_torch(pred01, eps=eps)
    return (Q * torch.log((Q + eps) / (P + eps))).sum(dim=(-2, -1)).mean()


def cc_loss_torch(gt01: torch.Tensor, pred01: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    g = gt01.flatten(2).float()
    p = pred01.flatten(2).float()
    g = (g - g.mean(dim=2, keepdim=True)) / (g.std(dim=2, keepdim=True) + eps)
    p = (p - p.mean(dim=2, keepdim=True)) / (p.std(dim=2, keepdim=True) + eps)
    cc = (g * p).mean(dim=2)
    return (1.0 - cc).mean()


def sim_loss_torch(gt01: torch.Tensor, pred01: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    Q = prob_norm_torch(gt01, eps=eps)
    P = prob_norm_torch(pred01, eps=eps)
    sim = torch.minimum(P, Q).sum(dim=(-2, -1)).mean()
    return 1.0 - sim


def weighted_bce_logits(logits: torch.Tensor, gt01: torch.Tensor, pos_boost: float = 4.0) -> torch.Tensor:
    weight = 1.0 + pos_boost * gt01
    return F.binary_cross_entropy_with_logits(logits, gt01, weight=weight, reduction="mean")


def tv_loss_torch(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def total_loss(logits: torch.Tensor, gt01: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred01 = torch.sigmoid(logits)

    loss_kl = kl_torch(gt01, pred01)
    loss_cc = cc_loss_torch(gt01, pred01)
    loss_sim = sim_loss_torch(gt01, pred01)
    loss_wbce = weighted_bce_logits(logits, gt01, pos_boost=cfg.WBCE_POS_BOOST)
    loss_tv = tv_loss_torch(pred01)

    loss = (
        cfg.W_KL * loss_kl +
        cfg.W_CC * loss_cc +
        cfg.W_SIM * loss_sim +
        cfg.W_WBCE * loss_wbce +
        cfg.W_TV * loss_tv
    )

    parts = {
        "kl": float(loss_kl.item()),
        "cc": float(loss_cc.item()),
        "sim": float(loss_sim.item()),
        "wbce": float(loss_wbce.item()),
        "tv": float(loss_tv.item()),
    }
    return loss, parts


# ============================================================
# Eval-time smoothing / TTA
# ============================================================
def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_torch(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    B, C, H, W = x.shape
    kernel = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
    pad = kernel_size // 2
    x = F.conv2d(x, kernel, padding=pad, groups=C)
    return x


@torch.no_grad()
def predict_prob_eval(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(model(imgs))

    if cfg.EVAL_TTA_HFLIP:
        imgs_f = torch.flip(imgs, dims=[3])
        pred_f = torch.sigmoid(model(imgs_f))
        pred_f = torch.flip(pred_f, dims=[3])
        pred = 0.5 * (pred + pred_f)

    if cfg.EVAL_GAUSSIAN_BLUR:
        pred = gaussian_blur_torch(pred, cfg.EVAL_BLUR_KERNEL, cfg.EVAL_BLUR_SIGMA)

    return pred.clamp(0.0, 1.0)


# ============================================================
# Train / Eval loops
# ============================================================
def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        if batch is None:
            continue

        imgs, gt, _, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.AMP and device.type == "cuda")):
            logits = model(imgs)
            loss, _ = total_loss(logits, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0

    for batch in loader:
        if batch is None:
            continue

        imgs, gt, _, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        logits = model(imgs)
        loss, _ = total_loss(logits, gt)

        bs = imgs.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate_metrics(model, loader, device) -> Dict[str, float]:
    model.eval()
    cc_list, kld_list, sim_list = [], [], []
    skipped_batches = 0
    seen = 0

    for batch in loader:
        if batch is None:
            skipped_batches += 1
            continue

        imgs, gt, _, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        pred01 = predict_prob_eval(model, imgs)

        pred_np = pred01[:, 0].float().cpu().numpy()
        gt_np = gt[:, 0].float().cpu().numpy()

        for i in range(pred_np.shape[0]):
            p = norm_minmax_np(pred_np[i])
            g = norm_minmax_np(gt_np[i])

            cc_list.append(cc_np(p, g))
            kld_list.append(kld_np(g, p))
            sim_list.append(sim_np(p, g))
            seen += 1

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    out = {
        "mean_CC": mean(cc_list),
        "mean_KLD": mean(kld_list),
        "mean_SIM": mean(sim_list),
        "num_samples": float(seen),
        "skipped_batches": float(skipped_batches),
    }
    out["val_score"] = val_score_from_metrics(out)
    return out


# ============================================================
# Main
# ============================================================
def main_train_and_test():
    seed_all(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    print(f"[Device] {device}")
    print(f"[Root] {cfg.DREYEVE_ROOT}")

    train_pairs, val_pairs, test_pairs = build_dreyeve_split()

    train_ds = DreyeveSaliencyDataset(train_pairs, TrainPairTransform(cfg.IMG_SIZE))
    val_ds = DreyeveSaliencyDataset(val_pairs, EvalPairTransform(cfg.IMG_SIZE))
    test_ds = DreyeveSaliencyDataset(test_pairs, EvalPairTransform(cfg.IMG_SIZE))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=dreyeve_collate,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=dreyeve_collate,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=dreyeve_collate,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    model = build_model(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable={trainable} / total={total}")

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.AMP and device.type == "cuda"))

    best_score = -1e18
    best_val_loss = float("inf")

    print(f"[Train] epochs={cfg.EPOCHS} batch={cfg.BATCH_SIZE} img={cfg.IMG_SIZE}")
    print(
        f"[Loss] KL={cfg.W_KL:.2f} CC={cfg.W_CC:.2f} SIM={cfg.W_SIM:.2f} "
        f"WBCE={cfg.W_WBCE:.2f} TV={cfg.W_TV:.2f}"
    )
    print(
        f"[FT] unfreeze_last_n_blocks={cfg.UNFREEZE_LAST_N_BLOCKS} "
        f"lr_decoder={cfg.LR_DECODER:.2e} lr_backbone={cfg.LR_BACKBONE:.2e}"
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device)
        va_loss = eval_loss(model, val_loader, device)
        va_metrics = evaluate_metrics(model, val_loader, device)
        score = va_metrics["val_score"]

        lr_dec = optimizer.param_groups[0]["lr"]
        lr_bb = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_dec

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"train_loss={tr:.6f} | val_loss={va_loss:.6f} | "
            f"val_CC={va_metrics['mean_CC']:.4f} | "
            f"val_KLD={va_metrics['mean_KLD']:.4f} | "
            f"val_SIM={va_metrics['mean_SIM']:.4f} | "
            f"score={score:.4f} | "
            f"lr_dec={lr_dec:.2e} | lr_bb={lr_bb:.2e}"
        )

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": va_loss,
                "val_score": score,
                "val_metrics": va_metrics,
            },
            cfg.LAST_CKPT
        )

        if score > best_score:
            best_score = score
            best_val_loss = va_loss
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": va_loss,
                    "val_score": score,
                    "val_metrics": va_metrics,
                },
                cfg.BEST_CKPT
            )
            print(
                f"[Save] best -> {cfg.BEST_CKPT} "
                f"(score={best_score:.4f}, CC={va_metrics['mean_CC']:.4f}, KLD={va_metrics['mean_KLD']:.4f})"
            )

        scheduler.step()

    # Final test using best checkpoint
    print("\n[TEST] Loading best checkpoint...")
    ckpt = torch.load(cfg.BEST_CKPT, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    m = evaluate_metrics(model, test_loader, device)

    print("\n===== TEST SUMMARY (DR(eye)VE) =====")
    print(f"Samples: {int(m['num_samples'])}")
    print(f"Mean CC : {m['mean_CC']:.6f}")
    print(f"Mean KLD: {m['mean_KLD']:.6f}")
    print(f"Mean SIM: {m['mean_SIM']:.6f}")
    print(f"Score   : {m['val_score']:.6f}")

    with open(cfg.SUMMARY_TXT, "w") as f:
        f.write("Stronger saliency fine-tuning on DR(eye)VE\n")
        f.write(f"Backbone: {cfg.BACKBONE_NAME}\n")
        f.write(f"Init ckpt: {cfg.INIT_CKPT}\n")
        f.write(f"EPOCHS: {cfg.EPOCHS}\n")
        f.write(f"IMG_SIZE: {cfg.IMG_SIZE}\n")
        f.write(f"UNFREEZE_LAST_N_BLOCKS: {cfg.UNFREEZE_LAST_N_BLOCKS}\n")
        f.write(
            f"Loss weights: KL={cfg.W_KL}, CC={cfg.W_CC}, SIM={cfg.W_SIM}, "
            f"WBCE={cfg.W_WBCE}, TV={cfg.W_TV}\n"
        )
        for k, v in m.items():
            f.write(f"{k}: {v}\n")

    print(f"[Saved] best={cfg.BEST_CKPT}")
    print(f"[Saved] last={cfg.LAST_CKPT}")
    print(f"[Saved] summary={cfg.SUMMARY_TXT}")


def main_test_only():
    seed_all(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    print(f"[Device] {device}")
    print(f"[Root] {cfg.DREYEVE_ROOT}")

    test_pairs = build_dreyeve_test_pairs_only()
    test_ds = DreyeveSaliencyDataset(test_pairs, EvalPairTransform(cfg.IMG_SIZE))
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=dreyeve_collate,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    model = StrongSaliencyNet(
        backbone_name=cfg.BACKBONE_NAME,
        patch_size=cfg.PATCH_SIZE,
        img_size=cfg.IMG_SIZE,
        pretrained_backbone=False
    )

    if not os.path.isfile(cfg.TEST_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.TEST_CKPT}")

    ckpt = torch.load(cfg.TEST_CKPT, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    m = evaluate_metrics(model, test_loader, device)

    print("\n===== TEST SUMMARY (DR(eye)VE) =====")
    print(f"Samples: {int(m['num_samples'])}")
    print(f"Mean CC : {m['mean_CC']:.6f}")
    print(f"Mean KLD: {m['mean_KLD']:.6f}")
    print(f"Mean SIM: {m['mean_SIM']:.6f}")
    print(f"Score   : {m['val_score']:.6f}")

    with open(cfg.SUMMARY_TXT, "w") as f:
        f.write("TEST-ONLY evaluation on DR(eye)VE\n")
        f.write(f"Checkpoint: {cfg.TEST_CKPT}\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")

    print(f"[Saved] summary={cfg.SUMMARY_TXT}")


if __name__ == "__main__":
    if cfg.RUN_MODE == "train_and_test":
        main_train_and_test()
    elif cfg.RUN_MODE == "test_only":
        main_test_only()
    else:
        raise ValueError(f"Unknown RUN_MODE: {cfg.RUN_MODE}")
