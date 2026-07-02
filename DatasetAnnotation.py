

import os
import glob
import time
from collections import OrderedDict

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
BASE_ROOT = "/home/aixlab/DrEYEve"

SEQ_START, SEQ_END = 5, 10         # folders 05..10
SAMPLE_EVERY = 5                   # sample every N frames

# Frame geometry (Garmin) – usually 1920x1080 but we always read actual image size
IMGSZ = 640
DEVICE = 0                         # GPU index or "cpu"
HALF = True

# Option A: proposal pool fairness
PROPOSAL_CONF = 0.001              # very low conf to harvest proposals
IOU_THRES = 0.50                   # NMS IoU
MAX_DET = 300                      # ask YOLO for many
CAND_TOP_N = 120                   # keep fixed N proposals per frame AFTER filtering

# Attended-object rule (Rule 1′)
SAL_ALPHA = 0.28
TOP_K_PER_FRAME = 2

# Visualization style
HEAT_ALPHA = 0.35
ALL_COLOR = (255, 0, 0)            # blue (BGR)
ATT_COLOR = (0, 220, 0)            # green (BGR)

# Our 7 classes
OUR_CLASSES = [
    "people",        # 0
    "car",           # 1
    "motorcycle",    # 2
    "traffic-light", # 3
    "traffic-sign",  # 4
    "bus",           # 5
    "truck"          # 6
]
OUR_TO_ID = {n: i for i, n in enumerate(OUR_CLASSES)}

# Robust aliases (name-based mapping, NOT id-based)
COCO_NAME_ALIASES = {
    "people": {"person", "people", "pedestrian"},
    "car": {"car", "automobile"},
    "motorcycle": {"motorcycle", "motorbike", "motor bike"},
    "traffic-light": {"traffic light", "traffic-light"},
    "traffic-sign": {"stop sign", "traffic sign", "traffic-sign"},
    "bus": {"bus"},
    "truck": {"truck"},
}

# Compare these 4 models (adjust paths if needed)
STUDENT_MODELS = OrderedDict([
    ("yolo11x", "yolo11x.pt"),
    ("yolo12x", "yolo12x.pt"),
    ("yolo26x", "yolo26x.pt"),
    ("yolov8x-worldv2", "yolov8x-worldv2.pt"),
])

# =============================================================================
# FP CONTROL + EGO HOOD SUPPRESSION (STRONG)
# =============================================================================
SMALL_OBJ_CLASS_IDS = {OUR_TO_ID["traffic-light"], OUR_TO_ID["traffic-sign"]}
VEHICLE_CLASS_IDS = {OUR_TO_ID["car"], OUR_TO_ID["bus"], OUR_TO_ID["truck"]}

SMALL_OBJ_MIN_SIDE_PX = 4
GENERAL_MIN_SIDE_PX = 8
SMALL_OBJ_MIN_AREA_NORM = 3e-6
GENERAL_MIN_AREA_NORM = 1e-5
VEHICLE_MIN_AREA_NORM = 7e-4

# Ego hood ROI (bottom-center). Tune if needed.
EGO_HOOD_SUPPRESS = True
EGO_HOOD_ROI_NORM = (0.10, 0.66, 0.90, 1.00)   # x1,y1,x2,y2 normalized
EGO_HOOD_OVERLAP_THR = 0.25                    # more aggressive
EGO_HOOD_MIN_BOTTOM_N = 0.84                   # allow catching slightly higher bottom edges

# Extra “thin wide strip” hood catcher (common failure)
EGO_HOOD_THIN_W_N = 0.70
EGO_HOOD_THIN_H_MAX_N = 0.13
EGO_HOOD_THIN_TOP_MIN_N = 0.58

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# UTILITIES
# =============================================================================
def natural_sorted_frame_list(folder):
    exts = ("*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    if not files:
        raise FileNotFoundError(f"No images found under: {folder}")

    def keyfunc(p):
        base = os.path.basename(p)
        nums = "".join([c if c.isdigit() else " " for c in base]).split()
        return int(nums[-1]) if nums else base

    return sorted(files, key=keyfunc)


def _denorm_xywh_to_xyxy(cx_n, cy_n, w_n, h_n, W, H):
    cx, cy = cx_n * W, cy_n * H
    bw, bh = w_n * W, h_n * H
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    return x1, y1, x2, y2


def norm_cls_name(x: str) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def build_alias_to_our_id():
    alias_to_our = {}
    for our_name, aliases in COCO_NAME_ALIASES.items():
        for a in aliases:
            alias_to_our[norm_cls_name(a)] = OUR_TO_ID[our_name]
    return alias_to_our


ALIAS_TO_OUR = build_alias_to_our_id()


def yolo_cls_to_our_id(names_dict, cls_id: int):
    n = names_dict.get(int(cls_id), None)
    if n is None:
        return None
    nn = norm_cls_name(n)
    return ALIAS_TO_OUR.get(nn, None)


def box_area_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def inter_area_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def hood_roi_xyxy(W, H):
    x1n, y1n, x2n, y2n = EGO_HOOD_ROI_NORM
    return [x1n * W, y1n * H, x2n * W, y2n * H]


def is_ego_hood_detection(cls_id: int, xyxy, W, H) -> bool:
    """
    Strong hood suppression:
    - applies to vehicle-like classes (car/bus/truck)
    - overlap with bottom-center ROI
    - plus a thin-wide-strip catcher (dashboard band)
    """
    if not EGO_HOOD_SUPPRESS:
        return False
    if cls_id not in VEHICLE_CLASS_IDS:
        return False

    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 2 or bh < 2:
        return False

    w_n = bw / (W + 1e-9)
    h_n = bh / (H + 1e-9)
    top_n = y1 / (H + 1e-9)
    bot_n = y2 / (H + 1e-9)

    # must be low-ish
    if bot_n < EGO_HOOD_MIN_BOTTOM_N:
        return False

    # thin wide strip catcher (common in your mosaics)
    if (w_n >= EGO_HOOD_THIN_W_N) and (h_n <= EGO_HOOD_THIN_H_MAX_N) and (top_n >= EGO_HOOD_THIN_TOP_MIN_N):
        return True

    # ROI overlap
    roi = hood_roi_xyxy(W, H)
    inter = inter_area_xyxy([x1, y1, x2, y2], roi)
    area = box_area_xyxy([x1, y1, x2, y2]) + 1e-9
    overlap = inter / area
    return overlap >= EGO_HOOD_OVERLAP_THR


def passes_geometry_filter(cls_id: int, xyxy, W, H) -> bool:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_n = (bw * bh) / ((W * H) + 1e-9)

    min_side = SMALL_OBJ_MIN_SIDE_PX if cls_id in SMALL_OBJ_CLASS_IDS else GENERAL_MIN_SIDE_PX
    if bw < min_side or bh < min_side:
        return False

    if cls_id in SMALL_OBJ_CLASS_IDS:
        if area_n < SMALL_OBJ_MIN_AREA_NORM:
            return False
    else:
        if area_n < GENERAL_MIN_AREA_NORM:
            return False

    if cls_id in VEHICLE_CLASS_IDS and area_n < VEHICLE_MIN_AREA_NORM:
        return False

    # absurd aspect for vehicles
    if cls_id in VEHICLE_CLASS_IDS:
        aspect = max(bw / (bh + 1e-9), bh / (bw + 1e-9))
        if aspect > 10.0:
            return False

    return True


def compute_saliency_share(sal_gray, xyxy):
    """
    share = sum(saliency inside bbox) / sum(saliency over full frame)
    """
    if sal_gray is None:
        return 0.0
    Hs, Ws = sal_gray.shape[:2]
    tot = float(sal_gray.sum())
    if tot <= 0:
        tot = 1e-6

    x1, y1, x2, y2 = xyxy
    x1i = max(0, min(Ws - 1, int(round(x1))))
    x2i = max(0, min(Ws - 1, int(round(x2))))
    y1i = max(0, min(Hs - 1, int(round(y1))))
    y2i = max(0, min(Hs - 1, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    patch = sal_gray[y1i:y2i + 1, x1i:x2i + 1]
    s = float(patch.sum())
    return float(s / tot)


def overlay_saliency(img_bgr, sal_gray):
    if sal_gray is None:
        return img_bgr
    H, W = img_bgr.shape[:2]
    sal_resized = cv2.resize(sal_gray, (W, H), interpolation=cv2.INTER_LINEAR)
    sal_color = cv2.applyColorMap(sal_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(sal_color, HEAT_ALPHA, img_bgr, 1.0 - HEAT_ALPHA, 0)


# =============================================================================
# CORE: Run one model on sampled frames and save overlays
# =============================================================================
@torch.no_grad()
def run_model_on_sequence(seq_id: str, model_tag: str, weights: str, sampled_pairs, out_dir):
    """
    sampled_pairs: list of (stem, img_path, sal_path)
    Saves: out_dir/{stem}.jpg
    """
    print(f"[{seq_id}][{model_tag}] Loading model: {weights}")
    model = YOLO(weights)

    # a bit safer: force fp16 only if CUDA
    use_half = bool(HALF and (DEVICE != "cpu") and torch.cuda.is_available())

    for stem, img_path, sal_path in sampled_pairs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        if img is None or sal is None:
            continue

        H, W = img.shape[:2]

        # predict with low conf for proposal harvest (Option A)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(
            source=img_rgb,
            imgsz=IMGSZ,
            conf=PROPOSAL_CONF,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=DEVICE,
            half=use_half,
            verbose=False,
        )

        r = results[0]
        names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            conf = r.boxes.conf.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)

            for bb, cf, cc in zip(xyxy, conf, cls):
                our_id = yolo_cls_to_our_id(names, int(cc))
                if our_id is None:
                    continue

                x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                x1 = max(0.0, min(W - 1.0, x1))
                x2 = max(0.0, min(W - 1.0, x2))
                y1 = max(0.0, min(H - 1.0, y1))
                y2 = max(0.0, min(H - 1.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                xy = [x1, y1, x2, y2]

                # geometry + hood suppression (IMPORTANT)
                if not passes_geometry_filter(our_id, xy, W, H):
                    continue
                if is_ego_hood_detection(our_id, xy, W, H):
                    continue

                dets.append({
                    "our_id": int(our_id),
                    "our_name": OUR_CLASSES[int(our_id)],
                    "conf": float(cf),
                    "xyxy": xy
                })

        # Option A: fixed proposal pool size
        dets.sort(key=lambda d: -d["conf"])
        dets = dets[:min(CAND_TOP_N, len(dets))]

        # compute saliency shares
        for d in dets:
            d["share"] = compute_saliency_share(sal, d["xyxy"])

        # attended selection (Rule 1′)
        attended = [d for d in dets if d["share"] >= SAL_ALPHA]
        if len(attended) == 0 and len(dets) > 0:
            dets_sorted_by_share = sorted(dets, key=lambda d: -d["share"])
            attended = dets_sorted_by_share[:min(TOP_K_PER_FRAME, len(dets_sorted_by_share))]

        attended_set = set(id(x) for x in attended)

        # draw
        vis = overlay_saliency(img, sal)

        # all detections (blue)
        for d in dets:
            x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), ALL_COLOR, 2)
            txt = f"{d['our_name']} c={d['conf']:.2f} sh={d['share']:.2f}"
            cv2.putText(vis, txt, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ALL_COLOR, 2, cv2.LINE_AA)

        # attended (green, thicker)
        for d in attended:
            x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), ATT_COLOR, 3)
            txt = f"{d['our_name']} (att) sh={d['share']:.2f}"
            cv2.putText(vis, txt, (x1, min(H - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, ATT_COLOR, 2, cv2.LINE_AA)

        # model tag
        cv2.putText(vis, model_tag, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, model_tag, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, vis)

    # cleanup GPU memory
    del model
    if torch.cuda.is_available() and DEVICE != "cpu":
        torch.cuda.empty_cache()


def make_mosaic_2x2(img_tl, img_tr, img_bl, img_br):
    """
    2x2 mosaic, all images resized to same size.
    """
    H = min(img_tl.shape[0], img_tr.shape[0], img_bl.shape[0], img_br.shape[0])
    W = min(img_tl.shape[1], img_tr.shape[1], img_bl.shape[1], img_br.shape[1])

    def rs(im):
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)

    tl = rs(img_tl)
    tr = rs(img_tr)
    bl = rs(img_bl)
    br = rs(img_br)

    top = np.concatenate([tl, tr], axis=1)
    bot = np.concatenate([bl, br], axis=1)
    return np.concatenate([top, bot], axis=0)


# =============================================================================
# PER-SEQUENCE DRIVER
# =============================================================================
def process_sequence(seq_id: str):
    t0 = time.time()
    seq_dir = os.path.join(BASE_ROOT, seq_id)
    frames_dir = os.path.join(seq_dir, "garmin_frames")
    sal_dir = os.path.join(seq_dir, "saliency_frames")

    if not os.path.isdir(frames_dir):
        print(f"[WARN] Missing frames: {frames_dir}")
        return
    if not os.path.isdir(sal_dir):
        print(f"[WARN] Missing saliency: {sal_dir}")
        return

    frame_paths = natural_sorted_frame_list(frames_dir)
    sal_paths = natural_sorted_frame_list(sal_dir)
    T = min(len(frame_paths), len(sal_paths))
    frame_paths = frame_paths[:T]
    sal_paths = sal_paths[:T]

    # sampled pairs
    sampled_pairs = []
    for t in range(0, T, SAMPLE_EVERY):
        img_path = frame_paths[t]
        sal_path = sal_paths[t]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        sampled_pairs.append((stem, img_path, sal_path))

    print("=" * 90)
    print(f"[SEQ {seq_id}] frames={T} | sampled={len(sampled_pairs)} | {frames_dir}")
    print("=" * 90)

    # outputs
    out_root = os.path.join(seq_dir, "compare_vis_optionA_4models")
    per_model_root = os.path.join(out_root, "per_model")
    mosaic_root = os.path.join(out_root, "mosaic")
    os.makedirs(per_model_root, exist_ok=True)
    os.makedirs(mosaic_root, exist_ok=True)

    # 1) generate per-model overlays (one model loaded at a time to avoid crashes)
    for model_tag, weights in STUDENT_MODELS.items():
        out_dir = os.path.join(per_model_root, model_tag)
        run_model_on_sequence(seq_id, model_tag, weights, sampled_pairs, out_dir)

    # 2) build mosaics
    model_tags = list(STUDENT_MODELS.keys())
    if len(model_tags) != 4:
        raise RuntimeError("This script expects exactly 4 models for 2x2 mosaic.")

    tl_tag, tr_tag, bl_tag, br_tag = model_tags  # order in STUDENT_MODELS

    for stem, _, _ in sampled_pairs:
        p_tl = os.path.join(per_model_root, tl_tag, f"{stem}.jpg")
        p_tr = os.path.join(per_model_root, tr_tag, f"{stem}.jpg")
        p_bl = os.path.join(per_model_root, bl_tag, f"{stem}.jpg")
        p_br = os.path.join(per_model_root, br_tag, f"{stem}.jpg")

        if not (os.path.isfile(p_tl) and os.path.isfile(p_tr) and os.path.isfile(p_bl) and os.path.isfile(p_br)):
            continue

        im_tl = cv2.imread(p_tl, cv2.IMREAD_COLOR)
        im_tr = cv2.imread(p_tr, cv2.IMREAD_COLOR)
        im_bl = cv2.imread(p_bl, cv2.IMREAD_COLOR)
        im_br = cv2.imread(p_br, cv2.IMREAD_COLOR)
        if any(x is None for x in [im_tl, im_tr, im_bl, im_br]):
            continue

        mos = make_mosaic_2x2(im_tl, im_tr, im_bl, im_br)
        out_path = os.path.join(mosaic_root, f"{stem}.jpg")
        cv2.imwrite(out_path, mos)

    print(f"[SEQ {seq_id}] Saved: {out_root}")
    print(f"[SEQ {seq_id}] Done in {time.time() - t0:.1f}s")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Slight speed help
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for i in range(SEQ_START, SEQ_END + 1):
        seq_id = f"{i:02d}"
        process_sequence(seq_id)
