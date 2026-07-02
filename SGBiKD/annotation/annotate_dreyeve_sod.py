from __future__ import annotations

import argparse
import glob
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


BASE_ROOT = os.environ.get("KD_DREYEVE_ROOT", "/home/aixlab/DrEYEve")
START_SEQ = int(os.environ.get("KD_DREYEVE_SEQ_START", "1"))
END_SEQ = int(os.environ.get("KD_DREYEVE_SEQ_END", "74"))

YOLO_WEIGHTS = os.environ.get("KD_DREYEVE_ANN_WEIGHTS", "yolo12x.pt")
IMGSZ = int(os.environ.get("KD_DREYEVE_ANN_IMGSZ", "640"))
DEVICE = os.environ.get("KD_DREYEVE_ANN_DEVICE", "0")
HALF = os.environ.get("KD_DREYEVE_ANN_HALF", "1") != "0"

CONF_THRES = float(os.environ.get("KD_DREYEVE_ANN_CONF", "0.25"))
IOU_THRES = float(os.environ.get("KD_DREYEVE_ANN_IOU", "0.50"))
MAX_DET = int(os.environ.get("KD_DREYEVE_ANN_MAX_DET", "100"))

SAL_ALPHA = float(os.environ.get("KD_DREYEVE_SAL_ALPHA", "0.28"))
TOP_K_PER_FRAME = int(os.environ.get("KD_DREYEVE_TOPK", "2"))
K_SAL_WINDOW = int(os.environ.get("KD_DREYEVE_K_WINDOW", "5"))
MIN_CANDIDATE_FRAMES = int(os.environ.get("KD_DREYEVE_MIN_CAND", "5"))
N_PROPAGATE = int(os.environ.get("KD_DREYEVE_PROPAGATE", "60"))
SAMPLE_EVERY = int(os.environ.get("KD_DREYEVE_SAMPLE_EVERY", "5"))
OUT_LABEL_DIR = os.environ.get("KD_DREYEVE_LABEL_DIR", "labels_Y12")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

OUR_CLASSES = [
    "people",
    "car",
    "motorcycle",
    "traffic-light",
    "traffic-sign",
    "bus",
    "truck",
]
OUR_TO_ID = {n: i for i, n in enumerate(OUR_CLASSES)}

COCO_NAME_ALIASES = {
    "people": {"person", "people", "pedestrian"},
    "car": {"car", "automobile"},
    "motorcycle": {"motorcycle", "motorbike", "motor bike", "scooter"},
    "traffic-light": {"traffic light", "traffic-light"},
    "traffic-sign": {"stop sign", "traffic sign", "traffic-sign"},
    "bus": {"bus", "coach bus", "school bus"},
    "truck": {"truck", "pickup truck", "lorry", "semi-truck", "semi truck", "train"},
}

EGO_HOOD_ZONE = (0.18, 0.78, 0.82, 1.00)
EGO_MIN_ZONE_IOB = 0.35
EGO_BOTTOM_TOUCH = 0.94
EGO_MIN_CENTER_Y = 0.78
EGO_MIN_WIDTH = 0.16
EGO_WIDE_WIDTH = 0.32
EGO_VERY_WIDE = 0.50
EGO_WIDE_TOP = 0.60
EGO_CENTER_X_RANGE = (0.20, 0.80)


def natural_sorted_frame_list(folder: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No frames found under: {folder}")

    def keyfunc(path: str):
        base = os.path.basename(path)
        nums = "".join(ch if ch.isdigit() else " " for ch in base).split()
        return int(nums[-1]) if nums else base

    return sorted(files, key=keyfunc)


def norm_cls_name(x: str) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def build_alias_to_our_id() -> Dict[str, int]:
    alias_to_our: Dict[str, int] = {}
    for our_name, aliases in COCO_NAME_ALIASES.items():
        for alias in aliases:
            alias_to_our[norm_cls_name(alias)] = OUR_TO_ID[our_name]
    return alias_to_our


ALIAS_TO_OUR_ID = build_alias_to_our_id()


def yolo_names_to_ours(names, cls_id: int) -> Optional[int]:
    if isinstance(names, dict):
        name = names.get(int(cls_id), None)
    else:
        name = names[int(cls_id)] if 0 <= int(cls_id) < len(names) else None
    if name is None:
        return None
    return ALIAS_TO_OUR_ID.get(norm_cls_name(name), None)


def bbox_xyxy_to_yolo_txt(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def _box_area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return _box_area(ix1, iy1, ix2, iy2)


def _rect_norm_to_abs(rect_norm, W, H):
    rx1n, ry1n, rx2n, ry2n = rect_norm
    return (rx1n * W, ry1n * H, rx2n * W, ry2n * H)


def _iob_with_norm_rect(x1, y1, x2, y2, rect_norm, W, H):
    box = (x1, y1, x2, y2)
    rect = _rect_norm_to_abs(rect_norm, W, H)
    inter = _intersection_area_xyxy(box, rect)
    area = _box_area(x1, y1, x2, y2)
    if area <= 1e-6:
        return 0.0
    return inter / area


def is_ego_car_bbox(x1, y1, x2, y2, W, H, cls_name):
    if cls_name != OUR_TO_ID["car"]:
        return False
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 1 or box_h <= 1:
        return False
    width_ratio = box_w / float(W)
    height_ratio = box_h / float(H)
    cx_norm = ((x1 + x2) / 2.0) / float(W)
    cy_norm = ((y1 + y2) / 2.0) / float(H)
    top_norm = y1 / float(H)
    bottom_norm = y2 / float(H)
    centered = EGO_CENTER_X_RANGE[0] <= cx_norm <= EGO_CENTER_X_RANGE[1]
    near_bottom = cy_norm >= EGO_MIN_CENTER_Y
    touches_bottom = bottom_norm >= EGO_BOTTOM_TOUCH
    wide_enough = width_ratio >= EGO_MIN_WIDTH
    hood_zone_iob = _iob_with_norm_rect(x1, y1, x2, y2, EGO_HOOD_ZONE, W, H)
    rule_zone = centered and near_bottom and wide_enough and hood_zone_iob >= EGO_MIN_ZONE_IOB
    rule_bottom_wide = centered and touches_bottom and width_ratio >= EGO_WIDE_WIDTH and top_norm >= EGO_WIDE_TOP
    rule_very_wide = centered and touches_bottom and near_bottom and width_ratio >= EGO_VERY_WIDE
    rule_shallow_wide = centered and near_bottom and width_ratio >= 0.22 and height_ratio <= 0.22 and bottom_norm >= 0.90
    return rule_zone or rule_bottom_wide or rule_very_wide or rule_shallow_wide


def suppress_ego_tracks(per_frame, W, H):
    stats = defaultdict(lambda: {"n": 0, "ego_votes": 0, "cx": [], "cy": [], "w": [], "bottom": []})
    for dets in per_frame:
        for det in dets:
            if det["our_cls_id"] != OUR_TO_ID["car"]:
                continue
            tid = det["track_id"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            stats[tid]["n"] += 1
            stats[tid]["ego_votes"] += int(det.get("is_ego", False))
            stats[tid]["cx"].append(((x1 + x2) / 2.0) / float(W))
            stats[tid]["cy"].append(((y1 + y2) / 2.0) / float(H))
            stats[tid]["w"].append((x2 - x1) / float(W))
            stats[tid]["bottom"].append(y2 / float(H))

    ego_track_ids = set()
    for tid, s in stats.items():
        if s["n"] == 0:
            continue
        vote_ratio = s["ego_votes"] / float(s["n"])
        med_cx = float(np.median(s["cx"]))
        med_cy = float(np.median(s["cy"]))
        med_w = float(np.median(s["w"]))
        med_b = float(np.median(s["bottom"]))
        centered = EGO_CENTER_X_RANGE[0] <= med_cx <= EGO_CENTER_X_RANGE[1]
        if centered and (
            (vote_ratio >= 0.30 and med_w >= 0.16 and med_cy >= 0.78)
            or (med_w >= 0.32 and med_b >= 0.95 and med_cy >= 0.80)
            or (vote_ratio >= 0.20 and med_w >= 0.22 and med_b >= 0.92 and med_cy >= 0.78)
        ):
            ego_track_ids.add(tid)

    for dets in per_frame:
        for det in dets:
            if det["our_cls_id"] == OUR_TO_ID["car"] and det["track_id"] in ego_track_ids:
                det["is_ego"] = True
    return ego_track_ids


def process_sequence(seq_id: str, model: YOLO):
    start = time.time()
    seq_dir = os.path.join(BASE_ROOT, seq_id)
    frames_dir = os.path.join(seq_dir, "garmin_frames")
    sal_dir = os.path.join(seq_dir, "saliency_frames")
    out_lbl_dir = os.path.join(seq_dir, OUT_LABEL_DIR)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not os.path.isdir(frames_dir):
        print(f"[WARN] missing frames dir: {frames_dir}")
        return
    if not os.path.isdir(sal_dir):
        print(f"[WARN] missing saliency dir: {sal_dir}")
        return

    frame_paths_sorted = natural_sorted_frame_list(frames_dir)
    sal_paths_sorted = natural_sorted_frame_list(sal_dir)
    probe = cv2.imread(frame_paths_sorted[0], cv2.IMREAD_COLOR)
    if probe is None:
        print(f"[WARN] failed to read first frame in {frames_dir}")
        return
    frame_h, frame_w = probe.shape[:2]

    use_half = bool(HALF and DEVICE != "cpu")
    results_gen = model.track(
        source=frames_dir,
        stream=True,
        imgsz=IMGSZ,
        device=DEVICE,
        half=use_half,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
    )

    per_frame = []
    yolo_names = None
    for r in results_gen:
        if yolo_names is None:
            yolo_names = r.names
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.arange(len(xyxy))
            for bb, c, tid in zip(xyxy, cls, ids):
                our_cls_id = yolo_names_to_ours(yolo_names, c)
                if our_cls_id is None:
                    continue
                x1, y1, x2, y2 = map(float, bb.tolist())
                x1 = max(0.0, min(frame_w - 1.0, x1))
                x2 = max(0.0, min(frame_w - 1.0, x2))
                y1 = max(0.0, min(frame_h - 1.0, y1))
                y2 = max(0.0, min(frame_h - 1.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                ego_flag = is_ego_car_bbox(x1, y1, x2, y2, frame_w, frame_h, our_cls_id)
                dets.append({"track_id": int(tid), "bbox_xyxy": [x1, y1, x2, y2], "our_cls_id": int(our_cls_id), "is_ego": ego_flag})
        per_frame.append(dets)

    T = min(len(per_frame), len(frame_paths_sorted), len(sal_paths_sorted))
    per_frame = per_frame[:T]
    frame_paths_sorted = frame_paths_sorted[:T]
    sal_paths_sorted = sal_paths_sorted[:T]
    ego_track_ids = suppress_ego_tracks(per_frame, frame_w, frame_h)
    print(f"[{seq_id}] suppressed ego tracks: {len(ego_track_ids)} | frames={T}")

    frame_to_all_lines = {}
    for t in range(T):
        stem = os.path.splitext(os.path.basename(frame_paths_sorted[t]))[0]
        img = cv2.imread(frame_paths_sorted[t], cv2.IMREAD_COLOR)
        if img is None:
            frame_to_all_lines[stem] = []
            continue
        H_use, W_use = img.shape[:2]
        all_lines = []
        for det in per_frame[t]:
            if det.get("is_ego", False):
                continue
            cls_id = det["our_cls_id"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy, bw, bh = bbox_xyxy_to_yolo_txt([x1, y1, x2, y2], W_use, H_use)
            all_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        frame_to_all_lines[stem] = all_lines

    track_to_frames = defaultdict(list)
    for t in range(T):
        for det in per_frame[t]:
            if not det.get("is_ego", False):
                track_to_frames[det["track_id"]].append(t)
    for tid in track_to_frames:
        track_to_frames[tid] = sorted(set(track_to_frames[tid]))

    sal_share = [dict() for _ in range(T)]
    for t in range(T):
        sal = cv2.imread(sal_paths_sorted[t], cv2.IMREAD_GRAYSCALE)
        if sal is None:
            continue
        Hs, Ws = sal.shape[:2]
        total_sal = float(sal.sum())
        if total_sal <= 0:
            total_sal = 1e-6
        for det in per_frame[t]:
            if det.get("is_ego", False):
                continue
            tid = det["track_id"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            x1i = max(0, min(Ws - 1, int(round(x1))))
            x2i = max(0, min(Ws - 1, int(round(x2))))
            y1i = max(0, min(Hs - 1, int(round(y1))))
            y2i = max(0, min(Hs - 1, int(round(y2))))
            if x2i <= x1i or y2i <= y1i:
                continue
            patch = sal[y1i:y2i + 1, x1i:x2i + 1]
            share = float(patch.sum()) / total_sal
            if share > 0:
                sal_share[t][tid] = share

    is_candidate = [dict() for _ in range(T)]
    for t in range(T):
        shares = sal_share[t]
        if not shares:
            continue
        candidates_t = {tid for tid, sh in shares.items() if sh >= SAL_ALPHA}
        if not candidates_t and sum(shares.values()) > 0:
            sorted_by_share = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
            candidates_t = {tid for tid, _ in sorted_by_share[:TOP_K_PER_FRAME]}
        for tid in candidates_t:
            is_candidate[t][tid] = True

    is_salient = [dict() for _ in range(T)]
    for tid, frames_list in track_to_frames.items():
        for t in frames_list:
            r1 = is_candidate[t].get(tid, False)
            lo = max(0, t - K_SAL_WINDOW)
            hi = min(T - 1, t + K_SAL_WINDOW)
            count_cand = sum(1 for tau in range(lo, hi + 1) if is_candidate[tau].get(tid, False))
            r2 = count_cand >= MIN_CANDIDATE_FRAMES
            if r1 or r2:
                is_salient[t][tid] = True

    for tid, frames_list in track_to_frames.items():
        labeled = sorted([t for t in frames_list if is_salient[t].get(tid, False)])
        if len(labeled) < 2:
            continue
        for i in range(len(labeled) - 1):
            t1, t2 = labeled[i], labeled[i + 1]
            if (t2 - t1) <= N_PROPAGATE:
                for tau in range(t1 + 1, t2):
                    is_salient[tau][tid] = True

    exported = 0
    for t in range(0, T, SAMPLE_EVERY):
        img_path = frame_paths_sorted[t]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H_use, W_use = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_lbl = os.path.join(out_lbl_dir, f"{stem}.txt")
        attended_lines = []
        sal_t = is_salient[t]
        if len(per_frame[t]) > 0 and len(sal_t) > 0:
            for det in per_frame[t]:
                if det.get("is_ego", False):
                    continue
                tid = det["track_id"]
                if not sal_t.get(tid, False):
                    continue
                cls_id = det["our_cls_id"]
                x1, y1, x2, y2 = det["bbox_xyxy"]
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy, bw, bh = bbox_xyxy_to_yolo_txt([x1, y1, x2, y2], W_use, H_use)
                attended_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        all_lines = frame_to_all_lines.get(stem, [])
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("# Y = all detections, A = salient objects\n")
            f.write(f"# IMG_W {W_use} IMG_H {H_use}\n")
            for la in all_lines:
                f.write("Y " + la + "\n")
            for ls in attended_lines:
                f.write("A " + ls + "\n")
        exported += 1

    print(f"[{seq_id}] exported={exported} | elapsed={time.time() - start:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Annotate DR(eye)VE-SOD labels from saliency maps and YOLO12x detections.")
    p.add_argument("--start-seq", type=int, default=START_SEQ)
    p.add_argument("--end-seq", type=int, default=END_SEQ)
    p.add_argument("--weights", default=YOLO_WEIGHTS)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    for i in range(args.start_seq, args.end_seq + 1):
        process_sequence(f"{i:02d}", model)


if __name__ == "__main__":
    main()

