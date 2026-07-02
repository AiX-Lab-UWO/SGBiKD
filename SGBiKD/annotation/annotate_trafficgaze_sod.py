from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from paper_sgbikd._legacy import import_root_module


svit = import_root_module("trafficgaze_eval_dinov2_teacher_metrics")

TG_ROOT = Path(os.environ.get("KD_TG_ROOT", "/home/aixlab/datasets/TrafficGaze/TrafficGaze/Traffic_Gaze"))
TRAFFIC_ROOT = TG_ROOT / "trafficframe"
SALIENCY_ROOT = TG_ROOT / "saliencyframe"
FIXATION_ROOT = TG_ROOT / "fixationframe"
OUT_ROOT = Path(os.environ.get("KD_TG_YOLO12_ROOT", "/home/aixlab/datasets/TrafficGaze/Yolo12"))

TRAIN_JSON = os.environ.get("KD_TG_TRAIN_JSON", str(TG_ROOT / "train.json"))
VALID_JSON = os.environ.get("KD_TG_VALID_JSON", str(TG_ROOT / "valid.json"))
TEST_JSON = os.environ.get("KD_TG_TEST_JSON", str(TG_ROOT / "test.json"))

YOLO_WEIGHTS = os.environ.get("KD_TG_ANN_WEIGHTS", "yolo12x.pt")
YOLO_IMGSZ = int(os.environ.get("KD_TG_ANN_IMGSZ", "640"))
YOLO_DEVICE = os.environ.get("KD_TG_ANN_DEVICE", "0")
YOLO_HALF = os.environ.get("KD_TG_ANN_HALF", "1") != "0"
CONF_THRES = float(os.environ.get("KD_TG_ANN_CONF", "0.25"))
IOU_THRES = float(os.environ.get("KD_TG_ANN_IOU", "0.50"))
MAX_DET = int(os.environ.get("KD_TG_ANN_MAX_DET", "100"))

SAMPLE_EVERY = int(os.environ.get("KD_TG_SAMPLE_EVERY", "5"))
FIXATION_THRESHOLD = float(os.environ.get("KD_TG_FIX_COUNT_THRESHOLD", "5.0"))
TEMP_WINDOW = int(os.environ.get("KD_TG_TEMP_WINDOW", "5"))
MIN_FIXATED_FRAMES = int(os.environ.get("KD_TG_MIN_FIXATED_FRAMES", "5"))
N_PROPAGATE = int(os.environ.get("KD_TG_PROPAGATE", "60"))

YOLO_TO_TG_CLASS = {0: 0, 1: 1, 2: 1, 3: 2, 5: 5, 6: 6, 7: 6}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_split(path: str) -> List[str]:
    return svit.read_path_list_file(path)


def rel_video_key(rel: str) -> str:
    rel = svit._normalize_rel_path(rel)
    return rel.split("/")[0]


def rel_stem(rel: str) -> str:
    return os.path.splitext(os.path.basename(svit._normalize_rel_path(rel)))[0]


def resolve_image(rel: str) -> Optional[str]:
    return svit.resolve_existing_with_ext(str(TRAFFIC_ROOT), rel)


def resolve_saliency(rel: str) -> Optional[str]:
    return svit.resolve_existing_with_ext(str(SALIENCY_ROOT), rel, default_ext=".png")


def resolve_fixation(rel: str) -> Optional[str]:
    if not FIXATION_ROOT.is_dir():
        return None
    return svit.resolve_existing_with_ext(str(FIXATION_ROOT), rel, default_ext=".png")


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def natural_sort_key(path: str):
    base = os.path.basename(path)
    nums = "".join(ch if ch.isdigit() else " " for ch in base).split()
    return int(nums[-1]) if nums else base


def group_split_by_video(split_rels: Iterable[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for rel in split_rels:
        rel = svit._normalize_rel_path(rel)
        grouped[rel_video_key(rel)].append(rel)
    for vid in grouped:
        grouped[vid] = sorted(grouped[vid], key=natural_sort_key)
    return grouped


def parse_frame_id(rel: str) -> str:
    return rel_stem(rel)


def fixation_count_in_box(fix_map: np.ndarray, xyxy: List[float]) -> float:
    h, w = fix_map.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = fix_map[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
    if patch.max() <= 1.0:
        return float((patch > 0).sum())
    return float(patch.sum())


def bbox_xyxy_to_yolo_txt(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def video_output_dir(video_key: str) -> Path:
    try:
        vid = int(video_key)
        video_name = f"Video{vid}_salient_dataset_yolo12x"
    except Exception:
        video_name = f"{video_key}_salient_dataset_yolo12x"
    return OUT_ROOT / video_name


def export_support_files(video_key: str, rels: List[str]):
    out_dir = video_output_dir(video_key)
    for rel in rels:
        stem = parse_frame_id(rel)
        img_src = resolve_image(rel)
        sal_src = resolve_saliency(rel)
        if img_src is not None:
            dst = out_dir / "images" / f"{stem}{Path(img_src).suffix}"
            ensure_parent(dst)
            if not dst.exists():
                shutil.copy2(img_src, dst)
        if sal_src is not None:
            dst = out_dir / "saliency" / f"{stem}{Path(sal_src).suffix}"
            ensure_parent(dst)
            if not dst.exists():
                shutil.copy2(sal_src, dst)


def process_video(video_key: str, rels: List[str], model: YOLO):
    if not rels:
        return

    export_support_files(video_key, rels)
    out_dir = video_output_dir(video_key)
    label_dir = out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    frame_items = []
    for rel in rels:
        img_path = resolve_image(rel)
        sal_path = resolve_saliency(rel)
        if img_path is None or sal_path is None:
            continue
        fix_path = resolve_fixation(rel)
        frame_items.append((rel, img_path, sal_path, fix_path))
    frame_items.sort(key=lambda x: natural_sort_key(x[0]))
    if not frame_items:
        print(f"[TrafficGaze:{video_key}] no valid frames")
        return

    first_img = cv2.imread(frame_items[0][1], cv2.IMREAD_COLOR)
    if first_img is None:
        print(f"[TrafficGaze:{video_key}] failed to read first frame")
        return
    H0, W0 = first_img.shape[:2]

    frame_dir = os.path.dirname(frame_items[0][1])
    results_gen = model.track(
        source=frame_dir,
        stream=True,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
        half=bool(YOLO_HALF and YOLO_DEVICE != "cpu"),
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
    )

    tracked_by_stem: Dict[str, List[dict]] = {}
    names = None
    ordered_dir_files = sorted(
        [p for ext in IMG_EXTS for p in Path(frame_dir).glob(f"*{ext}")],
        key=lambda p: natural_sort_key(str(p)),
    )
    for path_obj, result in zip(ordered_dir_files, results_gen):
        if names is None:
            names = result.names
        dets = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(xyxy))
            for bb, score, cls_id, tid in zip(xyxy, conf, cls, ids):
                if int(cls_id) not in YOLO_TO_TG_CLASS:
                    continue
                x1, y1, x2, y2 = map(float, bb.tolist())
                x1 = max(0.0, min(W0 - 1.0, x1))
                x2 = max(0.0, min(W0 - 1.0, x2))
                y1 = max(0.0, min(H0 - 1.0, y1))
                y2 = max(0.0, min(H0 - 1.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                dets.append(
                    {
                        "track_id": int(tid),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "our_cls_id": int(YOLO_TO_TG_CLASS[int(cls_id)]),
                        "score": float(score),
                    }
                )
        tracked_by_stem[path_obj.stem] = dets

    valid_stems = {parse_frame_id(rel) for rel, *_ in frame_items}
    per_frame: List[List[dict]] = []
    fix_maps: List[np.ndarray] = []
    stem_order: List[str] = []
    for rel, _img_path, sal_path, fix_path in frame_items:
        stem = parse_frame_id(rel)
        if stem not in valid_stems:
            continue
        if fix_path is not None and os.path.isfile(fix_path):
            fix = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
        else:
            sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
            if sal is None:
                continue
            fix = (svit.derive_fixation_from_saliency(sal.astype(np.float32) / max(1.0, float(sal.max()))) * 255.0).astype(np.uint8)
        if fix is None:
            continue
        per_frame.append(tracked_by_stem.get(stem, []))
        fix_maps.append(fix)
        stem_order.append(stem)

    track_to_frames: Dict[int, List[int]] = defaultdict(list)
    is_candidate = [dict() for _ in range(len(per_frame))]
    for t, dets in enumerate(per_frame):
        for det in dets:
            track_to_frames[det["track_id"]].append(t)
            count = fixation_count_in_box(fix_maps[t], det["bbox_xyxy"])
            if count >= FIXATION_THRESHOLD:
                is_candidate[t][det["track_id"]] = True

    is_salient = [dict() for _ in range(len(per_frame))]
    for tid, frames_list in track_to_frames.items():
        for t in frames_list:
            r1 = is_candidate[t].get(tid, False)
            lo = max(0, t - TEMP_WINDOW)
            hi = min(len(per_frame) - 1, t + TEMP_WINDOW)
            count_fix = sum(1 for tau in range(lo, hi + 1) if is_candidate[tau].get(tid, False))
            r2 = count_fix >= MIN_FIXATED_FRAMES
            if r1 or r2:
                is_salient[t][tid] = True

    for tid, frames_list in track_to_frames.items():
        labeled = sorted([t for t in frames_list if is_salient[t].get(tid, False)])
        for i in range(len(labeled) - 1):
            t1, t2 = labeled[i], labeled[i + 1]
            if (t2 - t1) <= N_PROPAGATE:
                for tau in range(t1 + 1, t2):
                    is_salient[tau][tid] = True

    exported = 0
    for t in range(0, len(per_frame), SAMPLE_EVERY):
        stem = stem_order[t]
        img_path = resolve_image(rels[min(t, len(rels) - 1)])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) if img_path else None
        if img is None:
            continue
        H, W = img.shape[:2]
        y_lines = []
        a_lines = []
        for det in per_frame[t]:
            cls_id = det["our_cls_id"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy, bw, bh = bbox_xyxy_to_yolo_txt([x1, y1, x2, y2], W, H)
            line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            y_lines.append(line)
            if is_salient[t].get(det["track_id"], False):
                a_lines.append(line)
        out_lbl = label_dir / f"{stem}.txt"
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("# Y = all detections, A = salient objects\n")
            f.write(f"# IMG_W {W} IMG_H {H}\n")
            for line in y_lines:
                f.write("Y " + line + "\n")
            for line in a_lines:
                f.write("A " + line + "\n")
        exported += 1
    print(f"[TrafficGaze:{video_key}] exported={exported} labels to {label_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Annotate TrafficGaze-SOD labels using YOLO12x detections and fixation/saliency rules.")
    p.add_argument("--weights", default=YOLO_WEIGHTS)
    p.add_argument("--train-json", default=TRAIN_JSON)
    p.add_argument("--valid-json", default=VALID_JSON)
    p.add_argument("--test-json", default=TEST_JSON)
    return p.parse_args()


def main():
    args = parse_args()
    rels = read_split(args.train_json) + read_split(args.valid_json) + read_split(args.test_json)
    grouped = group_split_by_video(rels)
    model = YOLO(args.weights)
    for video_key, video_rels in grouped.items():
        process_video(video_key, video_rels, model)


if __name__ == "__main__":
    main()

