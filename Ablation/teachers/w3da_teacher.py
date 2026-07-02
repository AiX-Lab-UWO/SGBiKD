from __future__ import annotations

import os

from ablation._legacy import import_root_module


W3DA_CKPT = os.environ.get(
    "KD_W3DA_CKPT",
    "/home/aixlab/Downloads/W3DA/runs/llada_bs2/ckpt_best.pt",
)


def _helper():
    return import_root_module("w3da_local_teacher_helper")


def build_teacher(ckpt_path: str | None = None):
    helper = _helper()
    teacher = helper.build_teacher(ckpt_path or W3DA_CKPT)
    teacher.kd_teacher_type = "w3da"
    return teacher


def teacher_saliency_map(teacher, img_bgr, gamma: float = 1.0):
    helper = _helper()
    return helper.teacher_saliency_map(teacher, img_bgr, gamma=gamma)

