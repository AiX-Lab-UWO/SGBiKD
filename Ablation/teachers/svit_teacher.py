from __future__ import annotations

import os

from ablation._legacy import import_root_module


SVIT_CKPT = os.environ.get(
    "KD_SVIT_CKPT",
    "/home/aixlab/PyCharmMiscProject/KD/teacher_saliency_dino_224.pth",
)


def _helper():
    return import_root_module("kd_dreyeve_compare_teachers_yolo12x")


def build_teacher(ckpt_path: str | None = None):
    helper = _helper()
    helper.SVIT_CKPT = ckpt_path or SVIT_CKPT
    teacher = helper.build_svit_teacher()
    teacher.kd_teacher_type = getattr(teacher, "kd_teacher_type", "svit")
    return teacher


def teacher_saliency_map(teacher, img_bgr, gamma: float = 1.0):
    helper = _helper()
    return helper.scf_vit_saliency_map(teacher, img_bgr, gamma=gamma)

