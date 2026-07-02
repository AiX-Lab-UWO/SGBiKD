from __future__ import annotations

from ablation._legacy import import_root_module


def _helper():
    return import_root_module("kd_dreyeve_compare_students_single_teacher")


def build_student(model_tag: str, weight_path: str):
    return _helper().build_student(model_tag, weight_path)


def validate_student_detect(student, tag: str):
    return _helper().validate_student_detect(student, tag)


def yolo_predict_students(model, img_bgr, class_mode, yolo_to_our_id, conf_th, max_det, mode):
    return _helper().yolo_predict_students(
        model,
        img_bgr,
        class_mode,
        yolo_to_our_id,
        conf_th,
        max_det,
        mode,
    )

