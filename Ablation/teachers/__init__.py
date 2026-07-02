from .svit_teacher import SVIT_CKPT, build_teacher as build_svit_teacher
from .w3da_teacher import W3DA_CKPT, build_teacher as build_w3da_teacher

__all__ = [
    "SVIT_CKPT",
    "W3DA_CKPT",
    "build_svit_teacher",
    "build_w3da_teacher",
]

