"""OCR 子模块：图像预处理、识别、多帧结果汇总。"""
from .preprocess import crop_subtitle_region, preprocess_image
from .recognizer import OCRResult, create_backend, get_available_backend
from .resolver import resolve_frames

__all__ = [
    "crop_subtitle_region",
    "preprocess_image",
    "OCRResult",
    "create_backend",
    "get_available_backend",
    "resolve_frames",
]

