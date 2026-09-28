"""OCR 子模块：图像预处理、识别、全帧多文本检测、多帧结果汇总。"""
from .preprocess import crop_subtitle_region, preprocess_image
from .recognizer import DetectedBlock, OCRResult, create_backend, get_available_backend
from .resolver import resolve_frames
from .scene_text import (
    ContentTypeClassifier,
    FrameOCR,
    LocalSceneTextOCR,
    SceneTextOCR,
    TextBlock,
    normalize_type,
    parse_blocks,
    resize_long_edge,
)
from .style_features import VisualStyle, extract_visual_style, style_similarity

__all__ = [
    "crop_subtitle_region",
    "preprocess_image",
    "OCRResult",
    "DetectedBlock",
    "create_backend",
    "get_available_backend",
    "resolve_frames",
    "TextBlock",
    "FrameOCR",
    "SceneTextOCR",
    "LocalSceneTextOCR",
    "ContentTypeClassifier",
    "normalize_type",
    "parse_blocks",
    "resize_long_edge",
    "VisualStyle",
    "extract_visual_style",
    "style_similarity",
]

