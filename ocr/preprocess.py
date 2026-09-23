"""字幕区域裁剪与图像预处理。"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _load_image(frame_path) -> np.ndarray:
    if isinstance(frame_path, np.ndarray):
        return frame_path
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {frame_path}")
    return img


def crop_subtitle_region(frame, region: tuple | None = None) -> np.ndarray:
    """按归一化区域 (x0,y0,x1,y1) 裁剪画面。默认使用底部区域。"""
    img = _load_image(frame)
    h, w = img.shape[:2]
    if region is None:
        region = (0.0, 0.62, 1.0, 1.0)
    x0, y0, x1, y1 = region
    left = int(w * x0)
    right = int(w * x1)
    top = int(h * y0)
    bottom = int(h * y1)
    return img[top:bottom, left:right]


def preprocess_image(
    frame,
    region: tuple | None = None,
    upscale: float = 1.5,
    use_otsu: bool = False,
) -> np.ndarray:
    """裁剪 + 灰度化 + 缩放，返回便于 OCR 的图像。"""
    crop = crop_subtitle_region(frame, region)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)
    if use_otsu:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray


def save_image(array: np.ndarray, path) -> str:
    Image.fromarray(array).save(path)
    return str(path)

