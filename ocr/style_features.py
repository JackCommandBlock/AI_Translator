"""从文字框像素提取视觉样式，并与 ASS Style 做相似度比较。

核心思路：对文字框做 KMeans 聚类得到主色，用边界像素估计背景色，再按亮度区分
文字与描边，输出可比较的 VisualStyle（文字色/描边色/粗体程度/字高/位置）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisualStyle:
    text_color: tuple[int, int, int] | None = None
    outline_color: tuple[int, int, int] | None = None
    font_height: float | None = None   # 像素
    bold: float | None = None          # 0..1，粗体程度
    region: tuple[float, float] | None = None  # (cx, cy) 归一化

    def to_dict(self) -> dict:
        return {
            "text_color": list(self.text_color) if self.text_color else None,
            "outline_color": list(self.outline_color) if self.outline_color else None,
            "font_height": self.font_height,
            "bold": self.bold,
            "region": list(self.region) if self.region else None,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "VisualStyle | None":
        if not d:
            return None

        def _color(v):
            if not v:
                return None
            try:
                return tuple(int(round(float(x))) for x in v)
            except (TypeError, ValueError):
                return None

        def _num(v):
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        region = d.get("region")
        if region:
            try:
                region = (float(region[0]), float(region[1]))
            except (TypeError, ValueError, IndexError):
                region = None
        return cls(
            text_color=_color(d.get("text_color")),
            outline_color=_color(d.get("outline_color")),
            font_height=_num(d.get("font_height")),
            bold=_num(d.get("bold")),
            region=region,
        )


def box_to_rect(box, w, h):
    if not box:
        return None
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
    except (TypeError, ValueError, IndexError):
        return None
    x0, x1 = max(0, int(min(xs))), min(w - 1, int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(h - 1, int(max(ys)) + 1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def extract_visual_style(frame, box, image_w, image_h) -> VisualStyle:
    vis = VisualStyle()
    rect = box_to_rect(box, image_w, image_h)
    if rect is None:
        return vis
    x0, y0, x1, y1 = rect
    crop = frame[y0:y1, x0:x1]
    vis.font_height = float(y1 - y0)
    vis.region = (
        (x0 + x1) / 2 / max(1, image_w),
        (y0 + y1) / 2 / max(1, image_h),
    )

    px = crop.reshape(-1, 3).astype(np.float32)
    if len(px) < 3:
        return vis
    k = min(3, max(1, len(px) // 20))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(px, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)

    def to_rgb(i):
        b, g, r = centers[i]
        return (int(r), int(g), int(b))

    # 边界像素近似背景色，离背景最近的簇视为背景
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
    bg = border.reshape(-1, 3).mean(axis=0)
    bg_idx = int(np.argmin(np.linalg.norm(centers - bg, axis=1)))
    others = [i for i in range(k) if i != bg_idx]
    if not others:
        vis.text_color = to_rgb(bg_idx)
        return vis

    hsv = cv2.cvtColor(
        centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
    )[0, :, 2]

    def brightness(i):
        return float(hsv[i])

    if len(others) == 1:
        text_idx = others[0]
        outline_idx = None
    else:
        ordered = sorted(others, key=brightness, reverse=True)
        text_idx = ordered[0]
        outline_idx = (
            ordered[-1] if brightness(ordered[-1]) < brightness(text_idx) - 20 else None
        )

    vis.text_color = to_rgb(text_idx)
    if outline_idx is not None:
        vis.outline_color = to_rgb(outline_idx)
    mask = np.isin(labels.reshape(-1), others).reshape(crop.shape[:2])
    vis.bold = float(mask.mean())  # 非背景像素占比，粗体笔画更粗
    return vis


def alignment_region(alignment: int) -> tuple[float, float] | None:
    if not 1 <= alignment <= 9:
        return None
    col = (alignment - 1) % 3          # 0左/1中/2右
    row = (alignment - 1) // 3         # 0下/1中/2上
    return ((0.25, 0.5, 0.75)[col], (0.8, 0.5, 0.2)[row])


def _color_sim(a, b):
    d = np.linalg.norm(np.array(a, float) - np.array(b, float))
    return max(0.0, 1.0 - d / 300.0)


def style_similarity(vis, ass_style, cfg) -> float:
    if vis is None or ass_style is None:
        return 0.0
    parts = []
    if vis.text_color is not None:
        parts.append(
            (_color_sim(vis.text_color, ass_style.primary), cfg.style_color_weight)
        )
    if vis.outline_color is not None:
        parts.append(
            (_color_sim(vis.outline_color, ass_style.outline), cfg.style_outline_weight)
        )
    if vis.bold is not None:
        parts.append(
            (1.0 - abs(float(ass_style.bold) - vis.bold), cfg.style_bold_weight)
        )
    exp = alignment_region(ass_style.alignment)
    if exp is not None and vis.region is not None:
        d = np.linalg.norm(np.array(vis.region) - np.array(exp))
        parts.append((max(0.0, 1.0 - d / 1.2), cfg.style_position_weight))
    if not parts:
        return 0.0
    total_w = sum(w for _, w in parts)
    if total_w <= 0.0:
        return 0.0
    return sum(s * w for s, w in parts) / total_w
