"""基于多模态大模型的批量 OCR。

把多条字幕区域的截图纵向拼接成一张图，一次性识别多条，
降低接口调用次数。输出结构化 JSON 后按索引映射回事件。
"""
from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

from translate.client import chat_completion

from .recognizer import OCRResult
from .scene_text import (
    SCENE_TEXT_PROMPT,
    TextBlock,
    _parse_json,
    parse_blocks,
    resize_long_edge,
)


logger = logging.getLogger("pipeline.ocr.llm_vision")


def _to_png_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _stack_images(images: list[np.ndarray]) -> np.ndarray:
    """纵向拼接，条与条之间用白色细线分隔。"""
    sep = 6
    h, w = images[0].shape[:2]
    sep_rows = np.full((sep, w, 3), 255, dtype=np.uint8)
    parts = []
    for img in images:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        parts.append(img)
        parts.append(sep_rows)
    if parts:
        parts.pop()
    return np.vstack(parts)


class LLMVisionOCR:
    """把一组字幕截图拼接后交给多模态模型批量识别。"""

    def __init__(self, client, model: str, source_lang: str = "ja", batch_size: int = 8):
        self.client = client
        self.model = model
        self.source_lang = source_lang
        self.batch_size = batch_size

    def ocr_batch(self, images: list[np.ndarray]) -> list[OCRResult]:
        results: list[OCRResult] = [OCRResult("", None, "llm") for _ in images]
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            stacked = _stack_images(chunk)
            b64 = _to_png_b64(stacked)
            prompt = (
                f"这是一张按时间顺序纵向拼接的字幕截图，每两条字幕之间用白色横线分隔，"
                f"从上到下依次编号 0 到 {len(chunk) - 1}。"
                f"请逐条识别其中的{self.source_lang}字幕，"
                '只输出 JSON 数组，格式为 [{"i": 0, "text": "字幕原文"}, ...]，'
                "没有文字的条目 text 为空字符串。不要解释，不要加代码块标记。"
            )
            resp = chat_completion(
                self.client,
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
            )
            content = resp.choices[0].message.content or ""
            data = _parse_json(content)
            if isinstance(data, dict):
                data = data.get("items") or data.get("results") or [data]
            for item in data:
                if isinstance(item, dict):
                    idx = int(item.get("i", item.get("index", -1)))
                    if 0 <= idx < len(chunk):
                        results[start + idx] = OCRResult(
                            text=str(item.get("text", "")).strip(),
                            confidence=None,
                            backend="llm",
                        )
        return results

    def ocr_full_frame(self, image: np.ndarray) -> list[TextBlock]:
        """全帧多文本识别：返回画面中所有文字块及其内容类型。"""
        img = resize_long_edge(image, 1280)
        b64 = _to_png_b64(img)
        resp = chat_completion(
            self.client,
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCENE_TEXT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return parse_blocks(resp.choices[0].message.content or "")

