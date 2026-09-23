"""全局配置模板。

首次运行时会自动把本文件复制为 config.py（config.py 已加入 .gitignore，
不会被提交）。请复制后按需修改其中的路径与 API Key。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # 输入 / 输出
    video_path: str = "sample.mp4"
    ass_path: str = "sample.ass"
    output_path: str = "translated.ass"
    work_dir: str = "work"

    # 语言
    source_lang: str = "ja"
    target_lang: str = "zh-CN"

    api_base_url: str = "" # 请填写你的 BASE url
    api_key: str = ""  # 请填写你的 API Key
    model: str = "" # 你的模型名称
    llm_temperature: float = 0.2

    # 模块开关
    enable_ocr: bool = True
    enable_asr: bool = True
    enable_translate: bool = True
    enable_review: bool = True
    force_refresh: bool = False  # 忽略缓存，重新提取 / 识别

    # OCR / ASR 后端
    ocr_backend: str = "auto"      # auto | easyocr | rapidocr | tesseract | windows | llm
    ocr_llm_batch_size: int = 8    # llm 视觉 OCR 每张拼接图包含的字幕条数
    asr_backend: str = "auto"      # auto | faster_whisper | whisper | none
    asr_model_size: str = "small"  # tiny/base/small/medium/large-v3
    asr_device: str = "auto"       # auto / cpu / cuda

    # 阈值
    ocr_confidence_threshold: float = 0.6
    asr_confidence_threshold: float = 0.5

    # 字幕区域（归一化坐标：x0, y0, x1, y1）
    subtitle_region: tuple = (0.0, 0.85, 1.0, 1.0)
    frames_per_event: int = 3

    # 对齐 / 分组
    gap_threshold_ms: int = 1200
    min_overlap_ratio: float = 0.3

    # 翻译
    batch_size: int = 20
    max_chars_per_line: int = 42
    max_lines: int = 2
    glossary_path: str = "glossary.json"

    # 其他
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    extra: dict = field(default_factory=dict)

    @staticmethod
    def load(path: str | None = None) -> "Config":
        cfg = Config()
        if path and Path(path).exists():
            import json
            raw = Path(path).read_text(encoding="utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()
            for k, v in data.items():
                if hasattr(cfg, k):
                    cur = getattr(cfg, k)
                    if isinstance(cur, bool):
                        setattr(cfg, k, str(v).lower() in ("1", "true", "yes", "on"))
                    elif isinstance(cur, int):
                        setattr(cfg, k, int(v))
                    elif isinstance(cur, float):
                        setattr(cfg, k, float(v))
                    else:
                        setattr(cfg, k, v)
        env_key = os.environ.get("AI_TRANSLATOR_API_KEY")
        if env_key:
            cfg.api_key = env_key
        return cfg

    def resolve(self, path: str) -> Path:
        return Path(path)

