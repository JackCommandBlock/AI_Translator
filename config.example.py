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

    # 日志
    log_dir: str = "logs"
    log_file: str = "pipeline.log"
    log_level: str = "INFO"
    log_console: bool = True

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
    enable_positioning: bool = True  # 用 LLM 评分给译文字幕判别位置
    force_refresh: bool = False  # 忽略缓存，重新提取 / 识别

    # 字幕定位（enable_positioning=True 时生效）
    positioning_force_refresh: bool = False
    positioning_long_edge: int = 1280     # LLM 全帧 OCR 的长边缩放，用于坐标映射
    positioning_gap_min: float = 12.0     # 译文底边距原字幕顶边的最小间距
    positioning_gap_default: float = 36.0 # 默认间距
    positioning_gap_max: float = 80.0     # 最大间距
    positioning_top_margin: float = 24.0  # 屏幕上边保留
    positioning_side_margin: float = 48.0 # 屏幕左右保留
    positioning_reuse_bonus: float = 0.3   # 复用历史位置的评分加成
    positioning_max_candidates: int = 6
    positioning_history_cap: int = 8
    positioning_use_binary_search: bool = True  # 用二分查找确定纵向位置
    positioning_search_iterations: int = 9   # 二分查找迭代次数（完整高度范围）
    positioning_reuse_tolerance: float = 48.0  # 历史位置复用的距离容忍度
    positioning_workers: int = 4   # 字幕定位并发线程数

    # OCR / ASR 后端
    ocr_backend: str = "auto"      # auto | easyocr | rapidocr | tesseract | windows | llm
    ocr_llm_batch_size: int = 8    # llm 视觉 OCR 每张拼接图包含的字幕条数
    ocr_workers: int = 4           # 全帧 OCR（LLM 后端）并发线程数
    asr_backend: str = "auto"      # auto | faster_whisper | whisper | none
    asr_model_size: str = "small"  # tiny/base/small/medium/large-v3
    asr_device: str = "auto"       # auto / cpu / cuda

    # 阈值
    ocr_confidence_threshold: float = 0.6
    asr_confidence_threshold: float = 0.5

    # 字幕区域（已弃用：全帧检测不再依赖固定区域；仅 ocr_mode="crop" 兼容旧用法时使用）
    subtitle_region: tuple = (0.0, 0.85, 1.0, 1.0)

    # 多类型字幕识别（全帧检测）
    ocr_mode: str = "full_frame"          # full_frame=全帧检测；crop=兼容旧用法
    frame_sample_strategy: str = "union"  # union=事件首/中/尾采样点并集；midpoint=仅中点
    frames_per_event: int = 3             # 每事件采样密度
    asr_similarity_threshold: float = 0.6 # 文本与 ASR 相似度阈值
    text_track_gap_ms: int = 1500         # 相同文本拆分时间轨道的间隔
    use_ass_type_hint: bool = False       # Style/Layer/Effect 弱先验（默认关）

    # 样式识别一一对应
    use_style_match: bool = True          # 总开关
    style_match_weight: float = 0.8       # 样式相似度的得分权重
    style_similarity_min: float = 0.35    # 低于该值不给予样式加分
    style_color_weight: float = 1.0       # 主色权重
    style_outline_weight: float = 1.0     # 描边色权重
    style_bold_weight: float = 0.5        # 粗体权重（弱）
    style_position_weight: float = 0.3    # 位置权重（弱，仅作 tiebreak）

    # 对齐 / 分组
    gap_threshold_ms: int = 1200
    min_overlap_ratio: float = 0.3

    # 翻译
    batch_size: int = 20
    max_chars_per_line: int = 42
    max_lines: int = 2
    glossary_path: str = "glossary.json"
    glossary_dir: str = "glossaries"   # 多译名表目录；仅 glossary_name 非空时使用
    glossary_name: str = ""            # 从 glossary_dir 中选择译名表，如 "parako" -> glossaries/parako.json；留空则使用 glossary_path

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

    def glossary_file(self) -> Path:
        """返回实际生效的术语表文件路径。

        优先使用 ``glossary_name`` 在 ``glossary_dir`` 下按名称查找；
        未指定时回退到 ``glossary_path``。
        """
        name = getattr(self, "glossary_name", "")
        if name:
            if not name.endswith(".json"):
                name += ".json"
            return Path(getattr(self, "glossary_dir", "glossaries")) / name
        return Path(getattr(self, "glossary_path", "glossary.json"))

