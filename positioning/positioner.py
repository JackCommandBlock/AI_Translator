"""基于「模拟摆放 + LLM 评分」的字幕定位。

工作流程
--------
1. 从 ``frame_ocr.json`` / ``text_timeline.json`` / ``matched.json`` 中恢复每个
   ASS 事件对应的原字幕包围盒，并把坐标从 OCR 画布映射到 PlayRes 坐标。
2. 判断该字幕是否已位于默认位置：是则不加任何 ``\pos``，保持原样。
3. 否则用 ffmpeg 把译文烧录到事件代表帧上，用多模态 LLM 作为“是否压到原字幕”的
   判定器，对纵向位置做二分查找，找到刚好不重叠的边界后再上移一个合适间距。
   （``positioning_use_binary_search=False`` 时退回旧的“多候选 + 打分”模式。）
4. 在满足“位于原字幕上方、不越界”的前提下，优先复用之前已使用过的历史锚点。
5. 把最终锚点写成 ``{\\an2\\pos(x,y)}`` 插入译文 Text 头部。

说明：横坐标固定使用字幕样式的默认横坐标（画面水平居中），只识别纵向位置；
这里不修改原文 ASS 的样式与时间轴，只给译文追加 override 标签。
"""
from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from align.text_timeline import text_similarity
from ass.styles import parse_styles
from translate.client import chat_completion
from video.ffmpeg import Ffmpeg
from video.frame_extract import extract_frames_at_times


logger = logging.getLogger("pipeline.positioning")

NEWLINE_TOKEN = "\u23ce"


@dataclass
class BBox:
    """轴对齐包围盒（PlayRes 坐标）。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class Candidate:
    x: float
    y: float
    origin: str = "ideal"  # ideal | reuse | below
    reuse: bool = False


@dataclass
class PlacementProposal:
    """并行阶段产出的、与历史位置无关的定位建议。"""

    x: float
    y: float
    boundary_y: float


@dataclass
class PositionParams:
    playres_x: int
    playres_y: int
    gap_min: float
    gap_default: float
    gap_max: float
    top_margin: float
    side_margin: float
    reuse_bonus: float
    max_candidates: int
    history_cap: int
    use_binary_search: bool
    search_iterations: int
    reuse_tolerance: float

    @classmethod
    def from_cfg(cls, cfg, playres_x: int, playres_y: int) -> "PositionParams":
        return cls(
            playres_x=int(playres_x),
            playres_y=int(playres_y),
            gap_min=float(getattr(cfg, "positioning_gap_min", 12.0)),
            gap_default=float(getattr(cfg, "positioning_gap_default", 36.0)),
            gap_max=float(getattr(cfg, "positioning_gap_max", 80.0)),
            top_margin=float(getattr(cfg, "positioning_top_margin", 24.0)),
            side_margin=float(getattr(cfg, "positioning_side_margin", 48.0)),
            reuse_bonus=float(getattr(cfg, "positioning_reuse_bonus", 0.3)),
            max_candidates=int(getattr(cfg, "positioning_max_candidates", 6)),
            history_cap=int(getattr(cfg, "positioning_history_cap", 8)),
            use_binary_search=bool(getattr(cfg, "positioning_use_binary_search", True)),
            search_iterations=int(getattr(cfg, "positioning_search_iterations", 9)),
            reuse_tolerance=float(getattr(cfg, "positioning_reuse_tolerance", 48.0)),
        )


class PositionHistory:
    """记录已使用过的纵向锚点，供后续字幕优先复用（横坐标固定为样式默认值）。"""

    def __init__(self, quantize: float = 8.0):
        self.quantize = float(quantize)
        self._counts: dict[tuple[int], int] = {}
        self._order: list[tuple[int]] = []

    def _key(self, x: float, y: float) -> tuple[int]:
        q = self.quantize
        return (int(round(y / q) * q),)

    def add(self, x: float, y: float) -> None:
        key = self._key(x, y)
        if key not in self._counts:
            self._order.append(key)
        self._counts[key] = self._counts.get(key, 0) + 1

    def top(self, n: int) -> list[float]:
        """返回最常复用的纵向锚点（升序去重后的值）。"""
        ordered = sorted(self._order, key=lambda k: (-self._counts[k], self._order.index(k)))[:n]
        return [float(k[0]) for k in ordered]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _loads(text: str):
    """从 LLM 返回文本中尽量解析出 JSON。"""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    candidates = [t]
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        c = re.sub(r",\s*([}\]])", r"\1", c)
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法解析 JSON: {t[:160]!r}")


def _ass_lines(text: str) -> list[str]:
    t = (text or "").replace(NEWLINE_TOKEN, "\n").replace("\\N", "\n").replace("\\n", "\n")
    lines = t.split("\n")
    return [line for line in lines] if any(line.strip() for line in lines) else [""]


def estimate_text_size(text: str, style) -> tuple[float, float]:
    """粗略估计译文在 PlayRes 中的宽高，仅用于候选合法性过滤。"""
    lines = _ass_lines(text)
    fontsize = max(1.0, float(getattr(style, "fontsize", 20.0) or 20.0))
    line_height = fontsize * 1.25
    max_chars = max((len(line) for line in lines), default=1)
    width = max_chars * fontsize * 0.95
    return max(1.0, width), line_height * max(1, len(lines))


def _read_playres(ass, video_w: int, video_h: int) -> tuple[int, int]:
    rx = ry = None
    for line in ass.header_lines:
        m = re.match(r"\s*PlayResX\s*:\s*(\d+)", line, re.IGNORECASE)
        if m:
            rx = int(m.group(1))
        m = re.match(r"\s*PlayResY\s*:\s*(\d+)", line, re.IGNORECASE)
        if m:
            ry = int(m.group(1))
    return rx or video_w, ry or video_h


def _video_dimensions(video_path: str, ffprobe_bin: str = "ffprobe") -> tuple[int, int]:
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 读取视频失败: {proc.stderr.decode('utf-8', 'replace')}")
    data = json.loads(proc.stdout.decode("utf-8", "replace"))
    for s in data.get("streams") or []:
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        if w and h:
            return w, h
    raise RuntimeError("无法从视频中读取画面尺寸")


def _ocr_canvas(cfg, video_w: int, video_h: int) -> tuple[int, int]:
    """返回 frame_ocr.json 中 box 所在的画布尺寸。"""
    try:
        from ocr.recognizer import get_available_backend

        backend = get_available_backend(cfg)
    except Exception:
        backend = "llm"
    if backend != "llm":
        return video_w, video_h
    long_edge = int(getattr(cfg, "positioning_long_edge", 1280) or 1280)
    scale = long_edge / max(video_w, video_h)
    if scale >= 1.0:
        return video_w, video_h
    return max(1, int(round(video_w * scale))), max(1, int(round(video_h * scale)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def collect_boxes_by_track(frame_ocrs, tracks) -> dict[int, list[list]]:
    """把逐帧文字块按文本相似度归到 text track，返回 {track_key: [box, ...]}。"""
    result: dict[int, list[list]] = {t.key: [] for t in tracks}
    for fo in frame_ocrs:
        for b in fo.blocks:
            box = getattr(b, "box", None)
            if not box:
                continue
            best_key = None
            best_sim = 0.0
            for t in tracks:
                sim = text_similarity(b.text, t.text)
                if sim > best_sim:
                    best_sim = sim
                    best_key = t.key
            if best_key is not None and best_sim >= 0.75:
                result[best_key].append(box)
    return result


def _aggregate_box(boxes: list[list]) -> BBox | None:
    if not boxes:
        return None
    x1s, y1s, x2s, y2s = [], [], [], []
    for box in boxes:
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
        except (TypeError, ValueError, IndexError):
            continue
        x1s.append(min(xs))
        y1s.append(min(ys))
        x2s.append(max(xs))
        y2s.append(max(ys))
    if not x1s:
        return None
    return BBox(_median(x1s), _median(y1s), _median(x2s), _median(y2s))


def map_box_to_playres(box: BBox, ocr_w: int, ocr_h: int, playres_x: int, playres_y: int) -> BBox:
    sx = playres_x / max(1, ocr_w)
    sy = playres_y / max(1, ocr_h)
    return BBox(box.x1 * sx, box.y1 * sy, box.x2 * sx, box.y2 * sy)


def style_default_anchor(style, playres_x: int, playres_y: int) -> tuple[float, float]:
    """根据 Style 的 Alignment/MarginV 估算默认锚点。

    横坐标统一取画面水平居中（即字幕样式的默认横坐标），不再根据原字幕的
    OCR 包围盒识别横坐标；纵向坐标仍按对齐方式与 MarginV 计算。
    """
    x = playres_x / 2.0
    al = int(getattr(style, "alignment", 2) or 2)
    margin_v = float(getattr(style, "margin_v", 0) or 0)
    if al in (1, 2, 3):  # 底部对齐（\an 编号：1 左下 / 2 下中 / 3 右下）
        y = playres_y - margin_v
    elif al in (4, 5, 6):  # 垂直居中
        y = playres_y / 2.0
    else:  # 顶部对齐
        y = margin_v
    return x, y


def _on_screen(c: Candidate, width: float, height: float, params: PositionParams) -> bool:
    return (
        params.side_margin <= c.x - width / 2
        and c.x + width / 2 <= params.playres_x - params.side_margin
        and params.top_margin <= c.y - height
        and c.y <= params.playres_y - params.side_margin
    )


def generate_candidates(
    source: BBox,
    default_anchor: tuple[float, float],
    style,
    text: str,
    history: PositionHistory,
    params: PositionParams,
) -> list[Candidate]:
    """按需求生成候选锚点（使用 ``\\an2``，即锚点为译文底边中心）。

    所有候选的横坐标固定为字幕样式的默认横坐标，仅纵向位置不同。
    """
    est_w, est_h = estimate_text_size(text, style)
    x = default_anchor[0]
    top = source.y1
    out: list[Candidate] = []

    gaps: list[float] = []
    for g in (params.gap_default, params.gap_min, params.gap_max):
        if g not in gaps:
            gaps.append(g)
    for gap in gaps:
        c = Candidate(x, top - gap, "ideal", False)
        if _on_screen(c, est_w, est_h, params):
            out.append(c)

    # 优先复用历史锚点：横坐标保持样式默认值，只复用纵向位置，且必须仍在原字幕上方且不越界
    for hy in history.top(params.history_cap):
        if hy > top - params.gap_min:
            continue
        c = Candidate(x, float(hy), "reuse", True)
        if _on_screen(c, est_w, est_h, params):
            out.append(c)

    # 上方没有可用位置时，退化为放在原字幕下方
    if not out:
        c = Candidate(x, source.y2 + params.gap_default, "below", False)
        if _on_screen(c, est_w, est_h, params):
            out.append(c)

    # 去重并限制数量（保持顺序：ideal 优先，reuse 最后）
    seen: set[tuple[int, int]] = set()
    unique: list[Candidate] = []
    for c in out:
        key = (int(round(c.x / 8.0) * 8), int(round(c.y / 8.0) * 8))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique[: params.max_candidates]


def build_position_tag(x: float, y: float, alignment: int = 2) -> str:
    return "{\\an%d\\pos(%d,%d)}" % (alignment, int(round(x)), int(round(y)))


def apply_positions(translations: dict[int, str], positions: dict[int | str, str]) -> dict[int, str]:
    """把 ``\\pos`` 标签插到译文 Text 头部。"""
    out = dict(translations)
    for ev_id, tag in positions.items():
        key = int(ev_id)
        if tag and key in out and out[key].strip():
            out[key] = tag + out[key]
    return out


def _escape_filter_path(path: Path) -> str:
    s = str(path.resolve()).replace("\\", "/")
    # ffmpeg 的 filter 参数解析会先把 ``\\`` 还原成 ``\``，因此驱动盘符后的冒号
    # 需要写成 ``\\:``，否则会被当作 filter 选项分隔符。
    return s.replace(":", "\\\\:")


def _write_candidate_ass(ass, ev, text: str, tag: str, path: Path) -> None:
    lines: list[str] = []
    for line in ass.header_lines:
        if line.strip().lower().startswith("dialogue:"):
            continue
        lines.append(line)
    visible = (text or "").replace(NEWLINE_TOKEN, "\\N")
    # 烧录到单帧 PNG 时，输入帧时间戳为 0；必须把事件时间改成覆盖 0，
    # 否则保留原时间轴的 Dialogue 不会被激活，字幕不会显示。
    prefix = ev.raw_prefix
    if ev.end_raw:
        prefix = prefix.replace(ev.end_raw, "0:00:10.00", 1)
    if ev.start_raw:
        prefix = prefix.replace(ev.start_raw, "0:00:00.00", 1)
    lines.append(f"{prefix}{tag}{visible}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8-sig", newline="")


def render_overlay(frame_path: Path, ass_path: Path, out_path: Path, ffmpeg: Ffmpeg) -> Path:
    """把临时 ASS 烧录到帧上，返回输出图路径。"""
    escaped = _escape_filter_path(ass_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for filt in (f"ass={escaped}", f"subtitles={escaped}"):
        try:
            ffmpeg.run_checked(
                ["-y", "-i", str(frame_path), "-vf", filt, "-frames:v", "1", str(out_path)]
            )
            return out_path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"渲染字幕失败: {last_err}")


def _to_png_b64(image: Image.Image) -> str:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_file_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_contact_sheet(paths: list[Path], labels: list[str], tile_width: int = 360) -> Image.Image:
    """把参考帧与候选渲染图横向拼成一张图，左上角标注标签。"""
    imgs: list[Image.Image] = []
    for p, label in zip(paths, labels):
        im = Image.open(p).convert("RGB")
        scale = tile_width / max(1, im.width)
        im = im.resize((tile_width, max(1, int(round(im.height * scale)))))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 64, 30], fill=(255, 255, 255))
        d.text((6, 6), label, fill=(0, 0, 0))
        imgs.append(im)
    height = max(im.height for im in imgs)
    sheet = Image.new("RGB", (tile_width * len(imgs), height), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * tile_width, 0))
    return sheet


SCORING_PROMPT = (
    "图中第一张 REF 是原始视频帧，后面 A/B/C… 是把候选翻译字幕烧录到同一帧后的效果。"
    "请从可读性、是否遮挡原字幕或画面重要信息、是否保持字幕位置稳定、是否贴近原字幕"
    "但不重叠等维度，给每个候选打分。分数 0~10，10 为最佳。只输出 JSON 数组，"
    '格式：[{"id":"A","score":8,"reason":"..."}]，不要输出其它内容。'
)

POSITION_CLASSIFY_PROMPT = (
    "第一张 REF 是原始视频帧，第二张 CAND 是在同一帧上叠加了一条翻译字幕后的效果。"
    "请判断 CAND 中新出现的翻译字幕相对于原画面日文字幕的位置，只输出 JSON："
    '{"position":"above"}、{"position":"overlap"} 或 {"position":"below"}。'
    "above=完全在原字幕上方；overlap=与原字幕重叠或遮挡；below=完全在原字幕下方。"
    "不要输出其它内容。"
)

DEFAULT_POSITION_PROMPT = (
    "这张图是原始视频帧。请只判断：画面中的原字幕是否大致位于屏幕正下方、"
    "水平居中的标准字幕默认位置（即无需再为翻译字幕单独移动）。"
    "若位于画面中部、顶部、靠左、靠右或其它非默认位置，应返回 false。"
    '只输出 JSON：{"default":true} 或 {"default":false}，不要输出其它内容。'
)


class LLMScorer:
    """多模态 LLM 评分器。"""

    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client
        self.model = getattr(cfg, "model", "")

    def score(self, ref_path: Path, candidate_paths: list[Path]) -> dict[str, float]:
        if not candidate_paths:
            return {}
        labels = [chr(ord("A") + i) for i in range(len(candidate_paths))]
        sheet = make_contact_sheet([ref_path] + candidate_paths, ["REF"] + labels)
        b64 = _to_png_b64(sheet)
        resp = chat_completion(
            self.client,
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCORING_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        content = resp.choices[0].message.content or ""
        data = _loads(content)
        return self._parse_scores(data, labels)

    def _parse_scores(self, data, labels: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("scores") or data.get("results") or [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            id_ = str(item.get("id") or item.get("label") or "").strip().upper()
            if id_ not in labels:
                continue
            try:
                scores[id_] = float(item.get("score"))
            except (TypeError, ValueError):
                continue
        return scores

    def _ask_image_b64(self, b64: str, prompt: str, key: str) -> bool:
        resp = chat_completion(
            self.client,
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        content = resp.choices[0].message.content or ""
        value = None
        try:
            data = _loads(content)
        except Exception:
            data = None
        if isinstance(data, dict):
            value = data.get(key)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            value = data[0].get(key)
        elif isinstance(data, str):
            value = data
        if value is not None:
            if isinstance(value, bool):
                return value
            low = str(value).strip().lower()
            if low in ("true", "1", "yes", "是"):
                return True
            if low in ("false", "0", "no", "否"):
                return False
        # 模型偶尔不返回合法 JSON，用正则兜底
        m = re.search(rf'"{key}"\s*:\s*(true|false)', content, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
        m = re.search(r"\b(true|false)\b", content, re.IGNORECASE)
        return bool(m and m.group(1).lower() == "true")

    def _ask_image_bool_pil(self, image: Image.Image, prompt: str, key: str) -> bool:
        return self._ask_image_b64(_to_png_b64(image), prompt, key)

    def _ask_image_bool(self, image_path: Path, prompt: str, key: str) -> bool:
        return self._ask_image_b64(_image_file_to_b64(image_path), prompt, key)

    def classify_position(self, ref_path: Path, cand_path: Path) -> str:
        sheet = make_contact_sheet([ref_path, cand_path], ["REF", "CAND"])
        b64 = _to_png_b64(sheet)
        resp = chat_completion(
            self.client,
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": POSITION_CLASSIFY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        content = resp.choices[0].message.content or ""
        value = None
        try:
            data = _loads(content)
        except Exception:
            data = None
        if isinstance(data, dict):
            value = data.get("position")
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            value = data[0].get("position")
        elif isinstance(data, str):
            value = data
        if value is not None:
            low = str(value).strip().lower()
            if low in ("above", "上", "上方"):
                return "above"
            if low in ("overlap", "重叠", "遮挡", "压到"):
                return "overlap"
            if low in ("below", "下", "下方"):
                return "below"
        m = re.search(r'"position"\s*:\s*"(above|overlap|below)"', content, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        m = re.search(r"\b(above|overlap|below)\b", content, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return "overlap"

    def is_default_position(self, image_path: Path) -> bool:
        return self._ask_image_bool(image_path, DEFAULT_POSITION_PROMPT, "default")


class PositionEngine:
    def __init__(self, cfg, ass, work_dir: Path):
        self.cfg = cfg
        self.ass = ass
        self.work = Path(work_dir)
        self.ffmpeg = Ffmpeg(
            getattr(cfg, "ffmpeg_bin", "ffmpeg"),
            getattr(cfg, "ffprobe_bin", "ffprobe"),
        )
        self.scorer: LLMScorer | None = None

    def _client(self):
        if self.scorer is None:
            from translate.translator import build_client

            self.scorer = LLMScorer(self.cfg, build_client(self.cfg))
        return self.scorer

    def _load_translations(self) -> dict[int, str]:
        p = self.work / "translated.json"
        if not p.exists():
            return {}
        data = _read_json(p)
        return {int(k): (v or "") for k, v in data.items()}

    def _load_frame_ocrs(self):
        from ocr.scene_text import FrameOCR

        p = self.work / "frame_ocr.json"
        if not p.exists():
            return []
        return [FrameOCR.from_dict(d) for d in _read_json(p)]

    def _load_tracks(self):
        from align.text_timeline import TextTrack

        p = self.work / "text_timeline.json"
        if not p.exists():
            return []
        return [TextTrack.from_dict(d) for d in _read_json(p)]

    def _load_matched(self) -> dict[int, int]:
        p = self.work / "matched.json"
        if not p.exists():
            return {}
        return {int(k): int(v) for k, v in _read_json(p).items()}

    def _event_frame(self, ev) -> Path | None:
        out = self.work / "position_frames"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"ev{ev.id:04d}_0000.png"
        if fp.exists():
            return fp
        mid = ev.start_ms + ev.duration_ms // 2
        paths = extract_frames_at_times(self.cfg.video_path, [mid], out, prefix=f"ev{ev.id:04d}")
        return paths[0] if paths else None

    def _choose(self, ev, candidates: list[Candidate], frame_path: Path, text: str, params: PositionParams) -> Candidate:
        render_dir = self.work / "position_render"
        render_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.work / "position_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, c in enumerate(candidates):
            tag = build_position_tag(c.x, c.y)
            tmp_ass = tmp_dir / f"ev{ev.id:04d}_{i}.ass"
            _write_candidate_ass(self.ass, ev, text, tag, tmp_ass)
            out = render_dir / f"ev{ev.id:04d}_{i}.png"
            render_overlay(frame_path, tmp_ass, out, self.ffmpeg)
            paths.append(out)

        scores: dict[str, float] = {}
        try:
            scores = self._client().score(frame_path, paths)
        except Exception as exc:  # noqa: BLE001
            logger.warning("事件 %s 评分失败，使用确定性回退：%s", ev.id, exc)

        best = candidates[0]
        best_score = -1.0
        for i, c in enumerate(candidates):
            label = chr(ord("A") + i)
            s = scores.get(label, 0.0) + (params.reuse_bonus if c.reuse else 0.0)
            if s > best_score + 1e-6:
                best_score = s
                best = c
        return best

    def _render_candidate(self, ev, frame_path: Path, text: str, x: float, y: float, out_path: Path) -> None:
        tag = build_position_tag(x, y)
        tmp = self.work / "position_tmp" / f"ev{ev.id:04d}_probe_{int(round(x))}_{int(round(y))}.ass"
        _write_candidate_ass(self.ass, ev, text, tag, tmp)
        render_overlay(frame_path, tmp, out_path, self.ffmpeg)

    def _classify_at(self, ev, frame_path: Path, text: str, x: float, y: float) -> str:
        out = self.work / "position_render" / f"ev{ev.id:04d}_probe_{int(round(x))}_{int(round(y))}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        self._render_candidate(ev, frame_path, text, x, y, out)
        return self._client().classify_position(frame_path, out)

    def _is_default(self, ev, frame_path: Path) -> bool:
        """默认位置与内容类型/样式无关，每次只做一次 LLM 识图判断。"""
        try:
            return self._client().is_default_position(frame_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("事件 %s LLM 默认位置判断失败，按非默认处理：%s", ev.id, exc)
            return False

    def _binary_search_y(
        self,
        ev,
        frame_path: Path,
        text: str,
        x: float,
        source: BBox,
        style,
        params: PositionParams,
    ) -> tuple[float, float]:
        """在完整高度范围 [0, PlayResY] 上二分查找“仍位于原字幕上方”的边界。"""
        lo = 0.0
        hi = float(params.playres_y)

        # 端点不调用 LLM：y=0 时译文在画面顶部（必在原字幕上方），
        # y=PlayResY 时译文在画面底部（默认位置之外的场景必在原字幕下方）。
        # 判定“是否在原字幕上方”随 y 增大单调地从 true 变为 false。
        for _ in range(max(1, params.search_iterations)):
            mid = (lo + hi) / 2.0
            try:
                label = self._classify_at(ev, frame_path, text, x, mid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("事件 %s 二分探测失败，使用当前区间上界：%s", ev.id, exc)
                label = "overlap"
            if label == "above":
                lo = mid
            else:
                hi = mid

        boundary = lo
        # 二分结果退到端点说明判定全程同向，改为确定性回退
        if boundary <= 1e-6 or boundary >= float(params.playres_y) - 1e-6:
            return self._fallback_default(source, params)
        final_y = boundary - max(params.gap_default, params.gap_min)
        final_y = min(max(final_y, 0.0), float(params.playres_y))
        return final_y, boundary

    def _fallback_below(self, source: BBox, params: PositionParams) -> tuple[float, float]:
        """原字幕贴顶、上方放不下时，退化为放到原字幕下方。"""
        y = min(max(source.y2 + params.gap_default, 0.0), float(params.playres_y))
        return y, 0.0

    def _fallback_default(self, source: BBox, params: PositionParams) -> tuple[float, float]:
        y = min(max(source.y1 - params.gap_default, 0.0), float(params.playres_y))
        return y, source.y1

    def _nearest_reuse(
        self,
        default_x: float,
        ideal_y: float,
        boundary_y: float,
        est_w: float,
        est_h: float,
        history: PositionHistory,
        params: PositionParams,
    ) -> tuple[float, float] | None:
        best: tuple[float, float] | None = None
        best_d2: float | None = None
        for hy in history.top(params.history_cap):
            if hy > boundary_y - params.gap_min:
                continue
            if not _on_screen(Candidate(default_x, float(hy), "reuse", True), est_w, est_h, params):
                continue
            # 横坐标固定为样式默认值，复用历史位置时只比较纵向距离。
            d2 = (hy - ideal_y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (default_x, float(hy))
        if best is not None and best_d2 is not None and best_d2 <= params.reuse_tolerance ** 2:
            return best
        return None

    def _propose_binary(
        self,
        ev,
        frame_path: Path,
        text: str,
        source: BBox,
        default_anchor: tuple[float, float],
        style,
        params: PositionParams,
    ) -> PlacementProposal:
        est_w, _ = estimate_text_size(text, style)
        # 横坐标固定使用字幕样式的默认坐标，仅识别纵向位置。
        x = default_anchor[0]
        half_w = est_w / 2.0
        min_x = params.side_margin + half_w
        max_x = params.playres_x - params.side_margin - half_w
        x = params.playres_x / 2.0 if max_x < min_x else min(max(x, min_x), max_x)
        final_y, boundary_y = self._binary_search_y(ev, frame_path, text, x, source, style, params)
        return PlacementProposal(x=x, y=final_y, boundary_y=boundary_y)

    def _place_event_binary(
        self,
        ev,
        frame_path: Path,
        text: str,
        source: BBox,
        default_anchor: tuple[float, float],
        style,
        history: PositionHistory,
        params: PositionParams,
    ) -> Candidate:
        est_w, est_h = estimate_text_size(text, style)
        proposal = self._propose_binary(ev, frame_path, text, source, default_anchor, style, params)
        reused = self._nearest_reuse(
            default_anchor[0], proposal.y, proposal.boundary_y, est_w, est_h, history, params
        )
        if reused is not None:
            return Candidate(reused[0], reused[1], "reuse", True)
        return Candidate(proposal.x, proposal.y, "ideal", False)

    def run(self, translations: dict[int, str] | None = None) -> dict[int, str]:
        cache = self.work / "positions.json"
        if cache.exists() and not getattr(self.cfg, "positioning_force_refresh", False):
            logger.info("读取字幕位置缓存 positions.json")
            data = _read_json(cache)
            return {int(k): (v or "") for k, v in data.items()}

        if translations is None:
            translations = self._load_translations()
        frame_ocrs = self._load_frame_ocrs()
        tracks = self._load_tracks()
        matched = self._load_matched()
        if not frame_ocrs or not tracks or not matched:
            logger.warning("缺少 frame_ocr/text_timeline/matched 缓存，跳过字幕定位")
            return {}

        video_w, video_h = _video_dimensions(
            self.cfg.video_path, getattr(self.cfg, "ffprobe_bin", "ffprobe")
        )
        playres_x, playres_y = _read_playres(self.ass, video_w, video_h)
        ocr_w, ocr_h = _ocr_canvas(self.cfg, video_w, video_h)
        params = PositionParams.from_cfg(self.cfg, playres_x, playres_y)
        styles = parse_styles(self.ass)

        boxes_by_track = collect_boxes_by_track(frame_ocrs, tracks)
        history = PositionHistory()
        positions: dict[int, str] = {}

        # 先收集需要定位的事件，跳过默认位置 / 无译文 / 无原字幕的事件
        items: list[tuple] = []
        for ev in self.ass.events:
            text = (translations.get(ev.id) or "").strip()
            if not text:
                positions[ev.id] = ""
                continue
            style = styles.get(ev.style)
            if style is None:
                positions[ev.id] = ""
                continue

            track_key = matched.get(ev.id)
            boxes = boxes_by_track.get(track_key, []) if track_key is not None else []
            raw_box = _aggregate_box(boxes)
            source = map_box_to_playres(raw_box, ocr_w, ocr_h, playres_x, playres_y) if raw_box else None
            default_anchor = style_default_anchor(style, playres_x, playres_y)

            if source is None:
                positions[ev.id] = ""
                continue
            items.append((ev, text, source, default_anchor, style))

        total = len(items)

        if params.use_binary_search:
            workers = max(1, int(getattr(self.cfg, "positioning_workers", 4) or 4))
            logger.info("字幕定位并发：%d 线程，待处理 %d 条", workers, total)
            # 提前创建共享的 LLM 客户端，避免多个线程同时初始化
            self._client()

            done = 0
            lock = threading.Lock()

            def _propose_one(item):
                nonlocal done
                ev, text, source, default_anchor, style = item
                try:
                    frame_path = self._event_frame(ev)
                    if frame_path is None:
                        return ev.id, None
                    if self._is_default(ev, frame_path):
                        return ev.id, "default"
                    proposal = self._propose_binary(
                        ev, frame_path, text, source, default_anchor, style, params
                    )
                    return ev.id, proposal
                except Exception:
                    logger.exception("事件 %s 二分定位失败", ev.id)
                    return ev.id, None
                finally:
                    with lock:
                        done += 1
                        if done % 10 == 0 or done == total:
                            logger.info("字幕定位进度 %d/%d", done, total)

            proposals: dict[int, PlacementProposal] = {}
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as ex:
                for ev_id, proposal in ex.map(_propose_one, items):
                    if proposal == "default":
                        positions[ev_id] = ""
                    elif proposal is not None:
                        proposals[ev_id] = proposal

            # 按事件顺序回放历史位置复用，保证结果确定且满足“优先复用旧位置”
            for ev, text, source, default_anchor, style in items:
                proposal = proposals.get(ev.id)
                if proposal is None:
                    positions[ev.id] = ""
                    continue
                est_w, est_h = estimate_text_size(text, style)
                reused = self._nearest_reuse(
                    default_anchor[0], proposal.y, proposal.boundary_y, est_w, est_h, history, params
                )
                if reused is not None:
                    x, y = reused
                else:
                    x, y = proposal.x, proposal.y
                positions[ev.id] = build_position_tag(x, y)
                history.add(default_anchor[0], y)
        else:
            # 旧模式：候选生成 + LLM 打分，仍按顺序处理以便复用历史位置
            for idx, (ev, text, source, default_anchor, style) in enumerate(items, 1):
                frame_path = self._event_frame(ev)
                if frame_path is None:
                    positions[ev.id] = ""
                    continue
                if self._is_default(ev, frame_path):
                    positions[ev.id] = ""
                    continue
                candidates = generate_candidates(source, default_anchor, style, text, history, params)
                if not candidates:
                    positions[ev.id] = ""
                    continue
                best = self._choose(ev, candidates, frame_path, text, params)
                positions[ev.id] = build_position_tag(best.x, best.y)
                history.add(default_anchor[0], best.y)

                if idx % 10 == 0 or idx == total:
                    logger.info("字幕定位进度 %d/%d", idx, total)

        _write_json(cache, {str(k): v for k, v in positions.items()})
        return positions


def position_subtitles(cfg, ass, work_dir: str | Path, translations: dict[int, str] | None = None) -> dict[int, str]:
    """入口：返回 {event_id: position_tag}，tag 为空表示不移动。"""
    engine = PositionEngine(cfg, ass, Path(work_dir))
    return engine.run(translations)
