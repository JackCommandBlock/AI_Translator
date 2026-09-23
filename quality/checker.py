"""生成 ASS 后的自动质量检查。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CheckIssue:
    event_id: int | None
    category: str
    message: str
    severity: str = "warning"  # error | warning


@dataclass
class QualityReport:
    issues: list[CheckIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, event_id, category, message, severity="warning"):
        self.issues.append(CheckIssue(event_id, category, message, severity))


def _tag_balanced(text: str) -> bool:
    return text.count("{") == text.count("}")


def _line_len(text: str) -> list[int]:
    # 去掉标签，按换行切分后统计长度
    clean = re.sub(r"\{[^{}]*\}", "", text)
    return [len(line) for line in clean.split("\u23ce")]


def _has_japanese(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" for ch in text)


def check(events, translations: dict[int, str], originals: dict[int, dict], config=None) -> QualityReport:
    """执行全部自动检查。"""
    report = QualityReport()
    max_line = getattr(config, "max_chars_per_line", 42)
    max_lines = getattr(config, "max_lines", 2)
    ocr_th = getattr(config, "ocr_confidence_threshold", 0.6)
    asr_th = getattr(config, "asr_confidence_threshold", 0.5)
    n_translated = 0

    if len(translations) != len(events):
        report.add(None, "count_mismatch",
                   f"翻译结果数量 {len(translations)} 与事件数量 {len(events)} 不一致", "error")

    for ev in events:
        tid = ev.id
        original = originals.get(tid)
        translated = translations.get(tid, "")

        if ev.start_ms >= ev.end_ms:
            report.add(tid, "timeline_error", "开始时间大于等于结束时间", "error")

        if not translated or not translated.strip():
            report.add(tid, "empty_subtitle", "译文为空", "warning")
        else:
            n_translated += 1

        if translated and _has_japanese(translated):
            report.add(tid, "untranslated", "译文中仍残留原文（日语字符）", "warning")

        if not _tag_balanced(translated):
            report.add(tid, "broken_tags", "ASS 标签括号不匹配", "error")

        line_lens = _line_len(translated)
        if line_lens and max(line_lens) > max_line * 1.5:
            report.add(tid, "too_long", f"单行过长（{max(line_lens)} 字符）", "warning")
        if len(line_lens) > max_lines + 1:
            report.add(tid, "too_many_lines", f"行数过多（{len(line_lens)} 行）", "warning")

        if original:
            conf = original.get("confidence")
            source = original.get("source", "")
            if conf is not None and conf < (ocr_th if source != "asr" else asr_th):
                report.add(tid, "low_confidence",
                           f"{source} 置信度低（{conf:.2f}）", "warning")
            if original.get("conflict"):
                report.add(tid, "ocr_asr_conflict", "OCR 与 ASR 结果存在冲突", "warning")
        else:
            report.add(tid, "no_original", "未能识别出原文，无法翻译", "warning")

    report.stats = {
        "total_events": len(events),
        "translated": n_translated,
        "issue_count": len(report.issues),
        "error_count": sum(1 for i in report.issues if i.severity == "error"),
    }
    return report

