"""生成人工审核清单（Markdown）。"""
from __future__ import annotations

from .checker import QualityReport


def generate_review(report: QualityReport, events, translations: dict[int, str],
                    originals: dict[int, dict], output_path: str) -> str:
    """把需要人工处理的项目整理为可读的审核文档。"""
    lines = [
        "# 人工审核清单",
        "",
        f"- 事件总数：{report.stats.get('total_events', 0)}",
        f"- 已翻译：{report.stats.get('translated', 0)}",
        f"- 问题数：{report.stats.get('issue_count', 0)}（其中错误 {report.stats.get('error_count', 0)}）",
        f"- 样式匹配：{report.stats.get('style_matched', 0)} 条（平均置信度 {report.stats.get('style_similarity_mean')}）",
        "",
        "## 待审核项目",
        "",
    ]
    ev_by_id = {ev.id: ev for ev in events}
    for issue in report.issues:
        ev = ev_by_id.get(issue.event_id) if issue.event_id is not None else None
        time_range = (
            f"{_ms_to_ts(ev.start_ms)} ~ {_ms_to_ts(ev.end_ms)}"
            if ev else "-"
        )
        original = originals.get(issue.event_id, {}) if issue.event_id is not None else {}
        lines.append(f"### [{issue.category}] #{issue.event_id}")
        lines.append(f"- 时间：{time_range}")
        lines.append(f"- 问题：{issue.message}")
        if original:
            lines.append(f"- 原文（{original.get('source','?')}）：{original.get('text','')}")
            lines.append(f"- 类型：{original.get('type','?')}")
            if original.get("ass_style"):
                lines.append(f"- 样式：{original.get('ass_style')}")
            if original.get("style_similarity") is not None:
                lines.append(f"- 样式置信度：{original.get('style_similarity'):.3f}")
        if issue.event_id is not None:
            lines.append(f"- 译文：{translations.get(issue.event_id, '')}")
        lines.append("")
        lines.append("操作：确认 / 修改 / 跳过 / 重新翻译")
        lines.append("")

    text = "\n".join(lines)
    from pathlib import Path
    Path(output_path).write_text(text, encoding="utf-8")
    return text


def _ms_to_ts(ms: int) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, rem = divmod(rem, 1000)
    return f"{h}:{m:02d}:{s:02d}.{rem//10:02d}"

