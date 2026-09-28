"""翻译 / 纠错 / 冲突消解使用的提示词。"""
from __future__ import annotations

from .glossary import format_glossary

NL = "\u23ce"  # 换行占位符


TYPE_HINT = {
    "dialogue": "主台词，口语自然",
    "title":    "标题/专名，简洁，不加标点",
    "note":     "辅助说明，忠实直译",
    "sign":     "画面内文字，忠实直译",
    "lyrics":   "歌词，可保留韵律",
}


def _entries_type(entries: list[dict]) -> str:
    for e in entries:
        t = e.get("type", "dialogue")
        if t:
            return t
    return "dialogue"


def system_prompt(target_lang_name: str) -> str:
    return (
        "你是一名专业的影视字幕翻译。你输出的内容会直接写入 ASS 字幕文件，"
        "因此必须严格遵守输出格式，不得输出任何多余说明、代码块标记或解释。"
    )


def translation_user_prompt(
    source_lang: str,
    target_lang: str,
    entries: list[dict],
    glossary: dict[str, str],
    context: list[dict],
) -> str:
    """构造批量翻译的 user prompt。

    entries: [{id, name, text}]
    context: 供参考的上下文（前后若干条原文）
    """
    lines: list[str] = []
    lines.append(f"请把以下{source_lang}字幕逐条翻译成{target_lang}。")
    lines.append("")
    lines.append("要求：")
    lines.append("1. 只输出一个 JSON 数组，格式为 [{\"id\": 数字, \"text\": \"译文\"}]，按输入顺序，id 与输入完全一致。")
    lines.append("2. 只输出译文本身，不要解释，不要加引号包裹以外的内容，不要输出 ASS 特效标签。")
    lines.append(f"3. 文本中的特殊符号 {NL} 表示换行，请原样保留该符号，不要替换成其它字符。")
    lines.append("4. 保持人物称呼、专有名词、术语前后统一，结合上下文翻译。")
    lines.append("5. 控制译文长度，单条字幕尽量不超过两行，避免过长。")
    lines.append("6. 口语化、自然，符合目标语言习惯。")
    entry_type = _entries_type(entries)
    hint = TYPE_HINT.get(entry_type, "")
    if hint:
        lines.append(f"7. 本批内容类型为 {entry_type}（{hint}），请按该类型特点翻译。")
    if glossary:
        lines.append("")
        lines.append("术语表（必须优先遵守）：")
        lines.append(format_glossary(glossary))
    if context:
        lines.append("")
        lines.append("上下文参考（id: 说话人 | 原文）：")
        for c in context:
            lines.append(f"{c['id']}: {c.get('name','')} | {c['text']}")
    lines.append("")
    lines.append("待翻译条目：")
    for e in entries:
        lines.append(f"id={e['id']} | {e.get('type','dialogue')} | {e.get('name','')} | {e['text']}")
    return "\n".join(lines)


def correction_user_prompt(source_lang: str, entries: list[dict], context: list[dict]) -> str:
    """OCR/ASR 原文纠错提示词。"""
    lines = [
        f"以下是 OCR/ASR 自动识别出的{source_lang}字幕，可能存在错字、漏字或误识别。",
        "请在不改变原意的前提下修正为自然的原文。",
        f"只输出 JSON 数组 [{{\"id\": 数字, \"text\": \"修正后的原文\"}}]，id 与输入一致。",
        f"保留文本中的 {NL} 换行符号。不要解释。",
    ]
    if context:
        lines.append("")
        lines.append("上下文参考（id: 说话人 | 原文）：")
        for c in context:
            lines.append(f"{c['id']}: {c.get('name','')} | {c['text']}")
    lines.append("")
    lines.append("待修正条目：")
    for e in entries:
        lines.append(f"id={e['id']} | {e.get('name','')} | {e['text']}")
    return "\n".join(lines)


def conflict_user_prompt(source_lang: str, ocr_text: str, asr_text: str, context: list[dict]) -> str:
    lines = [
        f"同一个字幕片段，OCR 与 ASR 给出了不同的{source_lang}原文，请判断哪一个是正确原文。",
        "只输出最终确定的原文，不要解释，不要加引号。",
    ]
    if context:
        lines.append("上下文参考（id: 说话人 | 原文）：")
        for c in context:
            lines.append(f"{c['id']}: {c.get('name','')} | {c['text']}")
    lines.append("")
    lines.append(f"OCR 结果：{ocr_text}")
    lines.append(f"ASR 结果：{asr_text}")
    return "\n".join(lines)


def compress_user_prompt(target_lang: str, text: str, max_chars: int) -> str:
    return (
        f"请把下面这条{target_lang}字幕在不改变原意的前提下压缩到不超过 {max_chars} 个字符，"
        f"只输出压缩后的字幕，不要解释。\n\n{text}"
    )

