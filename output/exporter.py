"""把翻译结果组合为最终 ASS 文本。"""
from __future__ import annotations

from ass.tags import tokenize, extract_translatable, recompose


def finalize_text(original_ass_text: str, translated_text: str) -> str:
    """将译文与原始 ASS 标签结构重新组合。"""
    if not original_ass_text:
        return translated_text
    tokens = tokenize(original_ass_text)
    translatable = extract_translatable(tokens)
    if translatable.strip():
        # 原文 ASS 自带正文：按 token 结构替换
        return recompose(tokens, translated_text)
    # 原文 ASS 只有标签 / 换行：保留标签，把译文插入标签之后
    leading = []
    trailing = []
    for t in tokens:
        if t.kind == "tag":
            leading.append(t.value)
        elif t.kind == "newline":
            trailing.append(t.value)
    return "".join(leading) + translated_text + "".join(trailing)


def export(ass, translations: dict[int, str], output_path: str) -> None:
    """将翻译结果写回 ASS。"""
    from ass.writer import write_ass
    final = {}
    for ev in ass.events:
        translated = translations.get(ev.id, "")
        final[ev.id] = finalize_text(ev.text, translated)
    write_ass(ass, output_path, texts=final)

