"""大模型翻译客户端，调用方式参考 gemini_example.py。"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from .client import build_client, chat_completion


logger = logging.getLogger("pipeline.translate")


def _extract_json(text: str) -> Any:
    """从模型输出中稳健地解析 JSON。"""
    text = text.strip()
    # 去掉可能的代码块围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析模型输出为 JSON: {text[:200]!r}")


class Translator:
    def __init__(self, config):
        self.config = config
        self.client = build_client(config)
        self.model = config.model
        self.temperature = config.llm_temperature

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        resp = chat_completion(
            self.client,
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], temperature: float | None = None) -> Any:
        content = self.chat(messages, temperature=temperature)
        return _extract_json(content)

    def _batch(self, entries: list[dict], prompt_fn, context: list[dict]) -> dict[int, str]:
        """通用批量处理：按 batch_size 切分，返回 {id: text}。"""
        result: dict[int, str] = {}
        bs = self.config.batch_size
        ctx_by_id = {c["id"]: c for c in context} if context else {}
        for i in range(0, len(entries), bs):
            chunk = entries[i : i + bs]
            logger.info(
                "批量处理第 %d/%d 批，条数 %d",
                i // bs + 1,
                (len(entries) + bs - 1) // bs,
                len(chunk),
            )
            # 为当前批次挑选窗口内的上下文，避免把整部片子都塞进每次请求
            ids = [e["id"] for e in chunk]
            lo, hi = min(ids), max(ids)
            focused = [ctx_by_id[k] for k in sorted(ctx_by_id) if lo - 3 <= k <= hi + 3]
            user = prompt_fn(chunk, focused)
            messages = [
                {"role": "system", "content": "你是专业的字幕处理助手，只输出要求的 JSON。"},
                {"role": "user", "content": user},
            ]
            try:
                data = self.chat_json(messages)
            except Exception:
                logger.exception("批量处理失败，批次条目 %s", ids)
                raise
            if isinstance(data, dict):
                data = data.get("items") or data.get("results") or [data]
            if not isinstance(data, list):
                raise ValueError("模型未返回 JSON 数组")
            for item in data:
                if isinstance(item, dict) and "id" in item:
                    result[int(item["id"])] = str(item.get("text") or item.get("translation") or "")
            time.sleep(0.2)
        return result

    def translate(self, entries, context) -> dict[int, str]:
        from .prompt import translation_user_prompt
        from .glossary import load_glossary

        logger.info("开始翻译，待翻译条目 %d", len(entries))
        if hasattr(self.config, "glossary_file"):
            glossary_path = self.config.glossary_file()
        else:
            glossary_path = self.config.glossary_path
        glossary = load_glossary(glossary_path)

        def prompt_fn(chunk, ctx):
            return translation_user_prompt(
                self.config.source_lang, self.config.target_lang, chunk, glossary, ctx
            )

        # 按内容类型分组，同类型内取时间相邻事件做上下文，避免标题与台词混在同一窗口
        groups: dict[str, list] = {}
        for e in entries:
            groups.setdefault(e.get("type", "dialogue"), []).append(e)

        result: dict[int, str] = {}
        for typ, group in groups.items():
            logger.info("翻译类型 %s：%d 条", typ, len(group))
            typed_context = [c for c in context if c.get("type", "dialogue") == typ]
            result.update(self._batch(group, prompt_fn, typed_context))
        logger.info("翻译完成，成功 %d/%d 条", len(result), len(entries))
        return result

    def correct(self, entries, context) -> dict[int, str]:
        from .prompt import correction_user_prompt

        logger.info("开始原文纠错，条目 %d", len(entries))

        def prompt_fn(chunk, ctx):
            return correction_user_prompt(self.config.source_lang, chunk, ctx)

        result = self._batch(entries, prompt_fn, context)
        logger.info("原文纠错完成，成功 %d/%d 条", len(result), len(entries))
        return result

    def resolve_conflict(self, ocr_text: str, asr_text: str, context) -> str:
        from .prompt import conflict_user_prompt

        user = conflict_user_prompt(self.config.source_lang, ocr_text, asr_text, context)
        messages = [
            {"role": "system", "content": "你是字幕原文校正助手，只输出原文。"},
            {"role": "user", "content": user},
        ]
        return self.chat(messages).strip()

    def compress(self, text: str, max_chars: int) -> str:
        from .prompt import compress_user_prompt

        user = compress_user_prompt(self.config.target_lang, text, max_chars)
        messages = [
            {"role": "system", "content": "你是字幕压缩助手，只输出压缩后的字幕。"},
            {"role": "user", "content": user},
        ]
        return self.chat(messages).strip()
