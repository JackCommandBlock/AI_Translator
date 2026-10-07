"""AI 字幕翻译流水线入口。

用法：
    python main.py                    # 使用 config.py 中的默认配置
    python main.py --ass sample.ass --video sample.mp4 --out translated.ass
    python main.py --no-asr           # 关闭语音识别
    python main.py --refresh          # 忽略缓存，重新提取
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from logging_utils import setup_logging

_BASE_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _BASE_DIR / "config.py"
_CONFIG_TEMPLATE = _BASE_DIR / "config.example.py"

if not _CONFIG_PATH.exists():
    if _CONFIG_TEMPLATE.exists():
        shutil.copyfile(_CONFIG_TEMPLATE, _CONFIG_PATH)
        print(f"[pipeline] 未找到 config.py，已从 {_CONFIG_TEMPLATE.name} 创建，请按需修改")
    else:
        raise SystemExit(
            "缺少 config.py 且没有 config.example.py 模板，无法启动。"
        )

from config import Config


_PIPELINE_LOGGER = logging.getLogger("pipeline")


def log(msg: str, *args) -> None:
    _PIPELINE_LOGGER.info(msg, *args)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cache(path: Path, force: bool):
    if path.exists() and not force:
        log(f"读取缓存 {path.name}")
        return _read_json(path)
    return None


def _input_signature(cfg: Config) -> dict:
    """根据输入工作文件与影响缓存的关键配置生成签名。"""
    def file_sig(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"path": str(p.resolve()), "exists": False}
        st = p.stat()
        return {
            "path": str(p.resolve()),
            "exists": True,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }

    return {
        "video": file_sig(cfg.video_path),
        "ass": file_sig(cfg.ass_path),
        "config": {
            "source_lang": cfg.source_lang,
            "target_lang": cfg.target_lang,
            "model": cfg.model,
            "subtitle_region": list(cfg.subtitle_region),
            "ocr_mode": cfg.ocr_mode,
            "frame_sample_strategy": cfg.frame_sample_strategy,
            "frames_per_event": cfg.frames_per_event,
            "asr_similarity_threshold": cfg.asr_similarity_threshold,
            "text_track_gap_ms": cfg.text_track_gap_ms,
            "use_ass_type_hint": cfg.use_ass_type_hint,
            "asr_model_size": cfg.asr_model_size,
            "asr_backend": cfg.asr_backend,
            "ocr_backend": cfg.ocr_backend,
            "glossary_path": cfg.glossary_path,
            "use_style_match": getattr(cfg, "use_style_match", True),
            "style_match_weight": getattr(cfg, "style_match_weight", 0.8),
            "style_similarity_min": getattr(cfg, "style_similarity_min", 0.35),
            "style_color_weight": getattr(cfg, "style_color_weight", 1.0),
            "style_outline_weight": getattr(cfg, "style_outline_weight", 1.0),
            "style_bold_weight": getattr(cfg, "style_bold_weight", 0.5),
            "style_position_weight": getattr(cfg, "style_position_weight", 0.3),
        },
    }


def _prepare_work_dir(cfg: Config, work: Path) -> None:
    """输入工作文件或关键配置变化时，清除 work 文件夹内的全部缓存。"""
    work = Path(work)
    meta_path = work / ".cache_meta.json"
    signature = _input_signature(cfg)
    if work.exists():
        old = _read_json(meta_path) if meta_path.exists() else None
        if old == signature:
            return
        log("输入文件或配置发生变化，清除 work 文件夹缓存")
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    _write_json(meta_path, signature)


def _extract_subtitle_crops(cfg: Config, ass, work: Path) -> dict[int, object]:
    """为每个事件抽取中点帧并裁剪字幕区域，返回 {event_id: BGR 图像}。"""
    import cv2
    from ocr.preprocess import crop_subtitle_region
    from video.frame_extract import extract_frames_at_times

    frames_dir = work / "frames_mid"
    times = [ev.start_ms + ev.duration_ms // 2 for ev in ass.events]
    paths = extract_frames_at_times(cfg.video_path, times, frames_dir, prefix="mid")
    crops: dict[int, object] = {}
    for ev, fp in zip(ass.events, paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        crop = crop_subtitle_region(img, cfg.subtitle_region)
        crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        crops[ev.id] = crop
    return crops


def _llm_ocr(cfg: Config, ass, crops: dict[int, object]) -> dict[int, dict]:
    """基于多模态大模型的批量 OCR。"""
    from ocr.llm_vision import LLMVisionOCR
    from translate.translator import build_client

    client = build_client(cfg)
    engine = LLMVisionOCR(client, cfg.model, cfg.source_lang, cfg.ocr_llm_batch_size)
    ids = sorted(crops.keys())
    images = [crops[i] for i in ids]
    log(f"LLM 视觉 OCR：{len(images)} 条，按 {cfg.ocr_llm_batch_size} 条/图拼接")
    ocr_results = engine.ocr_batch(images)
    results: dict[int, dict] = {}
    for ev_id, r in zip(ids, ocr_results):
        results[ev_id] = {"text": r.text, "confidence": r.confidence, "backend": "llm"}
    for ev in ass.events:
        results.setdefault(ev.id, {"text": "", "confidence": None, "backend": "llm"})
    return results


def _local_ocr(cfg: Config, ass, work: Path, backend_name: str) -> dict[int, dict]:
    from ocr.recognizer import create_backend
    from ocr.preprocess import preprocess_image
    from ocr.resolver import resolve_frames
    from video.frame_extract import extract_frames_for_events

    backend = create_backend(backend_name, cfg)
    frames = extract_frames_for_events(
        cfg.video_path, ass.events, work / "frames", cfg.frames_per_event
    )
    results: dict[int, dict] = {}
    total = len(ass.events)
    for i, ev in enumerate(ass.events, 1):
        frame_results = []
        for fp in frames.get(ev.id, []):
            try:
                img = preprocess_image(fp, region=cfg.subtitle_region)
                frame_results.append(backend.recognize(img))
            except Exception as exc:
                _PIPELINE_LOGGER.exception("事件 %s 帧 %s OCR 失败：%s", ev.id, fp.name, exc)
        resolved = resolve_frames(frame_results)
        if resolved and resolved.text.strip():
            results[ev.id] = {
                "text": resolved.text.strip(),
                "confidence": resolved.confidence,
                "backend": resolved.backend,
            }
        else:
            results[ev.id] = {"text": "", "confidence": None, "backend": backend_name}
        if i % 20 == 0 or i == total:
            log(f"OCR 进度 {i}/{total}")
    return results


def run_ocr(cfg: Config, ass, work: Path) -> dict[int, dict]:
    from ocr.recognizer import get_available_backend

    cache_path = work / "ocr.json"
    cached = _load_cache(cache_path, cfg.force_refresh)
    if cached is not None:
        return {int(k): v for k, v in cached.items()}

    backend_name = get_available_backend(cfg)
    log(f"OCR 后端：{backend_name}")

    ocr_ok = True
    try:
        if backend_name == "llm":
            crops = _extract_subtitle_crops(cfg, ass, work)
            results = _llm_ocr(cfg, ass, crops)
        else:
            results = _local_ocr(cfg, ass, work, backend_name)
            # 本地 OCR 置信度过低时，调用多模态大模型兜底处理这些事件
            low_ids = [
                ev.id for ev in ass.events
                if results[ev.id]["text"]
                and (
                    results[ev.id]["confidence"] is None
                    or results[ev.id]["confidence"] < cfg.ocr_confidence_threshold
                )
            ]
            if low_ids:
                log(f"{len(low_ids)} 条 OCR 置信度过低，调用大模型兜底")
                crops = _extract_subtitle_crops(cfg, ass, work)
                subset = {i: crops[i] for i in low_ids if i in crops}
                llm_res = _llm_ocr(cfg, ass, subset)
                for i in low_ids:
                    if llm_res[i]["text"]:
                        results[i] = llm_res[i]
    except Exception as exc:
        _PIPELINE_LOGGER.exception("OCR 失败（将交由 ASR 补充原文）：%s", exc)
        results = {ev.id: {"text": "", "confidence": None, "backend": "failed"} for ev in ass.events}
        ocr_ok = False

    # 失败时不写缓存，以便下次（例如充值后）自动重试
    if ocr_ok:
        _write_json(cache_path, results)
    return results


def run_asr(cfg: Config, ass, work: Path) -> dict[int, dict]:
    if cfg.asr_backend == "none":
        return {}
    from video.audio_extract import extract_audio_wav
    from asr.whisper import ASRSegment, transcribe
    from align.matcher import align_asr_to_events

    audio_path = work / "audio.wav"
    if not audio_path.exists():
        log("抽取音频…")
        extract_audio_wav(cfg.video_path, audio_path)

    cache_path = work / "asr.json"
    cached = _load_cache(cache_path, cfg.force_refresh)
    if cached is not None:
        matches = {int(k): v for k, v in cached.get("matches", {}).items()}
        return matches

    log(f"ASR 后端：{cfg.asr_backend or 'auto'}（模型 {cfg.asr_model_size}）")
    segments = transcribe(audio_path, cfg)
    log(f"ASR 得到 {len(segments)} 个片段")
    matches = align_asr_to_events(ass.events, segments, cfg.source_lang)
    seg_data = [
        {
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "text": s.text,
            "confidence": s.confidence,
            "words": [{"start_ms": w.start_ms, "end_ms": w.end_ms, "word": w.word} for w in s.words],
        }
        for s in segments
    ]
    match_data = {
        str(k): {"text": v.text, "confidence": v.confidence, "source": v.source}
        for k, v in matches.items()
    }
    _write_json(cache_path, {"segments": seg_data, "matches": match_data})
    return {int(k): v for k, v in match_data.items()}


def _event_sample_times(ev, frames_per_event: int) -> list[int]:
    """在事件窗口内取首/中/尾等距采样点。"""
    if frames_per_event <= 1:
        return [ev.start_ms]
    return [
        int(ev.start_ms + ev.duration_ms * i / (frames_per_event - 1))
        for i in range(frames_per_event)
    ]


def _sample_times_for_events(ass, cfg: Config) -> list[int]:
    """计算全帧采样的时间点并集（重叠事件共享同一时刻，天然去重）。"""
    strategy = getattr(cfg, "frame_sample_strategy", "union")
    n = getattr(cfg, "frames_per_event", 3)
    times: set[int] = set()
    for ev in ass.events:
        if strategy == "midpoint":
            times.add(ev.start_ms + ev.duration_ms // 2)
        else:
            times.update(_event_sample_times(ev, n))
    return sorted(times)


def run_full_frame_ocr(cfg: Config, ass, work: Path, classifier=None):
    """抽全帧 -> 全帧多文本 OCR，返回 list[FrameOCR]。"""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    from ocr.recognizer import create_backend, get_available_backend
    from ocr.scene_text import FrameOCR, LocalSceneTextOCR, SceneTextOCR
    from translate.translator import build_client
    from video.frame_extract import extract_frames_at_times_map

    cache_path = work / "frame_ocr.json"
    cached = _load_cache(cache_path, cfg.force_refresh)
    if cached is not None:
        return [FrameOCR.from_dict(d) for d in cached]

    times = _sample_times_for_events(ass, cfg)
    pairs = extract_frames_at_times_map(
        cfg.video_path, times, work / "frames_full", prefix="full"
    )

    backend_name = get_available_backend(cfg)
    log(f"全帧 OCR 后端：{backend_name}（采样 {len(times)} 帧）")

    if backend_name == "llm":
        engine = SceneTextOCR(
            build_client(cfg), cfg.model, cfg.source_lang, cfg.llm_temperature
        )
        workers = max(1, int(getattr(cfg, "ocr_workers", 4) or 4))
        log(f"LLM 全帧 OCR 并发：{workers} 线程")

        def recognize_one(item):
            t, fp = item
            img = cv2.imread(str(fp))
            if img is None:
                log(f"无法读取帧 {fp.name}，跳过")
                return None
            try:
                return FrameOCR(time_ms=t, blocks=engine.recognize(img))
            except Exception as exc:
                _PIPELINE_LOGGER.exception("帧 %sms OCR 失败：%s", t, exc)
                return FrameOCR(time_ms=t, blocks=[])

        frame_ocrs: list[FrameOCR] = []
        total = len(pairs)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, result in enumerate(ex.map(recognize_one, pairs), 1):
                if result is not None:
                    frame_ocrs.append(result)
                if i % 20 == 0 or i == total:
                    log(f"全帧 OCR 进度 {i}/{total}")
    else:
        engine = LocalSceneTextOCR(create_backend(backend_name, cfg), classifier=classifier)
        frame_ocrs = []
        total = len(pairs)
        for i, (t, fp) in enumerate(pairs, 1):
            img = cv2.imread(str(fp))
            if img is None:
                log(f"无法读取帧 {fp.name}，跳过")
                continue
            try:
                blocks = engine.recognize(img)
            except Exception as exc:
                _PIPELINE_LOGGER.exception("帧 %sms OCR 失败：%s", t, exc)
                blocks = []
            frame_ocrs.append(FrameOCR(time_ms=t, blocks=blocks))
            if i % 20 == 0 or i == total:
                log(f"全帧 OCR 进度 {i}/{total}")

    _write_json(cache_path, [f.to_dict() for f in frame_ocrs])
    return frame_ocrs


def run_text_timeline(cfg: Config, frame_ocrs, work: Path):
    """把全帧 OCR 结果聚成文本轨道。"""
    from align.text_timeline import TextTrack, build_text_timeline as _build

    cache_path = work / "text_timeline.json"
    cached = _load_cache(cache_path, cfg.force_refresh)
    if cached is not None:
        return [TextTrack.from_dict(d) for d in cached]

    tracks = _build(frame_ocrs, gap_ms=cfg.text_track_gap_ms)
    _write_json(cache_path, [t.to_dict() for t in tracks])
    return tracks


def run_event_track_matching(
    cfg: Config, ass, tracks, asr, work: Path, styles=None
) -> dict[int, int]:
    """事件窗口与文本轨道做时间覆盖 + ASR + 样式的一对一匹配。"""
    from align.text_timeline import match_events_to_tracks

    cache_path = work / "matched.json"
    cached = _load_cache(cache_path, cfg.force_refresh)
    if cached is not None:
        return {int(k): int(v) for k, v in cached.items()}

    matched = match_events_to_tracks(ass.events, tracks, asr, cfg, styles=styles)
    _write_json(cache_path, {str(k): v for k, v in matched.items()})
    return matched


def _resolve_dialogue_original(
    cfg, ev, ocr_text, ocr_conf, asr_text, asr_conf, translator
) -> dict:
    """台词原文判定：OCR 优先，冲突交 LLM 裁决，OCR 失败回退 ASR。"""
    record = {"text": "", "source": "none", "confidence": None, "conflict": False}
    if ocr_text and (ocr_conf is None or ocr_conf >= cfg.ocr_confidence_threshold):
        record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
    elif ocr_text and asr_text:
        if _norm(ocr_text) == _norm(asr_text):
            record.update(text=ocr_text, source="ocr+asr", confidence=ocr_conf)
        else:
            record["conflict"] = True
            if translator is not None:
                try:
                    chosen = translator.resolve_conflict(ocr_text, asr_text, [])
                    if chosen.strip():
                        record.update(text=chosen.strip(), source="ocr+asr", confidence=ocr_conf)
                    else:
                        record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
                except Exception as exc:
                    _PIPELINE_LOGGER.exception("事件 %s 冲突消解失败，暂用 OCR：%s", ev.id, exc)
                    record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
            else:
                record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
    elif ocr_text:
        record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
    elif asr_text:
        record.update(text=asr_text, source="asr", confidence=asr_conf)
    return record


def determine_originals_full_frame(
    cfg, ass, tracks, matched, asr, translator=None, classifier=None, styles=None
) -> dict[int, dict]:
    """按匹配到的文本轨道 + 推断类型判定原文。"""
    from align.text_timeline import _asr_text, infer_type
    from ocr.style_features import style_similarity

    tracks_by_key = {t.key: t for t in tracks}
    styles = styles or {}
    originals: dict[int, dict] = {}
    for ev in ass.events:
        a = asr.get(ev.id)
        asr_text = _asr_text(a)
        asr_conf = (
            a.get("confidence")
            if isinstance(a, dict)
            else (getattr(a, "confidence", None) if a else None)
        )

        track = tracks_by_key.get(matched.get(ev.id)) if ev.id in matched else None
        ocr_text = (track.text or "").strip() if track else ""
        block_type = (track.type or "unknown") if track else "unknown"
        ocr_conf = track.confidence if track else None

        record = {
            "text": "",
            "source": "none",
            "type": "unknown",
            "confidence": None,
            "conflict": False,
            "ocr_text": ocr_text,
            "asr_text": asr_text,
            "style": None,
            "style_similarity": None,
            "ass_style": None,
        }

        if track is not None:
            st = styles.get(ev.id)
            if st is not None:
                record["ass_style"] = st.name
            if track.style is not None:
                record["style"] = track.style.to_dict()
                if st is not None:
                    record["style_similarity"] = round(
                        float(style_similarity(track.style, st, cfg)), 3
                    )
            typ = infer_type(block_type, ocr_text, asr_text, cfg, classifier)
            record["type"] = typ
            if typ == "dialogue":
                record.update(
                    _resolve_dialogue_original(
                        cfg, ev, ocr_text, ocr_conf, asr_text, asr_conf, translator
                    )
                )
            else:
                # 非台词只用 OCR，不回退 ASR，避免台词污染标题
                if ocr_text:
                    record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
        else:
            # 未匹配到文字轨道：仅有语音时按台词处理
            if asr_text:
                record.update(
                    text=asr_text, source="asr", confidence=asr_conf, type="dialogue"
                )

        if not record["text"].strip():
            record["source"] = "none"
        originals[ev.id] = record
    return originals


def determine_originals(
    cfg: Config, ass, ocr: dict[int, dict], asr: dict[int, dict], translator=None
) -> dict[int, dict]:
    from ass.parser import format_time

    originals: dict[int, dict] = {}
    for ev in ass.events:
        o = ocr.get(ev.id, {})
        a = asr.get(ev.id)
        ocr_text = (o.get("text") or "").strip()
        ocr_conf = o.get("confidence")
        asr_text = (a.get("text") or "").strip() if a else ""
        asr_conf = a.get("confidence") if a else None

        record = {
            "text": "",
            "source": "none",
            "type": "dialogue",
            "confidence": None,
            "conflict": False,
            "ocr_text": ocr_text,
            "asr_text": asr_text,
        }

        if ocr_text and (ocr_conf is None or ocr_conf >= cfg.ocr_confidence_threshold):
            record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
        elif ocr_text and asr_text:
            if _norm(ocr_text) == _norm(asr_text):
                record.update(text=ocr_text, source="ocr+asr", confidence=ocr_conf)
            else:
                record["conflict"] = True
                if translator is not None:
                    try:
                        chosen = translator.resolve_conflict(ocr_text, asr_text, [])
                        if chosen.strip():
                            record.update(text=chosen.strip(), source="ocr+asr", confidence=ocr_conf)
                        else:
                            record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
                    except Exception as exc:
                        _PIPELINE_LOGGER.exception("事件 %s 冲突消解失败，暂用 OCR：%s", ev.id, exc)
                        record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
                else:
                    record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
        elif ocr_text:
            record.update(text=ocr_text, source="ocr", confidence=ocr_conf)
        elif asr_text:
            record.update(text=asr_text, source="asr", confidence=asr_conf)

        if not record["text"].strip():
            record["source"] = "none"
        originals[ev.id] = record

    return originals


def _norm(text: str) -> str:
    return "".join(text.split()).lower()


def correct_originals(cfg: Config, ass, originals: dict[int, dict], translator) -> dict[int, dict]:
    """用大模型对 OCR/ASR 原文纠错（原地修改 originals 的 text）。"""
    entries = []
    for ev in ass.events:
        rec = originals[ev.id]
        if rec.get("text") and rec.get("source") in ("ocr", "asr", "ocr+asr"):
            entries.append({"id": ev.id, "name": ev.name, "text": rec["text"]})
    if not entries:
        return originals
    log(f"原文纠错 {len(entries)} 条…")
    context = _build_context(ass, originals)
    try:
        corrected = translator.correct(entries, context)
    except Exception as exc:
        _PIPELINE_LOGGER.exception("原文纠错失败，跳过：%s", exc)
        return originals
    for ev_id, text in corrected.items():
        if text.strip() and ev_id in originals:
            originals[ev_id]["text"] = text.strip()
    return originals


def translate_events(cfg: Config, ass, originals: dict[int, dict], translator) -> dict[int, str]:
    entries = []
    for ev in ass.events:
        rec = originals[ev.id]
        if rec.get("text"):
            entries.append(
                {
                    "id": ev.id,
                    "name": ev.name,
                    "text": rec["text"],
                    "type": rec.get("type", "dialogue"),
                }
            )
    if not entries:
        return {}
    log(f"批量翻译 {len(entries)} 条…")
    context = _build_context(ass, originals)
    return translator.translate(entries, context)


def _build_context(
    ass, originals: dict[int, dict], window: int = 3, type_filter=None
) -> list[dict]:
    ctx = []
    for ev in ass.events:
        rec = originals.get(ev.id, {})
        typ = rec.get("type", "dialogue")
        if type_filter is not None and typ != type_filter:
            continue
        ctx.append(
            {"id": ev.id, "name": ev.name, "text": rec.get("text", ""), "type": typ}
        )
    return ctx


def length_check_and_compress(cfg: Config, ass, translations: dict[int, str], translator) -> dict[int, str]:
    from ass.tags import clean_tags

    out = dict(translations)
    for ev in ass.events:
        text = out.get(ev.id, "")
        if not text.strip():
            continue
        clean = clean_tags(text)
        lines = clean.split("\u23ce")
        if any(len(line) > cfg.max_chars_per_line * 1.5 for line in lines):
            log(f"事件 {ev.id} 译文过长，尝试压缩…")
            try:
                merged = "".join(lines)
                out[ev.id] = translator.compress(merged, cfg.max_chars_per_line * 2)
            except Exception as exc:
                _PIPELINE_LOGGER.exception("事件 %s 压缩失败：%s", ev.id, exc)
    return out


def build_pipeline(cfg: Config):
    from ass.parser import parse_ass

    ass = parse_ass(cfg.ass_path)
    log(f"解析 ASS：{len(ass.events)} 条 Dialogue 事件")
    work = Path(cfg.work_dir)
    _prepare_work_dir(cfg, work)

    asr = {}
    if cfg.enable_asr:
        asr = run_asr(cfg, ass, work)

    translator = None
    classifier = None
    if cfg.enable_translate:
        from translate.translator import Translator
        translator = Translator(cfg)

    ocr_mode = getattr(cfg, "ocr_mode", "full_frame")
    if cfg.enable_ocr and ocr_mode == "full_frame":
        from ocr.scene_text import ContentTypeClassifier
        from ass.styles import style_for_events
        if translator is not None:
            classifier = ContentTypeClassifier(
                translator.client, cfg.model, cfg.llm_temperature
            )
        style_map = (
            style_for_events(ass, ass.events)
            if getattr(cfg, "use_style_match", True)
            else {}
        )
        frame_ocrs = run_full_frame_ocr(cfg, ass, work, classifier=classifier)
        tracks = run_text_timeline(cfg, frame_ocrs, work)
        matched = run_event_track_matching(cfg, ass, tracks, asr, work, styles=style_map)
        originals = determine_originals_full_frame(
            cfg,
            ass,
            tracks,
            matched,
            asr,
            translator=translator,
            classifier=classifier,
            styles=style_map,
        )
    else:
        ocr = {}
        if cfg.enable_ocr:
            ocr = run_ocr(cfg, ass, work)
        originals = determine_originals(cfg, ass, ocr, asr, translator)

    _write_json(work / "original.json", originals)

    if cfg.enable_translate and translator is not None:
        try:
            originals = correct_originals(cfg, ass, originals, translator)
            _write_json(work / "original_corrected.json", originals)
            translations = translate_events(cfg, ass, originals, translator)
            translations = length_check_and_compress(cfg, ass, translations, translator)
            _write_json(work / "translated.json", translations)
        except Exception as exc:
            _PIPELINE_LOGGER.exception("翻译/纠错失败（译文将留空并进入审核）：%s", exc)
            translations = {ev.id: "" for ev in ass.events}
    else:
        translations = {ev.id: "" for ev in ass.events}

    # 可选：用 LLM 评分给译文字幕判别位置
    if getattr(cfg, "enable_positioning", False):
        from positioning.positioner import position_subtitles, apply_positions

        position_tags = position_subtitles(cfg, ass, work, translations)
        translations = apply_positions(translations, position_tags)
        log(f"字幕定位完成：{sum(1 for t in position_tags.values() if t)} 条移动")

    # 写回最终 ASS
    from output.exporter import export
    export(ass, translations, cfg.output_path)
    log(f"已生成成品字幕：{cfg.output_path}")

    # 质量检查 + 审核清单
    from quality.checker import check
    from quality.review import generate_review
    report = check(ass.events, translations, originals, cfg)
    _write_json(work / "quality_report.json", {
        "stats": report.stats,
        "issues": [
            {"event_id": i.event_id, "category": i.category, "message": i.message, "severity": i.severity}
            for i in report.issues
        ],
    })
    review_path = work / "review.md"
    if cfg.enable_review:
        generate_review(report, ass.events, translations, originals, str(review_path))
        log(f"人工审核清单：{review_path}")

    _print_summary(report, ass, originals)
    return cfg.output_path


def _print_summary(report, ass, originals) -> None:
    stats = report.stats
    src_count = {}
    for rec in originals.values():
        s = rec.get("source", "none")
        src_count[s] = src_count.get(s, 0) + 1
    type_count = {}
    for rec in originals.values():
        t = rec.get("type", "unknown")
        type_count[t] = type_count.get(t, 0) + 1

    _PIPELINE_LOGGER.info(
        "质量检查汇总：事件=%s 已翻译=%s 问题=%s 错误=%s 原文来源=%s 内容类型=%s",
        stats.get("total_events", 0),
        stats.get("translated", 0),
        stats.get("issue_count", 0),
        stats.get("error_count", 0),
        src_count,
        type_count,
    )

    print()
    print("=" * 46)
    print("质量检查汇总")
    print("=" * 46)
    print(f"事件总数：{stats.get('total_events', 0)}")
    print(f"已翻译：{stats.get('translated', 0)}")
    print(f"问题数：{stats.get('issue_count', 0)}")
    print(f"错误数：{stats.get('error_count', 0)}")
    print("原文来源统计：", src_count)
    print("内容类型统计：", type_count)


def parse_args(argv):
    p = argparse.ArgumentParser(description="AI 字幕翻译流水线")
    p.add_argument("--ass", help="预处理 ASS 文件")
    p.add_argument("--video", help="原始视频文件")
    p.add_argument("--out", help="输出 ASS 文件")
    p.add_argument("--config", help="配置文件路径")
    p.add_argument("--no-ocr", action="store_true", help="关闭 OCR")
    p.add_argument("--no-asr", action="store_true", help="关闭语音识别")
    p.add_argument("--no-translate", action="store_true", help="关闭翻译（仅识别原文）")
    p.add_argument("--positioning", action="store_true", help="用 LLM 评分给译文字幕判别位置")
    p.add_argument("--refresh", action="store_true", help="忽略缓存重新提取")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    # 先用默认参数初始化日志，确保加载配置阶段的异常也能落盘。
    setup_logging()
    try:
        cfg = Config.load(args.config)
    except Exception as exc:
        _PIPELINE_LOGGER.exception("加载配置失败：%s", exc)
        return 1
    if args.ass:
        cfg.ass_path = args.ass
    if args.video:
        cfg.video_path = args.video
    if args.out:
        cfg.output_path = args.out
    if args.no_ocr:
        cfg.enable_ocr = False
    if args.no_asr:
        cfg.enable_asr = False
    if args.no_translate:
        cfg.enable_translate = False
    if args.positioning:
        cfg.enable_positioning = True
    if args.refresh:
        cfg.force_refresh = True

    setup_logging(
        log_dir=cfg.log_dir,
        log_file=cfg.log_file,
        level=cfg.log_level,
        console=cfg.log_console,
    )
    log(
        "启动流水线：ass=%s video=%s out=%s ocr=%s asr=%s translate=%s",
        cfg.ass_path,
        cfg.video_path,
        cfg.output_path,
        cfg.enable_ocr,
        cfg.enable_asr,
        cfg.enable_translate,
    )

    t0 = time.time()
    try:
        out = build_pipeline(cfg)
    except Exception as exc:
        _PIPELINE_LOGGER.exception("流水线执行失败：%s", exc)
        return 1
    log(f"全部完成，耗时 {time.time() - t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
