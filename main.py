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
import shutil
import sys
import time
from pathlib import Path

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


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


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
            "frames_per_event": cfg.frames_per_event,
            "asr_model_size": cfg.asr_model_size,
            "asr_backend": cfg.asr_backend,
            "ocr_backend": cfg.ocr_backend,
            "glossary_path": cfg.glossary_path,
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
                log(f"事件 {ev.id} 帧 {fp.name} OCR 失败: {exc}")
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
        log(f"OCR 失败（将交由 ASR 补充原文）：{exc}")
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
                        log(f"事件 {ev.id} 冲突消解失败，暂用 OCR：{exc}")
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
        log(f"原文纠错失败，跳过：{exc}")
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
            entries.append({"id": ev.id, "name": ev.name, "text": rec["text"]})
    if not entries:
        return {}
    log(f"批量翻译 {len(entries)} 条…")
    context = _build_context(ass, originals)
    return translator.translate(entries, context)


def _build_context(ass, originals: dict[int, dict], window: int = 3) -> list[dict]:
    ctx = []
    for ev in ass.events:
        rec = originals.get(ev.id, {})
        ctx.append({"id": ev.id, "name": ev.name, "text": rec.get("text", "")})
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
                log(f"事件 {ev.id} 压缩失败：{exc}")
    return out


def build_pipeline(cfg: Config):
    from ass.parser import parse_ass

    ass = parse_ass(cfg.ass_path)
    log(f"解析 ASS：{len(ass.events)} 条 Dialogue 事件")
    work = Path(cfg.work_dir)
    _prepare_work_dir(cfg, work)

    ocr = {}
    asr = {}
    if cfg.enable_ocr:
        ocr = run_ocr(cfg, ass, work)
    if cfg.enable_asr:
        asr = run_asr(cfg, ass, work)

    translator = None
    if cfg.enable_translate:
        from translate.translator import Translator
        translator = Translator(cfg)

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
            log(f"翻译/纠错失败（译文将留空并进入审核）：{exc}")
            translations = {ev.id: "" for ev in ass.events}
    else:
        translations = {ev.id: "" for ev in ass.events}

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
    print()
    print("=" * 46)
    print("质量检查汇总")
    print("=" * 46)
    print(f"事件总数：{report.stats.get('total_events', 0)}")
    print(f"已翻译：{report.stats.get('translated', 0)}")
    print(f"问题数：{report.stats.get('issue_count', 0)}")
    print(f"错误数：{report.stats.get('error_count', 0)}")
    src_count = {}
    for rec in originals.values():
        s = rec.get("source", "none")
        src_count[s] = src_count.get(s, 0) + 1
    print("原文来源统计：", src_count)


def parse_args(argv):
    p = argparse.ArgumentParser(description="AI 字幕翻译流水线")
    p.add_argument("--ass", help="预处理 ASS 文件")
    p.add_argument("--video", help="原始视频文件")
    p.add_argument("--out", help="输出 ASS 文件")
    p.add_argument("--config", help="配置文件路径")
    p.add_argument("--no-ocr", action="store_true", help="关闭 OCR")
    p.add_argument("--no-asr", action="store_true", help="关闭语音识别")
    p.add_argument("--no-translate", action="store_true", help="关闭翻译（仅识别原文）")
    p.add_argument("--refresh", action="store_true", help="忽略缓存重新提取")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    cfg = Config.load(args.config)
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
    if args.refresh:
        cfg.force_refresh = True

    t0 = time.time()
    try:
        out = build_pipeline(cfg)
    except Exception as exc:
        log(f"流水线执行失败：{exc}")
        import traceback
        traceback.print_exc()
        return 1
    log(f"全部完成，耗时 {time.time() - t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
