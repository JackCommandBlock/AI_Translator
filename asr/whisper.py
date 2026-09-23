"""Whisper / faster-whisper 语音识别。"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ASRWord:
    start_ms: int
    end_ms: int
    word: str


@dataclass
class ASRSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    words: list[ASRWord] = field(default_factory=list)


def _seg_confidence(avg_logprob: float | None) -> float:
    if avg_logprob is None:
        return 0.5
    return min(1.0, max(0.0, math.exp(avg_logprob)))


class ASREngine:
    """统一封装 faster-whisper / openai-whisper。"""

    def __init__(self, config=None):
        self.config = config
        self.model_size = getattr(config, "asr_model_size", "small")
        self.language = _whisper_lang(getattr(config, "source_lang", "ja"))
        device = getattr(config, "asr_device", "auto")
        self._backend = None
        self._model = None
        if device != "auto":
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

    def _load(self):
        try:
            import faster_whisper
            self._load_faster(self.device)
            self._backend = "faster_whisper"
        except ImportError:
            import whisper
            self._model = whisper.load_model(self.model_size)
            self._backend = "whisper"

    def _load_faster(self, device: str):
        import faster_whisper
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = faster_whisper.WhisperModel(
            self.model_size, device=device, compute_type=compute_type
        )

    def transcribe(self, audio_path: str | Path) -> list[ASRSegment]:
        if self._model is None:
            self._load()
        audio_path = str(audio_path)
        if self._backend == "faster_whisper":
            try:
                return self._transcribe_faster(audio_path)
            except RuntimeError as exc:
                if self.device != "cpu" and ("cuda" in str(exc).lower() or "cublas" in str(exc).lower()):
                    # GPU 运行库不匹配（如 cuBLAS 缺失），回退 CPU
                    self.device = "cpu"
                    self._load_faster("cpu")
                    return self._transcribe_faster(audio_path)
                raise
        return self._transcribe_openai(audio_path)

    def _transcribe_faster(self, audio_path: str) -> list[ASRSegment]:
        segments_iter, _info = self._model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        out: list[ASRSegment] = []
        for seg in segments_iter:
            words = [
                ASRWord(int(w.start * 1000), int(w.end * 1000), w.word)
                for w in (seg.words or [])
            ]
            out.append(
                ASRSegment(
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    text=seg.text.strip(),
                    confidence=_seg_confidence(getattr(seg, "avg_logprob", None)),
                    words=words,
                )
            )
        return out

    def _transcribe_openai(self, audio_path: str) -> list[ASRSegment]:
        result = self._model.transcribe(
            audio_path, language=self.language, word_timestamps=True, verbose=False
        )
        out: list[ASRSegment] = []
        for seg in result.get("segments", []):
            words = [
                ASRWord(int(w["start"] * 1000), int(w["end"] * 1000), w["word"])
                for w in seg.get("words", [])
            ]
            out.append(
                ASRSegment(
                    start_ms=int(seg["start"] * 1000),
                    end_ms=int(seg["end"] * 1000),
                    text=seg["text"].strip(),
                    confidence=_seg_confidence(seg.get("avg_logprob")),
                    words=words,
                )
            )
        return out


def _whisper_lang(source_lang: str) -> str:
    if not source_lang:
        return "ja"
    return source_lang.split("-")[0].lower()


def transcribe(audio_path: str | Path, config=None) -> list[ASRSegment]:
    engine = ASREngine(config)
    return engine.transcribe(audio_path)
