"""OCR 识别后端。支持 rapidocr / tesseract / windows / llm 多种实现。"""
from __future__ import annotations

import base64
import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class OCRResult:
    text: str
    confidence: Optional[float]  # 0~1，None 表示未知
    backend: str


class OCRBackend:
    name = "base"

    def recognize(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError


class RapidOCRBackend(OCRBackend):
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._engine = RapidOCR()

    def recognize(self, image: np.ndarray) -> OCRResult:
        result, _ = self._engine(image)
        lines: list[str] = []
        scores: list[float] = []
        if result:
            for box, text, score in result:
                lines.append(str(text))
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    pass
        text = "".join(lines)
        conf = float(np.mean(scores)) if scores else None
        return OCRResult(text=text, confidence=conf, backend=self.name)


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def __init__(self, lang: str = "ja", gpu: bool = True):
        import easyocr
        self._lang = lang
        self._reader = easyocr.Reader([lang], gpu=gpu, verbose=False)

    def recognize(self, image: np.ndarray) -> OCRResult:
        results = self._reader.readtext(image, detail=1, paragraph=False)
        lines: list[str] = []
        scores: list[float] = []
        for item in results:
            text = str(item[1])
            try:
                score = float(item[2])
            except (TypeError, ValueError, IndexError):
                score = None
            if text.strip():
                lines.append(text)
                if score is not None:
                    scores.append(score)
        text = "".join(lines)
        conf = float(np.mean(scores)) if scores else None
        return OCRResult(text=text, confidence=conf, backend=self.name)


class TesseractBackend(OCRBackend):
    name = "tesseract"

    def __init__(self, lang: str = "jpn"):
        import pytesseract
        self._pt = pytesseract
        self.lang = lang

    def recognize(self, image: np.ndarray) -> OCRResult:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self._pt.image_to_data(rgb, lang=self.lang, output_type=self._pt.Output.DICT)
        lines: list[str] = []
        confs: list[float] = []
        for word, conf in zip(data["text"], data["conf"]):
            word = str(word).strip()
            if not word:
                continue
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if c < 0:
                continue
            lines.append(word)
            confs.append(c / 100.0)
        text = "".join(lines)
        conf = float(np.mean(confs)) if confs else None
        return OCRResult(text=text, confidence=conf, backend=self.name)


class WindowsOCRBackend(OCRBackend):
    """通过 PowerShell 调用 Windows 内置 OCR。

    需要 Windows 10/11 且已安装对应语言包，通常需要 pwsh 7 才能方便地使用 WinRT。
    """
    name = "windows"

    def __init__(self, lang: str = "ja"):
        self.lang = lang

    def recognize(self, image: np.ndarray) -> OCRResult:
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "ocr.png"
        cv2.imwrite(str(tmp), image)
        script = _WINDOWS_OCR_PS.replace("__IMAGE__", str(tmp).replace("'", "''")).replace("__LANG__", self.lang)
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Windows OCR 失败: {proc.stderr.strip()}")
        text = proc.stdout.strip()
        return OCRResult(text=text, confidence=None, backend=self.name)


class LLMVisionBackend(OCRBackend):
    """把字幕画面发给多模态大模型进行 OCR。"""
    name = "llm"

    def __init__(self, client, model: str, source_lang: str = "ja"):
        self.client = client
        self.model = model
        self.source_lang = source_lang

    def recognize(self, image: np.ndarray) -> OCRResult:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("图像编码失败")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"这是视频画面底部的字幕区域。请只输出其中的字幕原文"
                                f"（{self.source_lang}），不要解释，不要加引号，"
                                "如果没有任何文字就输出空。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return OCRResult(text=text, confidence=None, backend=self.name)


_WINDOWS_OCR_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Foundation.IAsyncOperation`1, Windows.Foundation, ContentType=WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]
function Await($op, $resultType) {
    $task = $asTaskGeneric.MakeGenericMethod($resultType).Invoke($null, @($op))
    $task.Wait(-1) | Out-Null
    $task.Result
}
$lang = [Windows.Globalization.Language]::new('__LANG__')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
if ($null -eq $engine) { throw "无法创建 OCR 引擎（语言未安装？）" }
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('__IMAGE__')) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$result.Lines | ForEach-Object { $_.Text }
"""


def get_available_backend(config=None) -> str:
    """自动探测可用的 OCR 后端，返回名称。"""
    requested = getattr(config, "ocr_backend", "auto")
    if requested and requested != "auto":
        return requested
    source_lang = getattr(config, "source_lang", "ja")
    # 日语/韩语本地 OCR 模型对动漫等花式字体识别不可靠，优先使用多模态大模型
    if source_lang.split("-")[0] in ("ja", "ko"):
        return "llm"
    if importlib.util.find_spec("easyocr"):
        return "easyocr"
    if importlib.util.find_spec("rapidocr_onnxruntime"):
        return "rapidocr"
    if importlib.util.find_spec("pytesseract"):
        return "tesseract"
    return "llm"


def create_backend(name: str, config=None):
    """按名称创建 OCR 后端。"""
    source_lang = getattr(config, "source_lang", "ja")
    if name == "easyocr":
        gpu = True
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        return EasyOCRBackend(lang=_easyocr_lang(source_lang), gpu=gpu)
    if name == "rapidocr":
        return RapidOCRBackend()
    if name == "tesseract":
        return TesseractBackend(lang=_ocr_lang(source_lang))
    if name == "windows":
        return WindowsOCRBackend(lang=_ocr_lang(source_lang))
    if name == "llm":
        from translate.translator import build_client
        client = build_client(config)
        return LLMVisionBackend(client, config.model, config.source_lang)
    raise ValueError(f"未知 OCR 后端: {name}")


def _ocr_lang(source_lang: str) -> str:
    return {"ja": "jpn", "en": "eng", "zh-CN": "chi_sim", "zh": "chi_sim"}.get(source_lang, "jpn")


def _easyocr_lang(source_lang: str) -> str:
    lang = source_lang.split("-")[0]
    return {"ja": "ja", "ko": "ko", "zh": "ch_sim", "en": "en"}.get(lang, "ja")
