# AI 字幕翻译流水线

从「原始视频 + 预处理 ASS（只有时间轴/样式，无正文）」生成成品字幕
`translated.ass`。ASS 是时间轴与排版的权威来源，OCR 与 ASR 负责提取原文，
大模型负责纠错与翻译，最终只替换 Dialogue 的 Text。

## 功能

- 解析 ASS，保留 Script Info / Styles / Format / 特效标签与时间轴。
- OCR：从视频画面底部字幕区域抽取代表帧，多帧投票得到原文与置信度。
- ASR：整段视频语音识别，按词级时间戳对齐回每个 ASS 事件。
- 原文判定：OCR 优先，OCR 失败用 ASR，两者冲突交给大模型裁决。
- 原文纠错：大模型结合上下文修正 OCR/ASR 错字。
- 术语表：保证专有名词与称呼统一。
- 批量上下文翻译：一次翻译多条，返回结构化 JSON，再按 Event ID 写回。
- ASS 特效标签与 `\N` 换行保留。
- 译文长度检查与自动压缩。
- 自动质量检查 + 人工审核清单 `work/review.md`。

## 环境

- Python 3.11+
- ffmpeg / ffprobe（需在 PATH 中）

安装依赖：

```bash
pip install -r requirements.txt
```

OCR 默认使用 `rapidocr_onnxruntime`（模型随包内置）。ASR 默认使用
`faster-whisper`，首次运行会下载模型权重；模型大小由 `asr_model_size`
控制。若 GPU 可用会优先使用 CUDA。

## 使用

```bash
python main.py
python main.py --ass sample.ass --video sample.mp4 --out translated.ass
python main.py --no-asr        # 关闭语音识别
python main.py --no-translate  # 仅识别原文，不翻译
python main.py --refresh       # 忽略缓存重新提取
```

默认配置在 `config.py`。API 配置（base_url / api_key / model）与
`gemini_example.py` 保持一致，也可用环境变量 `AI_TRANSLATOR_API_KEY`
覆盖。

## 目录结构

```text
main.py           流水线入口
config.py         全局配置
video/            帧抽取、音频抽取、ffmpeg 封装
ass/              ASS 解析、标签处理、写回
ocr/              图像预处理、OCR 识别、多帧汇总
asr/              Whisper/faster-whisper 语音识别
align/            时间轴匹配、字幕分组
translate/        术语表、提示词、大模型翻译
quality/          质量检查、审核清单
output/           最终 ASS 导出
glossary.json     术语表
```

## 中间产物（work/）

- `audio.wav`：抽取的 16kHz 单声道音频。
- `frames/`：按事件抽取的代表帧。
- `ocr.json` / `asr.json`：识别缓存（`--refresh` 可重新生成）。
- `original.json` / `original_corrected.json`：判定与纠错后的原文。
- `translated.json`：译文。
- `quality_report.json` / `review.md`：质量报告与人工审核清单。

