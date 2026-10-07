# AI 字幕翻译流水线

从「原始视频 + 预处理 ASS（只有时间轴/样式，无正文）」生成成品字幕
`translated.ass`。ASS 是时间轴与排版的权威来源，OCR 与 ASR 负责提取原文，
大模型负责纠错与翻译，最终只替换 Dialogue 的 Text。

## 功能

- 解析 ASS，保留 Script Info / Styles / Format / 特效标签与时间轴。
- OCR：默认全帧多文本识别，把画面中所有文字按内容类型（对白/标题/说明/招牌/歌词）
  拆成文本轨道，再用时间覆盖与 ASR 相似度匹配回事件；兼容旧的底部裁剪模式
  （`ocr_mode="crop"`）。
- ASR：整段视频语音识别，按词级时间戳对齐回每个 ASS 事件。
- 原文判定：OCR 优先，OCR 失败用 ASR，两者冲突交给大模型裁决。
- 原文纠错：大模型结合上下文修正 OCR/ASR 错字。
- 术语表：保证专有名词与称呼统一。
- 批量上下文翻译：一次翻译多条，返回结构化 JSON，再按 Event ID 写回。
- ASS 特效标签与 `\N` 换行保留。
- 译文长度检查与自动压缩。
- 字幕定位（可选）：根据原字幕包围盒生成候选位置，用 ffmpeg 模拟摆放后交多模态
  大模型评分，选择最优 `\pos` 写回译文。
- 自动质量检查 + 人工审核清单 `work/review.md`。
- 统一日志：记录流水线运行步骤、大模型网络请求（模型 / 消息数 / 耗时 / token
  用量）、ffmpeg/ASR 子流程以及异常堆栈，同时输出到控制台与 `logs/pipeline.log`。

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
python main.py --positioning    # 翻译后为译文字幕判别位置（LLM 评分）
python main.py --refresh       # 忽略缓存重新提取
```

默认配置在 `config.py`。API 配置（base_url / api_key / model）与
`gemini_example.py` 保持一致，也可用环境变量 `AI_TRANSLATOR_API_KEY`
覆盖。

## 日志

日志默认写入 `logs/pipeline.log`，并同时打印到控制台。可通过 `config.py` 调整：

- `log_dir`：日志目录，默认 `logs`。
- `log_file`：日志文件名，默认 `pipeline.log`。
- `log_level`：`DEBUG` / `INFO` / `WARNING` / `ERROR`，默认 `INFO`。
- `log_console`：是否同时输出到控制台，默认 `True`。

大模型请求会记录模型、消息数量、耗时与 token 用量；`DEBUG` 级别还会记录消息摘要
与响应前 500 字符。日志中的 API Key、token 等敏感字段会被自动脱敏。

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
positioning/      字幕定位（候选生成 + ffmpeg 渲染 + LLM 评分）
output/           最终 ASS 导出
logging_utils.py  日志配置与脱敏
glossary.json     术语表
```

## 中间产物（work/）

- `audio.wav`：抽取的 16kHz 单声道音频。
- `frames/`：按事件抽取的代表帧。
- `frames_full/`：全帧检测抽取的采样帧。
- `frame_ocr.json` / `text_timeline.json` / `matched.json`：全帧识别与匹配缓存。
- `ocr.json` / `asr.json`：旧裁剪模式与语音识别缓存（`--refresh` 可重新生成）。
- `original.json` / `original_corrected.json`：判定与纠错后的原文。
- `translated.json`：译文。
- `quality_report.json` / `review.md`：质量报告与人工审核清单。

## 运行时日志（logs/）

- `pipeline.log`：流水线步骤、网络请求、错误堆栈（自动滚动，保留最近 3 个备份）。

