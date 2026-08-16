# IndexTTS-2.5 on Google Colab Pro（L4 / UV 版）

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zencolab/WhatDreamsCost-ComfyUI/blob/main/IndexTTS2_5_ColabPro_L4_UV_20260817.ipynb)

适用于 **Google Colab Pro + NVIDIA L4** 的 IndexTTS-2.5 安装与使用 Notebook。

## 文件

- [`IndexTTS2_5_ColabPro_L4_UV_20260817.ipynb`](IndexTTS2_5_ColabPro_L4_UV_20260817.ipynb) — 完整、可逐格运行的 Colab Notebook

## 功能

- 固定官方 `v2.5.0` 发布版本
- 使用 `uv sync --extra webui --frozen` 安装官方锁定依赖
- L4 上自动启用 BF16
- 自动下载 IndexTTS-2.5 主模型与辅助模型
- 支持中文、英文、日语、西班牙语、阿拉伯语
- 支持音色克隆、独立情感音频、8 维情感向量、语速控制与发音标注
- 可选 Google Drive 模型缓存
- 可选 Gradio 公网临时 WebUI

## 快速开始

1. 点击上方 **在 Colab 中打开**。
2. 在 Colab 选择「运行时 → 更改运行时类型 → GPU」，优先选择 **L4**。
3. 从上到下运行单元格。
4. 上传一段 5–15 秒、已获授权的清晰参考音频。
5. 修改文本、语言和可选情感参数，运行推理单元格。

首次安装和模型下载耗时较长；可在 Notebook 中开启 Google Drive 缓存以跨会话复用模型。

## 版本与来源

- IndexTTS 代码：<https://github.com/index-tts/index-tts>
- 中文文档：<https://github.com/index-tts/index-tts/blob/main/docs/README_zh.md>
- 固定发布：[`v2.5.0`](https://github.com/index-tts/index-tts/releases/tag/v2.5.0)
- 模型：<https://huggingface.co/IndexTeam/IndexTTS-2.5>

## 合规与许可

只使用本人或已获明确授权的声音。请遵守适用法律，并在使用前阅读上游仓库的 `LICENSE`、`LICENSE_ZH.txt` 和 `DISCLAIMER`。本仓库不包含或重新分发模型权重。
