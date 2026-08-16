#!/usr/bin/env python3
"""Memory-conscious IndexTTS-2.5 inference for Google Colab L4."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from indextts.infer_v2_5 import IndexTTS2


def parse_vector(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    try:
        values = json.loads(raw) if raw.startswith("[") else [float(x) for x in raw.split(",")]
        values = [float(x) for x in values]
    except Exception as exc:
        raise ValueError("emo_vector must be 8 comma-separated numbers or a JSON array") from exc
    if len(values) != 8 or any(not 0 <= x <= 1 for x in values):
        raise ValueError("emo_vector must contain exactly 8 values, each between 0 and 1")
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prompt_wav", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--lang", choices=["ZH", "EN", "JA", "ES", "AR"], default="ZH")
    p.add_argument("--output", required=True)
    p.add_argument("--duration_factor", type=float, default=1.0)
    p.add_argument("--emo_vector", default="")
    p.add_argument("--emo_audio", default="")
    p.add_argument("--emo_alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    if not 0.5 <= args.duration_factor <= 2.0:
        raise ValueError("duration_factor must be between 0.5 and 2.0")
    if not 0.0 <= args.emo_alpha <= 1.0:
        raise ValueError("emo_alpha must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("BF16:", use_bf16)

    model_dir = Path(args.model_dir)
    tts = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        use_bf16=use_bf16,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
    )

    vector = parse_vector(args.emo_vector)
    if vector is not None:
        vector = tts.normalize_emo_vec(vector, apply_bias=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tts.infer(
        spk_audio_prompt=args.prompt_wav,
        text=args.text,
        lang=args.lang,
        output_path=str(output),
        emo_audio_prompt=args.emo_audio or None,
        emo_vector=vector,
        emo_alpha=args.emo_alpha,
        use_random=False,
        duration_factor=args.duration_factor,
        verbose=True,
    )
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
