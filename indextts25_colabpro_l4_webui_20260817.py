#!/usr/bin/env python3
"""Launch the official IndexTTS-2.5 WebUI with a Colab share URL."""
import argparse
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model_dir", required=True)
p.add_argument("--port", type=int, default=7860)
args = p.parse_args()

repo_dir = Path(__file__).resolve().parent
os.chdir(repo_dir)
sys.argv = [
    "webui.py",
    "--version", "2.5",
    "--model_dir", args.model_dir,
    "--fp16",
    "--host", "0.0.0.0",
    "--port", str(args.port),
]

import webui

webui.demo.queue(20)
webui.demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=args.port,
    show_error=True,
)
