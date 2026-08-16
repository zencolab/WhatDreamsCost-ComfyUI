import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pysrt
import soundfile as sf
from pydub import AudioSegment

SEED_ROOT = Path('/content/seed-vc')
VIDEO = Path('/content/video.mp4')
SRT = Path('/content/subtitles.srt')
TARGET_VOICE = Path('/content/voice.wav')
CONFIG = Path('/content/voice_conversion_config.json')
ORIGINAL_AUDIO = Path('/content/original_audio.wav')
VOCALS = Path('/content/vocals.wav')
BACKGROUND = Path('/content/no_vocals.wav')

OUTPUT = Path('/content/video_new_voice.mp4')
FINAL_AUDIO = Path('/content/final_mix.wav')
FINAL_SRT = Path('/content/subtitles_final.srt')
TEMP_DIR = Path('/content/seed_vc_temp')

DIFFUSION_STEPS = 20
INTELLIGIBILITY = 0.70
SIMILARITY = 0.80
CONTEXT_MS = 100
SEPARATOR_MS = 300


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'缺少文件：{path}')


for required in (VIDEO, SRT, TARGET_VOICE, CONFIG, ORIGINAL_AUDIO, VOCALS, BACKGROUND):
    require_file(required)
if not SEED_ROOT.exists():
    raise FileNotFoundError('缺少 /content/seed-vc，请重新运行步骤2。')

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(parents=True)


def parse_line_spec(value) -> list[int]:
    if isinstance(value, list):
        numbers = [int(item) for item in value]
    else:
        text = str(value).strip().replace('，', ',').replace('—', '-').replace('–', '-')
        numbers = []
        for part in filter(None, (piece.strip() for piece in text.split(','))):
            if '-' in part:
                left, right = part.split('-', 1)
                start, end = int(left), int(right)
                if end < start:
                    start, end = end, start
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(part))
    return sorted(set(numbers))


def load_srt(path: Path):
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return pysrt.open(str(path), encoding=encoding)
        except UnicodeError:
            pass
    raise RuntimeError('字幕编码无法识别。')


def pad_track(track: AudioSegment, duration_ms: int) -> AudioSegment:
    track = track.set_frame_rate(44100).set_channels(2)
    if len(track) < duration_ms:
        tail = AudioSegment.silent(duration=duration_ms - len(track), frame_rate=44100).set_channels(2)
        track += tail
    return track[:duration_ms]


def atempo_chain(rate: float) -> str:
    filters = []
    while rate > 2.0:
        filters.append('atempo=2.0')
        rate /= 2.0
    while rate < 0.5:
        filters.append('atempo=0.5')
        rate /= 0.5
    filters.append(f'atempo={rate:.8f}')
    return ','.join(filters)


def fit_exact_duration(clip: AudioSegment, target_ms: int, name: str) -> AudioSegment:
    if target_ms <= 0 or len(clip) <= 0:
        raise ValueError(f'{name}的音频时长无效。')
    delta = len(clip) - target_ms
    if abs(delta) <= 20:
        if len(clip) < target_ms:
            tail = AudioSegment.silent(duration=target_ms - len(clip), frame_rate=clip.frame_rate).set_channels(clip.channels)
            clip += tail
        return clip[:target_ms]

    source_path = TEMP_DIR / f'{name}_fit_in.wav'
    output_path = TEMP_DIR / f'{name}_fit_out.wav'
    clip.export(source_path, format='wav')
    speed = len(clip) / target_ms
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-i', str(source_path),
        '-filter:a', atempo_chain(speed), str(output_path),
    ], check=True)
    fitted = AudioSegment.from_file(output_path).set_frame_rate(clip.frame_rate).set_channels(clip.channels)
    if len(fitted) < target_ms:
        tail = AudioSegment.silent(duration=target_ms - len(fitted), frame_rate=fitted.frame_rate).set_channels(fitted.channels)
        fitted += tail
    return fitted[:target_ms]


def match_loudness(clip: AudioSegment, reference: AudioSegment) -> AudioSegment:
    if math.isfinite(clip.dBFS) and math.isfinite(reference.dBFS):
        gain = max(-7.0, min(7.0, reference.dBFS - clip.dBFS))
        clip = clip.apply_gain(gain)
    return clip


config = json.loads(CONFIG.read_text(encoding='utf-8'))
target_lines = parse_line_spec(config.get('target_lines', ''))
if not target_lines:
    raise ValueError('TARGET_LINES为空。请在步骤4填写需要更换音色的人物台词编号。')

subtitles = load_srt(SRT)
by_index = {row.index: row for row in subtitles}
missing = [number for number in target_lines if number not in by_index]
if missing:
    raise ValueError(f'这些字幕编号不存在：{missing}')
selected = sorted((by_index[number] for number in target_lines), key=lambda row: row.start.ordinal)
for previous, current in zip(selected, selected[1:]):
    if current.start.ordinal < previous.end.ordinal:
        raise ValueError(f'目标字幕 #{previous.index} 与 #{current.index} 时间重叠，请只保留其中一条。')

video_ms = len(AudioSegment.from_file(ORIGINAL_AUDIO))
original_audio = pad_track(AudioSegment.from_file(ORIGINAL_AUDIO), video_ms)
vocals_mix = pad_track(AudioSegment.from_file(VOCALS), video_ms)
background = pad_track(AudioSegment.from_file(BACKGROUND), video_ms)
vocals_vc = AudioSegment.from_file(VOCALS).set_frame_rate(22050).set_channels(1)

separator = AudioSegment.silent(duration=SEPARATOR_MS, frame_rate=22050).set_channels(1)
source_concat = separator
manifest = []
print('将更换以下台词的音色；文字、停顿、语速和抑扬顿挫来自原声音频：')
for row in selected:
    start_ms = max(0, min(video_ms, row.start.ordinal))
    end_ms = max(start_ms + 1, min(video_ms, row.end.ordinal))
    extended_start = max(0, start_ms - CONTEXT_MS)
    extended_end = min(video_ms, end_ms + CONTEXT_MS)
    source_segment = vocals_vc[extended_start:extended_end]
    if not math.isfinite(source_segment.dBFS):
        print(f'⚠️ 跳过 #{row.index}：分离后没有检测到有效人声。')
        continue

    concat_segment_start = len(source_concat)
    source_concat += source_segment
    core_start = concat_segment_start + (start_ms - extended_start)
    core_end = core_start + (end_ms - start_ms)
    manifest.append({
        'line_no': row.index,
        'start_ms': start_ms,
        'end_ms': end_ms,
        'concat_core_start': core_start,
        'concat_core_end': core_end,
        'text': row.text.replace('\n', ' '),
    })
    source_concat += separator
    print(f'  #{row.index}  {row.start} --> {row.end}  |  {row.text.replace(chr(10), " ")}')

if not manifest:
    raise RuntimeError('选中的字幕均未检测到有效人声。')

source_path = TEMP_DIR / 'selected_source.wav'
converted_path = TEMP_DIR / 'selected_converted_raw.wav'
source_concat.export(source_path, format='wav')

# Seed-VC V2：convert_style=False只转换音色，不转换源说话风格。
os.environ.setdefault('HF_HUB_CACHE', str(SEED_ROOT / 'checkpoints' / 'hf_cache'))
os.chdir(SEED_ROOT)
sys.path.insert(0, str(SEED_ROOT))
import torch
from inference_v2 import load_v2_models

args = SimpleNamespace(
    ar_checkpoint_path=None,
    cfm_checkpoint_path=None,
    compile=False,
)
print('正在加载 Seed-VC V2；首次运行会自动下载免费模型……')
vc_model = load_v2_models(args)
print('✅ Seed-VC V2加载完成')

full_audio = None
for _, completed in vc_model.convert_voice_with_streaming(
    source_audio_path=str(source_path),
    target_audio_path=str(TARGET_VOICE),
    diffusion_steps=DIFFUSION_STEPS,
    length_adjust=1.0,
    intelligebility_cfg_rate=INTELLIGIBILITY,
    similarity_cfg_rate=SIMILARITY,
    top_p=0.9,
    temperature=1.0,
    repetition_penalty=1.0,
    convert_style=False,
    anonymization_only=False,
    device=torch.device('cuda'),
    dtype=torch.float16,
    stream_output=True,
):
    if completed is not None:
        full_audio = completed

if full_audio is None:
    raise RuntimeError('Seed-VC没有返回转换音频。')
output_sr, output_wave = full_audio
sf.write(converted_path, np.asarray(output_wave, dtype=np.float32), int(output_sr))

converted_concat = AudioSegment.from_file(converted_path).set_frame_rate(22050).set_channels(1)
converted_concat = fit_exact_duration(converted_concat, len(source_concat), 'all_selected')
converted_concat = converted_concat.set_frame_rate(44100).set_channels(2)

# 只替换选中台词的人声；其余人物和时间段保持原视频音频。
final_mix = original_audio
for item in manifest:
    start_ms = item['start_ms']
    end_ms = item['end_ms']
    target_ms = end_ms - start_ms
    clip = converted_concat[item['concat_core_start']:item['concat_core_end']]
    clip = fit_exact_duration(clip, target_ms, f"line_{item['line_no']}")
    clip = clip.set_frame_rate(44100).set_channels(2)
    clip = match_loudness(clip, vocals_mix[start_ms:end_ms])
    clip = clip.fade_in(min(18, max(1, target_ms // 4))).fade_out(min(28, max(1, target_ms // 4)))

    final_mix = final_mix[:start_ms] + background[start_ms:end_ms] + final_mix[end_ms:]
    final_mix = final_mix.overlay(clip, position=start_ms)

final_mix.export(FINAL_AUDIO, format='wav')
shutil.copyfile(SRT, FINAL_SRT)
subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error', '-i', str(VIDEO), '-i', str(FINAL_AUDIO),
    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
    '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', str(OUTPUT),
], check=True)
print('✅ 已完成：选中人物的音色已替换，原说话内容、节奏、快慢、停顿和抑扬顿挫保持不变。')
print(f'✅ 输出：{OUTPUT}')
