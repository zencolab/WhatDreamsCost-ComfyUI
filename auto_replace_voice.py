import os
import re
import shutil
import subprocess
from pathlib import Path

import pysrt
from pydub import AudioSegment
from indextts.infer_v2_5 import IndexTTS2

# 上传单元格会把三个输入文件统一保存到这些位置。
VIDEO = Path('/content/video.mp4')
SRT = Path('/content/subtitles.srt')
VOICE = Path('/content/voice.wav')

OUTPUT = Path('/content/video_new_voice.mp4')
NEW_VOICE_WAV = Path('/content/new_voice_track.wav')
TEMP_DIR = Path('/content/index_dub_temp')
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'缺少文件：{path}。请重新运行“上传三个文件”单元格。')


for input_path in (VIDEO, SRT, VOICE):
    require_file(input_path)


def video_duration_ms(path: Path) -> int:
    value = subprocess.check_output([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path),
    ], text=True).strip()
    return int(float(value) * 1000)


def atempo_chain(rate: float) -> str:
    """把任意速度倍率拆成 FFmpeg atempo 支持的多个倍率。"""
    filters = []
    while rate > 2.0:
        filters.append('atempo=2.0')
        rate /= 2.0
    while rate < 0.5:
        filters.append('atempo=0.5')
        rate /= 0.5
    filters.append(f'atempo={rate:.6f}')
    return ','.join(filters)


def load_srt(path: Path):
    """兼容剪映常见的 UTF-8、UTF-8 BOM 和 GB18030 字幕。"""
    last_error = None
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return pysrt.open(str(path), encoding=encoding)
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(f'字幕编码无法识别：{last_error}')


print('正在加载 IndexTTS 2.5；第一次可能需要下载辅助模型……')
tts = IndexTTS2(
    cfg_path='/content/index-tts/checkpoints/config.yaml',
    model_dir='/content/index-tts/checkpoints',
    use_bf16=True,
)
print('✅ 模型加载完成')


def generate_fitted_line(text: str, target_ms: int, line_no: int) -> AudioSegment:
    """生成一句话，并自动把时长调整到对应字幕时间。"""
    raw_path = TEMP_DIR / f'line_{line_no:04d}_raw.wav'
    fitted_path = TEMP_DIR / f'line_{line_no:04d}_fitted.wav'
    duration_factor = 1.0

    # 先让 IndexTTS 最多尝试 4 次，尽量自然地接近目标时长。
    for attempt in range(1, 5):
        tts.infer(
            spk_audio_prompt=str(VOICE),
            text=text,
            lang='ZH',
            output_path=str(raw_path),
            duration_factor=duration_factor,
            verbose=False,
        )

        actual_ms = len(AudioSegment.from_file(raw_path))
        if actual_ms <= 0:
            raise RuntimeError(f'第 {line_no} 句没有生成有效音频。')

        error_ms = actual_ms - target_ms
        print(
            f'  尝试 {attempt}/4：目标 {target_ms}ms，'
            f'实际 {actual_ms}ms，误差 {error_ms:+d}ms'
        )
        if abs(error_ms) <= 80:
            break

        duration_factor *= target_ms / actual_ms
        duration_factor = max(0.5, min(2.0, duration_factor))

    # 对最后几十毫秒的误差做无变调微调，保证音频正好落入字幕时间段。
    actual_ms = len(AudioSegment.from_file(raw_path))
    if abs(actual_ms - target_ms) > 20:
        speed = actual_ms / target_ms
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(raw_path),
            '-filter:a', atempo_chain(speed),
            str(fitted_path),
        ], check=True)
    else:
        shutil.copyfile(raw_path, fitted_path)

    clip = AudioSegment.from_file(fitted_path).set_frame_rate(44100).set_channels(1)
    if len(clip) < target_ms:
        clip += AudioSegment.silent(duration=target_ms - len(clip), frame_rate=44100)
    elif len(clip) > target_ms:
        clip = clip[:target_ms]

    # 短淡入淡出，避免句子边缘出现爆音。
    return clip.fade_in(10).fade_out(20)


subtitles = load_srt(SRT)
if not subtitles:
    raise RuntimeError('字幕文件中没有可用台词。')

full_track = AudioSegment.silent(
    duration=video_duration_ms(VIDEO),
    frame_rate=44100,
).set_channels(1)

print(f'共读取 {len(subtitles)} 句字幕，开始逐句生成：')
for line_no, subtitle in enumerate(subtitles, start=1):
    start_ms = subtitle.start.ordinal
    end_ms = subtitle.end.ordinal
    target_ms = end_ms - start_ms

    text = re.sub(r'<[^>]+>', '', subtitle.text.replace('\n', ' ')).strip()
    if not text or target_ms < 200:
        print(f'[{line_no}/{len(subtitles)}] 跳过空字幕或过短字幕')
        continue

    print(f'\n[{line_no}/{len(subtitles)}] {text}')
    clip = generate_fitted_line(text, target_ms, line_no)
    full_track = full_track.overlay(clip, position=start_ms)

full_track.export(NEW_VOICE_WAV, format='wav')
print('\n✅ 新人声音轨生成完成，正在替换视频原音轨……')

# 默认丢弃视频中的全部旧声音，只保留新生成的人声。
# 这样不会出现新旧人声重叠；背景音乐也会同时被移除。
subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error',
    '-i', str(VIDEO),
    '-i', str(NEW_VOICE_WAV),
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-movflags', '+faststart',
    '-shortest',
    str(OUTPUT),
], check=True)

print(f'✅ 完成：{OUTPUT}')
