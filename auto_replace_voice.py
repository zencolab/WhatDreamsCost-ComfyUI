import copy
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pysrt
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from indextts.infer_v2_5 import IndexTTS2

VIDEO = Path('/content/video.mp4')
SRT = Path('/content/subtitles.srt')
VOICE = Path('/content/voice.wav')
EDITS = Path('/content/dialogue_edits.json')
ORIGINAL_AUDIO = Path('/content/original_audio.wav')
VOCALS = Path('/content/vocals.wav')
BACKGROUND = Path('/content/no_vocals.wav')

OUTPUT = Path('/content/video_new_voice.mp4')
FINAL_SRT = Path('/content/subtitles_final.srt')
FINAL_AUDIO = Path('/content/final_mix.wav')
TEMP_DIR = Path('/content/index_dub_temp')
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(parents=True)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'缺少文件：{path}。请从步骤3重新运行。')


for input_path in (VIDEO, SRT, VOICE, ORIGINAL_AUDIO, VOCALS, BACKGROUND):
    require_file(input_path)


def video_duration_ms(path: Path) -> int:
    value = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(path),
    ], text=True).strip()
    return int(float(value) * 1000)


def load_srt(path: Path):
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return pysrt.open(str(path), encoding=encoding)
        except UnicodeError:
            pass
    raise RuntimeError('字幕编码无法识别。')


def parse_time(value: str):
    try:
        return pysrt.SubRipTime.from_string(value.strip().replace('.', ','))
    except Exception as exc:
        raise ValueError(f'时间格式错误：{value}；应为 00:00:12,500') from exc


def clean_text(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text.replace('\n', ' ')).strip()


def pad_track(track: AudioSegment, duration_ms: int) -> AudioSegment:
    track = track.set_frame_rate(44100).set_channels(2)
    if len(track) < duration_ms:
        track += AudioSegment.silent(duration=duration_ms - len(track), frame_rate=44100).set_channels(2)
    return track[:duration_ms]


def silence(duration_ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=max(0, duration_ms), frame_rate=44100).set_channels(2)


def mute_range(track: AudioSegment, start_ms: int, end_ms: int) -> AudioSegment:
    start_ms = max(0, start_ms)
    end_ms = min(len(track), end_ms)
    return track[:start_ms] + silence(end_ms - start_ms) + track[end_ms:]


def atempo_chain(rate: float) -> str:
    filters = []
    while rate > 2.0:
        filters.append('atempo=2.0')
        rate /= 2.0
    while rate < 0.5:
        filters.append('atempo=0.5')
        rate /= 0.5
    filters.append(f'atempo={rate:.6f}')
    return ','.join(filters)


def match_loudness(clip: AudioSegment, reference: AudioSegment) -> AudioSegment:
    if math.isfinite(clip.dBFS) and math.isfinite(reference.dBFS):
        gain = max(-8.0, min(8.0, reference.dBFS - clip.dBFS))
        clip = clip.apply_gain(gain)
    return clip


def trim_outer_silence(clip: AudioSegment) -> AudioSegment:
    """去掉TTS首尾多余静音，先减少时长，再决定是否变速。"""
    if not math.isfinite(clip.dBFS):
        return clip
    ranges = detect_nonsilent(
        clip,
        min_silence_len=40,
        silence_thresh=max(-55.0, clip.dBFS - 18.0),
    )
    if not ranges:
        return clip
    start = max(0, ranges[0][0] - 20)
    end = min(len(clip), ranges[-1][1] + 40)
    return clip[start:end]


def remux_original_audio():
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-i', str(VIDEO),
        '-map', '0:v:0', '-map', '0:a:0?', '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(OUTPUT),
    ], check=True)


video_ms = video_duration_ms(VIDEO)
original_subtitles = load_srt(SRT)
if not original_subtitles:
    raise RuntimeError('字幕文件中没有可用台词。')

config = {'corrections': {}, 'additions': []}
if EDITS.exists():
    config.update(json.loads(EDITS.read_text(encoding='utf-8')))
corrections = config.get('corrections') or {}
additions = config.get('additions') or []

# 构建“只修改指定台词”的任务；其他原声完全保留。
original_by_index = {row.index: row for row in original_subtitles}
final_subtitles = copy.deepcopy(original_subtitles)
final_by_index = {row.index: row for row in final_subtitles}
jobs = []

# 每句最多可自然延伸到下一句前100ms，优先利用空白时间而不是强行加速。
ordered_subtitles = sorted(original_subtitles, key=lambda row: row.start.ordinal)
max_end_by_index = {}
for position, row in enumerate(ordered_subtitles):
    next_start = (
        ordered_subtitles[position + 1].start.ordinal
        if position + 1 < len(ordered_subtitles)
        else video_ms
    )
    max_end_by_index[row.index] = max(row.end.ordinal, next_start - 100)

for key, value in corrections.items():
    line_no = int(key)
    if line_no not in original_by_index:
        raise ValueError(f'修正编号 #{line_no} 不存在。')
    text = str(value).strip()
    if not text:
        raise ValueError(f'第 #{line_no} 条修正文字不能为空。')
    source = original_by_index[line_no]
    final_by_index[line_no].text = text
    jobs.append({
        'kind': 'correction', 'name': f'修正#{line_no}', 'line_no': line_no,
        'start': source.start.ordinal, 'end': source.end.ordinal,
        'max_end': max_end_by_index[line_no], 'text': text,
    })
    print(f'修正 #{line_no}：{clean_text(source.text)} → {text}')

occupied = [(row.start.ordinal, row.end.ordinal, f'原字幕#{row.index}') for row in original_subtitles]
for number, item in enumerate(additions, start=1):
    start = parse_time(str(item['start']))
    end = parse_time(str(item['end']))
    text = str(item['text']).strip()
    if not text or end.ordinal <= start.ordinal or end.ordinal > video_ms:
        raise ValueError(f'新增台词配置无效：{item}')
    for old_start, old_end, label in occupied:
        if max(start.ordinal, old_start) < min(end.ordinal, old_end):
            raise ValueError(f'新增台词“{text}”与{label}重叠，请选择无人说话的空白时间。')
    occupied.append((start.ordinal, end.ordinal, f'新增#{number}'))
    final_subtitles.append(pysrt.SubRipItem(index=0, start=start, end=end, text=text))
    jobs.append({
        'kind': 'addition', 'name': f'新增#{number}', 'line_no': 10000 + number,
        'start': start.ordinal, 'end': end.ordinal, 'max_end': end.ordinal, 'text': text,
    })
    print(f'新增：{start} --> {end} | {text}')

final_subtitles.sort(key=lambda row: row.start.ordinal)
for index, row in enumerate(final_subtitles, start=1):
    row.index = index
final_subtitles.save(str(FINAL_SRT), encoding='utf-8')

if not jobs:
    print('没有填写修正或新增台词，保留原视频全部声音。')
    remux_original_audio()
    print(f'✅ 完成：{OUTPUT}')
    raise SystemExit(0)

original_audio = pad_track(AudioSegment.from_file(ORIGINAL_AUDIO), video_ms)
vocals = pad_track(AudioSegment.from_file(VOCALS), video_ms)
background = pad_track(AudioSegment.from_file(BACKGROUND), video_ms)
final_mix = original_audio

print('正在加载 IndexTTS 2.5；只生成被修改或新增的台词……')
tts = IndexTTS2(
    cfg_path='/content/index-tts/checkpoints/config.yaml',
    model_dir='/content/index-tts/checkpoints',
    use_bf16=True,
    use_cuda_kernel=False,
)
print('✅ 模型加载完成')


def make_emotion_prompt(job) -> Path | None:
    if job['kind'] != 'correction':
        return None
    # 用原句声音作为情绪参考，尽量保留原来的抑扬顿挫和语气。
    start = max(0, job['start'] - 300)
    end = min(video_ms, job['end'] + 300)
    segment = vocals[start:end].set_channels(1).set_frame_rate(24000)
    if len(segment) < 1000:
        segment += AudioSegment.silent(duration=1000 - len(segment), frame_rate=24000)
    path = TEMP_DIR / f"emotion_{job['line_no']}.wav"
    segment.export(path, format='wav')
    return path


def generate_natural_line(job, emotion_prompt: Path | None) -> AudioSegment:
    target_ms = job['end'] - job['start']
    available_ms = max(target_ms, job['max_end'] - job['start'])
    raw = TEMP_DIR / f"line_{job['line_no']}_raw.wav"
    fitted = TEMP_DIR / f"line_{job['line_no']}_fitted.wav"
    duration_factor = 1.0

    # 最多生成两次；如果后面有空白，允许自然延长，不强压进原字幕长度。
    for attempt in range(1, 3):
        tts.infer(
            spk_audio_prompt=str(VOICE),
            emo_audio_prompt=str(emotion_prompt) if emotion_prompt else None,
            emo_alpha=0.85 if emotion_prompt else 1.0,
            text=job['text'], lang='ZH', output_path=str(raw),
            duration_factor=duration_factor, interval_silence=0,
            verbose=False,
        )
        actual_ms = len(AudioSegment.from_file(raw))
        if actual_ms <= 0:
            raise RuntimeError(f"{job['name']}没有生成有效音频。")
        print(
            f"{job['name']} 尝试{attempt}：原时长{target_ms}ms，"
            f"可用{available_ms}ms，生成{actual_ms}ms"
        )
        if actual_ms <= target_ms * 1.08 or actual_ms <= available_ms:
            break
        duration_factor = max(0.80, min(1.20, duration_factor * available_ms / actual_ms))

    # 先裁掉模型在首尾生成的静音，避免把停顿误算成说话时长。
    generated = trim_outer_silence(AudioSegment.from_file(raw))
    trimmed = TEMP_DIR / f"line_{job['line_no']}_trimmed.wav"
    generated.export(trimmed, format='wav')
    actual_ms = len(generated)

    if actual_ms > available_ms:
        # 没有空白时间时仍继续完成，不再抛错中断；只对这一句做精确变速。
        speed = actual_ms / available_ms
        print(
            f"⚠️ {job['name']}没有可用空白，裁静音后仍需压缩{speed:.2f}倍；"
            '将自动处理并继续生成视频。'
        )
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error', '-i', str(trimmed),
            '-filter:a', atempo_chain(speed), str(fitted),
        ], check=True)
    else:
        generated.export(fitted, format='wav')

    clip = AudioSegment.from_file(fitted).set_frame_rate(44100).set_channels(2)
    reference = vocals[job['start']:job['end']] if job['kind'] == 'correction' else AudioSegment.from_file(VOICE)
    clip = match_loudness(clip, reference)
    clip = clip[:available_ms].fade_in(15).fade_out(25)
    job['render_end'] = job['start'] + len(clip)
    return clip


for job in jobs:
    prompt = make_emotion_prompt(job)
    clip = generate_natural_line(job, prompt)
    if job['kind'] == 'correction':
        mute_start = max(0, job['start'] - 60)
        mute_end = min(video_ms, max(job['end'], job['render_end']) + 80)
        final_mix = (
            final_mix[:mute_start]
            + background[mute_start:mute_end]
            + final_mix[mute_end:]
        )
    final_mix = final_mix.overlay(clip, position=job['start'])

# 只有修改区间使用分离后的背景轨；其他区间保持原视频音频不变。
final_mix.export(FINAL_AUDIO, format='wav')
subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error', '-i', str(VIDEO), '-i', str(FINAL_AUDIO),
    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
    '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', str(OUTPUT),
], check=True)
print('✅ 已保留原声语气、其他人物、背景音乐和音效。')
print(f'✅ 完成：{OUTPUT}')
