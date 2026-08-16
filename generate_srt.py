import argparse
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel


def srt_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def main():
    parser = argparse.ArgumentParser(description='使用免费 Faster-Whisper 从视频生成 SRT 字幕')
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='large-v3')
    parser.add_argument('--language', default='zh')
    args = parser.parse_args()

    video = Path(args.video)
    output = Path(args.output)
    audio = output.with_name('whisper_audio.wav')
    if not video.exists():
        raise FileNotFoundError(f'找不到视频：{video}')

    # 提取16kHz单声道人声，Whisper识别更稳定。
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(video), '-vn', '-ac', '1', '-ar', '16000',
        str(audio),
    ], check=True)

    print(f'正在加载免费 Faster-Whisper {args.model} 模型……')
    model = WhisperModel(args.model, device='cuda', compute_type='float16')
    segments, info = model.transcribe(
        str(audio),
        language=args.language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={'min_silence_duration_ms': 300},
        condition_on_previous_text=True,
    )

    rows = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            rows.append((segment.start, segment.end, text))
    if not rows:
        raise RuntimeError('没有识别到可用人声，请检查视频是否包含清晰说话声。')

    with output.open('w', encoding='utf-8') as handle:
        for index, (start, end, text) in enumerate(rows, start=1):
            handle.write(f'{index}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n')

    print(f'✅ 免费SRT生成完成：{output}')
    print(f'识别语言：{info.language}，共 {len(rows)} 条字幕')


if __name__ == '__main__':
    main()
