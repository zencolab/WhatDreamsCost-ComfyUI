import shutil
import subprocess
import sys
from pathlib import Path

VIDEO = Path('/content/video.mp4')
ORIGINAL_AUDIO = Path('/content/original_audio.wav')
SEPARATED_ROOT = Path('/content/demucs_output')
VOCALS = Path('/content/vocals.wav')
BACKGROUND = Path('/content/no_vocals.wav')

if not VIDEO.exists():
    raise FileNotFoundError(f'找不到视频：{VIDEO}')

# 提取原视频完整音频，供人声/背景声分离。
subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error',
    '-i', str(VIDEO), '-vn', '-ac', '2', '-ar', '44100',
    '-c:a', 'pcm_s16le', str(ORIGINAL_AUDIO),
], check=True)

if SEPARATED_ROOT.exists():
    shutil.rmtree(SEPARATED_ROOT)

command = [
    sys.executable, '-m', 'demucs',
    '--two-stems=vocals', '-n', 'htdemucs',
    '-d', 'cuda', '--clip-mode', 'clamp',
    '-o', str(SEPARATED_ROOT), str(ORIGINAL_AUDIO),
]
print('正在用免费 Demucs 分离人声、背景音乐和音效……')
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError:
    print('GPU分离失败，自动改用CPU重试。')
    command[command.index('cuda')] = 'cpu'
    subprocess.run(command, check=True)

result_dir = SEPARATED_ROOT / 'htdemucs' / ORIGINAL_AUDIO.stem
source_vocals = result_dir / 'vocals.wav'
source_background = result_dir / 'no_vocals.wav'
if not source_vocals.exists() or not source_background.exists():
    raise RuntimeError(f'Demucs没有生成预期文件：{result_dir}')

shutil.copyfile(source_vocals, VOCALS)
shutil.copyfile(source_background, BACKGROUND)
print(f'✅ 原人声：{VOCALS}')
print(f'✅ 背景音乐和音效：{BACKGROUND}')
