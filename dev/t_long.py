"""拼一首 4 分半的长音频（mp3），测真实长度下的处理时间和显存。"""
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np, soundfile as sf
from app import audio_io

here = Path(__file__).parent
root = here / "testdata" / "MDBDrums" / "MDB Drums" / "audio" / "full_mix"
parts = [audio_io.load_audio(root / f"MusicDelta_{n}_MIX.wav") for n in ("Disco", "FunkJazz", "Britpop", "Grunge", "Punk", "LatinJazz")]
a = np.concatenate(parts, axis=1)
print("duration", a.shape[1] / 44100)
sf.write(str(here / "long.wav"), a.T, 44100, subtype="PCM_16")
subprocess.run([audio_io._ffmpeg(), "-y", "-v", "error", "-i", str(here / "long.wav"), "-b:a", "192k", str(here / "长测试曲.mp3")], check=True)
print("ok")
