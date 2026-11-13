"""音频读写：任意格式 -> 44.1k 立体声 numpy；numpy -> FLAC（浏览器可直接播放，时间轴无偏移）。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """解码任意音频/视频文件，返回 float32 数组 [2, T]。"""
    cmd = [
        _ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
        "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("无法解码音频文件：" + proc.stderr.decode("utf-8", "ignore")[-400:])
    data = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).T
    return np.ascontiguousarray(data)


def save_flac(path: str | Path, audio: np.ndarray, sr: int = SR) -> None:
    """audio: [C, T] 或 [T]。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    sf.write(str(path), data, sr, format="FLAC", subtype="PCM_16")


def read_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(r + 1e-9)
