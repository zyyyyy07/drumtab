"""对 MDB Drums 全部曲目：分离鼓轨、算 ADTOF 概率、拍点，缓存到磁盘供调参。"""
import sys, os, glob, time
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from app import audio_io, separate, drums, beats

here = Path(__file__).parent
root = here / "testdata" / "MDBDrums" / "MDB Drums"
cache = here / "cache"
cache.mkdir(exist_ok=True)
files = sorted(glob.glob(str(root / "audio" / "full_mix" / "*_MIX.wav")))
t0 = time.time()
for f in files:
    name = Path(f).name.replace("_MIX.wav", "")
    if (cache / f"{name}.npz").exists():
        continue
    a = audio_io.load_audio(f)
    st = separate.separate(a)
    dm = st["drums"].mean(0)
    audio_io.save_flac(cache / f"{name}_drums.flac", dm)
    ref = audio_io.load_audio(root / "audio" / "drum_only" / f"{name}_Drum.wav").mean(0)
    b, d = beats.track(a.mean(0), 44100)
    np.savez_compressed(cache / f"{name}.npz", stem=drums.activations(dm), mix=drums.activations(a.mean(0)),
                        ref=drums.activations(ref), beats=b, downbeats=d, dur=a.shape[1] / 44100)
    print(name, round(a.shape[1] / 44100, 1), "s", round(time.time() - t0, 1), flush=True)
print("done", time.time() - t0)
