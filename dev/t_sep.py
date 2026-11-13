import sys, time, glob, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from app import audio_io, separate, beats

root = os.path.join(os.path.dirname(__file__), "testdata", "MDBDrums", "MDB Drums")
name = sys.argv[1] if len(sys.argv) > 1 else "MusicDelta_Rock"
f = os.path.join(root, "audio", "full_mix", name + "_MIX.wav")
t0 = time.time()
a = audio_io.load_audio(f)
print("loaded", a.shape, a.shape[1] / 44100, "s", time.time() - t0)
t0 = time.time()
st = separate.separate(a, progress=lambda p, m: print(" ", p, m))
print("separated", time.time() - t0)
for k, v in st.items():
    print(k, round(audio_io.rms_db(v), 1), "dB")
print("sum err", float(np.abs(sum(st.values()) - a).max()))
out = os.path.join(os.path.dirname(__file__), "out", name)
for k, v in st.items():
    audio_io.save_flac(os.path.join(out, k + ".flac"), v)
t0 = time.time()
b, d = beats.track(a.mean(0), 44100)
print("beats", len(b), "downbeats", len(d), time.time() - t0)
ann = np.loadtxt(os.path.join(root, "annotations", "beats", name + "_MIX.beats"))
print("ann beats", len(ann), ann[:6, 0], "det", b[:6], "db", d[:4])
g = beats.build_grid(b, d, a.shape[1] / 44100)
print("grid N", g.numerator, "tempo", g.tempo, "bars", len(g.bars), g.bars[:5], "anacrusis", g.anacrusis)
