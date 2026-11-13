"""逐拍打印鼓轨/原曲在各频段的击打能量，直接从声音上看底鼓、军鼓落在哪些拍。
用法：python dev/t_bands.py <项目id> <起始秒> <结束秒>
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app import audio_io, beats, drums

d = Path(__file__).resolve().parent.parent / "projects" / sys.argv[1]
t0, t1 = float(sys.argv[2]), float(sys.argv[3])
an = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
g = beats.build_grid(np.array(an["beats"]), np.array(an["downbeats"]), an["duration"])
z = np.load(d / "drum_acts.npz")
st, mx = z["stem"].astype(float), z["mix"].astype(float)
for name in ("drums", "other", "mix"):
    a, _ = audio_io.read_audio_file(d / f"{name}.flac")
    F = drums.StemFeatures(a.mean(0))
    flux = lambda env: np.maximum(0, np.diff(env, prepend=env[0]))
    bands = {"low": F.low, "mid": F.mid, "snr": F.snr, "hf": F.hf}
    ref = {k: np.percentile(v, 99) for k, v in bands.items()}
    print(f"== {name}: 每半拍的击打能量(0-9)  low/mid/snr/hf" + ("   | ADTOF 鼓轨 KD SD TT HH CY / 原曲 SD" if name == "drums" else ""))
    for bi, (s, n) in enumerate(g.bars):
        bt = g.beat_times[s]
        if bt < t0 or bt > t1:
            continue
        cells = []
        for h in range(n * 2):
            t = float(g.time(s + h / 2))
            k = int(round(t * 100))
            v = [min(9, int(9 * flux(bands[b])[max(0, k - 2): k + 4].max() / (0.5 * ref[b]))) for b in bands]
            cell = "".join(map(str, v))
            if name == "drums":
                a_ = st[max(0, k - 3): k + 4].max(0); m_ = mx[max(0, k - 3): k + 4, 1].max()
                cell += "|" + "".join(str(min(9, int(x * 10))) for x in a_) + "/" + str(min(9, int(m_ * 10)))
            cells.append(cell)
        print(f"{bi+1:3d} [{bt:6.1f}] " + "  ".join(cells))
