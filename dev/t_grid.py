"""小节网格检查：对 23 首歌打印 主拍号/小节长度分布，并与人工标注的强拍比较。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from t_eval import *
from app import beats
import importlib

for n in names:
    z = np.load(cache / f"{n}.npz")
    raw = drums.detect(z["stem"].astype(np.float32), z["mix"].astype(np.float32)); raw.pop("_low", None)
    kk = np.array([t for t, _ in raw["KD"]])
    g = beats.build_grid(z["beats"], z["downbeats"], float(z["dur"]), kicks=kk if "--kick" in sys.argv else None)
    lens = {}
    for s, k in g.bars:
        lens[k] = lens.get(k, 0) + 1
    ann = np.loadtxt(root / "annotations" / "beats" / f"{n}_MIX.beats")
    ann_db = ann[ann[:, 1] == 1, 0]
    ann_tempo = 60 / np.median(np.diff(ann[:, 0]))
    ann_n = int(ann[:, 1].max())
    bt = np.array([g.bar_time(i) for i in range(len(g.bars))])
    hit = np.mean([np.min(np.abs(bt - t)) < 0.07 for t in ann_db]) if len(ann_db) else 0
    print(f"{n[11:]:13s} N={g.numerator} tempo={g.tempo:6.1f} (标注 {ann_n}/4 @{ann_tempo:6.1f})  bars={len(g.bars):3d} 长度分布={lens} 弱起={g.anacrusis}  标注强拍命中={hit*100:3.0f}%")
