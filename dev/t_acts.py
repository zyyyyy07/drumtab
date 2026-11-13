"""看某首歌里各类鼓的概率峰值分布，以及与底鼓同时出现的军鼓/通鼓峰值有多高。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app import drums, pipeline

d = Path(__file__).resolve().parent.parent / "projects" / sys.argv[1]
z = np.load(d / "drum_acts.npz")
stem, mix = z["stem"].astype(np.float32), z["mix"].astype(np.float32)
acts = pipeline.fuse_acts(stem, mix)
n = len(acts)
pk = {c: drums.pick_peaks(acts[:, i], 0.08) for i, c in enumerate(drums.CLASSES)}
kt = np.array([t for t, v in pk["KD"] if v >= 0.2])
for c in drums.CLASSES:
    v = np.array([x for _, x in pk[c]])
    t = np.array([x for x, _ in pk[c]])
    hist = np.histogram(v, bins=[0.08, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])[0]
    line = f"{c}: n={len(v):4d} 分布(0.08..1.0)={hist.tolist()}"
    if c != "KD" and len(t):
        near = np.array([np.min(np.abs(kt - x)) < 0.03 for x in t])
        line += f" | 与底鼓同时: n={near.sum()} 峰值中位={np.median(v[near]) if near.any() else 0:.2f} ; 不同时: n={(~near).sum()} 峰值中位={np.median(v[~near]) if (~near).any() else 0:.2f}"
        # 同一时刻 鼓轨 vs 原曲 的概率
        i = drums.CLASSES.index(c)
        f = np.clip((t * 100).round().astype(int), 0, n - 1)
        win = lambda arr: np.array([arr[max(0, k - 2): k + 3, i].max() for k in f])
        s, m = win(stem), win(mix)
        line += f"\n      与底鼓同时的峰: 鼓轨中位={np.median(s[near]) if near.any() else 0:.2f} 原曲中位={np.median(m[near]) if near.any() else 0:.2f} | 其余: 鼓轨={np.median(s[~near]) if (~near).any() else 0:.2f} 原曲={np.median(m[~near]) if (~near).any() else 0:.2f}"
    print(line)
if len(sys.argv) > 3:
    a, b = float(sys.argv[2]), float(sys.argv[3])
    print(f"--- {a}s ~ {b}s 的峰（类:时间:融合/鼓轨/原曲）")
    rows = []
    for i, c in enumerate(drums.CLASSES):
        for t, v in pk[c]:
            if a <= t <= b:
                k = int(round(t * 100))
                rows.append((t, f"{c}:{t:.2f}:{v:.2f}/{stem[max(0,k-2):k+3, i].max():.2f}/{mix[max(0,k-2):k+3, i].max():.2f}"))
    print("  ".join(r for _, r in sorted(rows)))
