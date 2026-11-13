import sys, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
with contextlib.redirect_stdout(io.StringIO()):
    import t_feat
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------- CY
rows = t_feat.rows_cy
song = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
raw = np.array([r[2:] for r in rows], dtype=float)
# near, lvl/med, lvl/locmed, sus_mid, sus_hf, tilt, mid/hf, logrise, mid/snare
def fx(raw, cols):
    near = np.minimum(raw[:, 0], 8) / 8
    f = {"near": near, "sus_mid": np.log(np.clip(raw[:, 3], .05, 8)), "sus_hf": np.log(np.clip(raw[:, 4], .05, 8)),
         "midhf": np.log(np.clip(raw[:, 6], .2, 10)), "msn": np.log(np.clip(raw[:, 8], .1, 5)), "tilt": np.log(np.clip(raw[:, 5], .05, 10)),
         "loc": np.log(np.clip(raw[:, 2], .2, 5))}
    return np.column_stack([f[c] for c in cols])

for cols in (["near"], ["near", "sus_hf"], ["near", "sus_hf", "sus_mid"], ["near", "sus_hf", "midhf"], ["near", "sus_hf", "midhf", "msn"], ["near", "sus_hf", "sus_mid", "midhf", "msn", "loc"]):
    X = fx(raw, cols)
    pred = np.zeros(len(y), dtype=bool)
    for s in np.unique(song):
        tr = song != s
        m = LogisticRegression(C=1.0, class_weight={False: 1, True: 2}).fit(X[tr], y[tr])
        pred[~tr] = m.predict(X[~tr])
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    m = LogisticRegression(C=1.0, class_weight={False: 1, True: 2}).fit(X, y)
    print(f"CY {cols}: LOSO crash P={tp/(tp+fp+1e-9):.2f} R={tp/(tp+fn):.2f}  ride->crash={fp} crash->ride={fn} | coef={np.round(m.coef_[0],2).tolist()} b={m.intercept_[0]:.2f}")

# ---------------- HH：简单阈值规则网格
rows = t_feat.rows_hh
y = np.array([r[1] for r in rows]); X = np.array([r[2:] for r in rows], dtype=float)
r1, rs, nxt = X[:, 0], X[:, 1], X[:, 4]
best = []
for a in (0.4, 0.45, 0.5, 0.55, 0.6):
    for b in (0.0, 0.2, 0.25, 0.3, 0.35):
        pred = (rs > a) & ((r1 > b) | (nxt <= 0.14))
        tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
        best.append((2 * tp / (2 * tp + fp + fn), a, b, tp, fp, fn))
for r in sorted(best, reverse=True)[:6]:
    print("HH rule rs>%.2f & r1>%.2f: F=%.3f tp=%d fp=%d fn=%d" % (r[1], r[2], r[0], r[3], r[4], r[5]))
