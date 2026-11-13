import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from t_eval import *

fuse = lambda z: 0.7 * z["stem"][: min(len(z["stem"]), len(z["mix"]))] + 0.3 * z["mix"][: min(len(z["stem"]), len(z["mix"]))]
for c in drums.CLASSES:
    row = []
    for th in (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.40, 0.50):
        thr = dict(drums.BASE_THRESHOLDS); thr[c] = th
        r = eval5(fuse, thr=thr)[c]
        row.append(f"{th:.2f}:{r[2]*100:4.1f}(P{r[0]*100:3.0f}/R{r[1]*100:3.0f})")
    print(c, " ".join(row))

# 军鼓按子类看召回
i = drums.CLASSES.index("SD")
cnt = {}
for n in names:
    z = np.load(cache / f"{n}.npz")
    est = [t for t, _ in drums.pick_peaks(fuse(z)[:, i], drums.BASE_THRESHOLDS["SD"])]
    ann = load_ann(n)
    ref = [(t, k) for t, k in ann if C5.get(k) == "SD"]
    tp, pairs = match([t for t, _ in ref], est)
    ref_sorted = sorted(ref)
    hit = {a for a, _ in pairs}
    for j, (t, k) in enumerate(ref_sorted):
        c = cnt.setdefault(k, [0, 0]); c[1] += 1; c[0] += j in hit
print("SD recall by subclass:", {k: f"{v[0]}/{v[1]}" for k, v in cnt.items()})
# 每首歌的总体 F
for n in names:
    z = np.load(cache / f"{n}.npz"); ann = load_ann(n); acts = fuse(z); s = [0, 0, 0]; per = []
    for i, c in enumerate(drums.CLASSES):
        ref = [t for t, k in ann if C5.get(k) == c]
        est = [t for t, _ in drums.pick_peaks(acts[:, i], drums.BASE_THRESHOLDS[c])]
        tp, _ = match(ref, est); s[0] += tp; s[1] += len(ref); s[2] += len(est)
        per.append(f"{c}{tp}/{len(ref)}/{len(est)}")
    print(f"{n[11:]:14s} F={prf(*s)[2]*100:5.1f}  " + " ".join(per))
