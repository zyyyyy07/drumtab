"""回归评测：用正式的 drums.detect（含去重、底鼓同时性规则）在 MDB Drums 上算 5 类 F 值。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from t_eval import *

tot = {c: [0, 0, 0] for c in drums.CLASSES}
for n in names:
    z = np.load(cache / f"{n}.npz")
    raw = drums.detect(z["stem"].astype(np.float32), z["mix"].astype(np.float32))
    if "--fill" in sys.argv:
        from app import beats as _b
        g = _b.build_grid(z["beats"], z["downbeats"], float(z["dur"]), onsets=np.array(sorted([t for t, _ in raw["KD"]] + [t for t, _ in raw["SD"]])))
        raw = drums.fill_by_pattern(raw, g)
    raw.pop("_low", None)
    ann = load_ann(n)
    for c in drums.CLASSES:
        ref = [t for t, k in ann if C5.get(k) == c]
        est = [t for t, _ in raw[c]]
        tp, _ = match(ref, est)
        tot[c][0] += tp; tot[c][1] += len(ref); tot[c][2] += len(est)
res = {c: prf(*v) for c, v in tot.items()}
res["ALL"] = prf(*[sum(v[i] for v in tot.values()) for i in range(3)])
show("drums.detect", res)
print({c: tot[c] for c in tot})
