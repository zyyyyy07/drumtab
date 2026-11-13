"""细分类评测：开/闭镲、吊镲/叮叮镲、通鼓高低、军鼓鬼音。"""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from t_eval import *
from app import pipeline

SUB = {"CHH": "hh_closed", "PHH": "hh_closed", "OHH": "hh_open",
       "RDC": "ride", "RDB": "ride", "CRC": "crash", "CHC": "crash", "SPC": "crash",
       "HIT": "tom_hi", "MHT": "tom_mid", "HFT": "tom_floor", "LFT": "tom_floor",
       "SD": "snare", "SDG": "ghost", "SDB": "snare", "SST": "snare", "SDF": "snare", "SDNS": "snare", "SDD": "snare"}
GROUP = {"hh_closed": "HH", "hh_open": "HH", "ride": "CY", "crash": "CY", "tom_hi": "TT", "tom_mid": "TT", "tom_floor": "TT", "snare": "SD", "ghost": "SD"}

conf = {}
per_song = {}
feats_cache = {}
for n in names:
    z = np.load(cache / f"{n}.npz")
    acts = pipeline.fuse_acts(z["stem"], z["mix"])
    raw = drums.detect(z["stem"].astype(np.float32), z["mix"].astype(np.float32))
    raw.pop("_low", None)
    if n not in feats_cache:
        a, _ = audio_io.read_audio_file(cache / f"{n}_drums.flac")
        feats_cache[n] = drums.StemFeatures(a.mean(0))
    bp = float(np.median(np.diff(z["beats"]))) if len(z["beats"]) > 2 else 0.5
    hits = drums.refine(raw, feats_cache[n], bp)
    ann = load_ann(n)
    for g in ("HH", "CY", "TT", "SD"):
        ref = sorted((t, SUB[k]) for t, k in ann if k in SUB and GROUP[SUB[k]] == g)
        est = sorted((h.t, ("ghost" if (h.kind == "snare" and h.ghost) else h.kind)) for h in hits if GROUP.get(h.kind) == g)
        tp, pairs = match([t for t, _ in ref], [t for t, _ in est])
        for i, j in pairs:
            key = (ref[i][1], est[j][1])
            conf[key] = conf.get(key, 0) + 1
            per_song.setdefault(n[11:], {}).setdefault(key, 0)
            per_song[n[11:]][key] += 1

for g, kinds in (("HH", ["hh_closed", "hh_open"]), ("CY", ["ride", "crash"]), ("TT", ["tom_hi", "tom_mid", "tom_floor"]), ("SD", ["snare", "ghost"])):
    print(f"--- {g}  (行=真实, 列=预测)   " + "  ".join(f"{k:>10s}" for k in kinds))
    for r in kinds:
        print(f"{r:>12s}             " + "  ".join(f"{conf.get((r, c), 0):10d}" for c in kinds))
if "-v" in sys.argv:
    for s, d in per_song.items():
        print(s, {f"{a}>{b}": v for (a, b), v in d.items() if a != b or a in ("hh_open", "crash", "ghost") or a.startswith("tom")})
