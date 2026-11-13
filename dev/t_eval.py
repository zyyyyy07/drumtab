"""MDB Drums 上的鼓识别评测（±50ms 命中窗）。"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from app import drums, audio_io

here = Path(__file__).parent
root = here / "testdata" / "MDBDrums" / "MDB Drums"
cache = here / "cache"
C5 = {"KD": "KD", "SD": "SD", "SDB": "SD", "SDD": "SD", "SDF": "SD", "SDG": "SD", "SDNS": "SD", "SST": "SD",
      "CHH": "HH", "OHH": "HH", "PHH": "HH", "HIT": "TT", "MHT": "TT", "HFT": "TT", "LFT": "TT",
      "RDC": "CY", "RDB": "CY", "CRC": "CY", "CHC": "CY", "SPC": "CY"}


def load_ann(name):
    out = []
    for line in open(root / "annotations" / "subclass" / f"{name}_subclass.txt"):
        p = line.split()
        if len(p) == 2:
            out.append((float(p[0]), p[1]))
    return out


def match(ref, est, tol=0.05):
    """贪心一对一匹配，返回 (tp, 匹配对列表[(ref_i, est_i)])。"""
    ref, est = sorted(ref), sorted(est)
    used = set()
    pairs = []
    j0 = 0
    for i, r in enumerate(ref):
        best, bd = None, tol
        for j in range(j0, len(est)):
            if est[j] < r - tol:
                j0 = j + 1
                continue
            if est[j] > r + tol:
                break
            if j not in used and abs(est[j] - r) <= bd:
                best, bd = j, abs(est[j] - r)
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return len(pairs), pairs


def prf(tp, nr, ne):
    p = tp / ne if ne else 0.0
    r = tp / nr if nr else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


names = sorted(Path(f).stem for f in glob.glob(str(cache / "*.npz")))


def eval5(fuse, sens=0.0, thr=None):
    tot = {c: [0, 0, 0] for c in drums.CLASSES}
    for n in names:
        z = np.load(cache / f"{n}.npz")
        acts = fuse(z)
        ann = load_ann(n)
        for i, c in enumerate(drums.CLASSES):
            ref = [t for t, k in ann if C5.get(k) == c]
            th = (thr or drums.BASE_THRESHOLDS)[c] * (1 - 0.5 * sens)
            est = [t for t, _ in drums.pick_peaks(acts[:, i], th)]
            tp, _ = match(ref, est)
            tot[c][0] += tp; tot[c][1] += len(ref); tot[c][2] += len(est)
    res = {c: prf(*v) for c, v in tot.items()}
    s = [sum(v[i] for v in tot.values()) for i in range(3)]
    res["ALL"] = prf(*s)
    return res


def show(tag, res):
    print(f"{tag:22s}" + "  ".join(f"{c}:{res[c][2]*100:5.1f}(P{res[c][0]*100:3.0f} R{res[c][1]*100:3.0f})" for c in drums.CLASSES + ["ALL"]))


if __name__ == "__main__":
    n_ = lambda z: min(len(z["stem"]), len(z["mix"]))
    show("mix only", eval5(lambda z: z["mix"]))
    show("demucs stem", eval5(lambda z: z["stem"]))
    show("true drum-only (上限)", eval5(lambda z: z["ref"]))
    show("mean(stem,mix)", eval5(lambda z: 0.5 * (z["stem"][:n_(z)] + z["mix"][:n_(z)])))
    show("max(stem,mix)", eval5(lambda z: np.maximum(z["stem"][:n_(z)], z["mix"][:n_(z)])))
    show(".7stem+.3mix", eval5(lambda z: 0.7 * z["stem"][:n_(z)] + 0.3 * z["mix"][:n_(z)]))
