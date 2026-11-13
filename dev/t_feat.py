"""为开镲、吊镲判别挑特征：算一堆候选特征，看各自的可分性（AUC）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from t_eval import *
from app import pipeline

FPS = 100


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rows_hh, rows_cy = [], []
for n in names:
    z = np.load(cache / f"{n}.npz")
    acts = pipeline.fuse_acts(z["stem"], z["mix"])
    raw = drums.detect(acts)
    a, _ = audio_io.read_audio_file(cache / f"{n}_drums.flac")
    F = drums.StemFeatures(a.mean(0))
    mag = F.mag; fr = F.freqs
    band = lambda lo, hi: np.sqrt((mag[(fr >= lo) & (fr < hi)] ** 2).sum(0) + 1e-12)
    e_hf, e_mid, e_lo, e_hi2 = band(7000, 16000), band(2000, 7000), band(300, 2000), band(10000, 16000)
    ann = load_ann(n)
    bp = float(np.median(np.diff(z["beats"])))

    def seg(env, t, a_ms, b_ms):
        f = int(round(t * FPS)); a_, b_ = max(0, f + a_ms // 10), min(len(env), f + b_ms // 10)
        return float(env[a_:b_].mean()) if b_ > a_ else 0.0

    def pk(env, t):
        f = int(round(t * FPS)); return float(env[max(0, f - 1): f + 5].max())

    # ---- HH
    hh = [t for t, _ in raw["HH"]]
    ref = sorted((t, k) for t, k in ann if k in ("CHH", "OHH", "PHH"))
    _, pairs = match([t for t, _ in ref], hh)
    lab = {j: ref[i][1] for i, j in pairs}
    others = np.array(sorted([t for t, _ in raw["SD"]] + [t for t, _ in raw["CY"]]))
    feats = []
    for j, t in enumerate(hh):
        nxt = hh[j + 1] - t if j + 1 < len(hh) else 9.0
        p = pk(e_hf, t) + 1e-9
        end = int(min(260, max(60, nxt * 1000 - 20)))
        r1 = seg(e_hf, t, 90, max(100, end)) / p
        r_short = seg(e_hf, t, 60, 110) / p
        r_long = seg(e_hf, t, 150, 250) / p if nxt > 0.27 else np.nan
        pre = seg(e_hf, t, -40, -10) / p
        busy = bool(others.size and np.min(np.abs(others - t)) < 0.05)
        feats.append([r1, r_short, r_long, pre, nxt, p, busy])
    feats = np.array(feats, dtype=float) if feats else np.zeros((0, 7))
    if len(feats):
        p25 = np.percentile(feats[:, 0], 25); med = np.median(feats[:, 0]); ps25 = np.percentile(feats[:, 1], 25)
        for j in lab:
            f = feats[j]
            rows_hh.append((n[11:], lab[j] == "OHH", f[0], f[1], f[2], f[3], f[4], f[0] - p25, f[0] / (p25 + 1e-3), f[1] - ps25, f[0] - f[3], f[5] / np.percentile(feats[:, 5], 90), f[6]))

    # ---- CY
    cy = [t for t, _ in raw["CY"]]
    ref = sorted((t, k) for t, k in ann if k in ("RDC", "RDB", "CRC", "CHC", "SPC"))
    _, pairs = match([t for t, _ in ref], cy)
    lab = {j: ref[i][1] for i, j in pairs}
    cyt = np.array(cy)
    sn = [pk(e_mid, t) for t, _ in raw["SD"]]
    sref = np.percentile(sn, 75) if sn else 1.0
    feats = []
    for j, t in enumerate(cy):
        near = int(np.sum(np.abs(cyt - t) <= max(1.2, 2.2 * bp))) - 1
        lvl = pk(e_mid, t) + pk(e_hf, t)
        sus = seg(e_mid, t, 200, 400) / (pk(e_mid, t) + 1e-9)
        sus_hf = seg(e_hf, t, 200, 400) / (pk(e_hf, t) + 1e-9)
        tilt = seg(e_lo, t, 20, 120) / (seg(e_hf, t, 20, 120) + 1e-9)
        midhf = seg(e_mid, t, 20, 120) / (seg(e_hf, t, 20, 120) + 1e-9)
        rise = pk(e_mid, t) / (seg(e_mid, t, -60, -20) + 1e-9)
        feats.append([near, lvl, sus, sus_hf, tilt, midhf, rise, pk(e_mid, t) / (sref + 1e-9)])
    feats = np.array(feats, dtype=float) if feats else np.zeros((0, 8))
    if len(feats):
        medlvl = np.median(feats[:, 1])
        for j in lab:
            f = feats[j]
            loc = feats[np.abs(cyt - cyt[j]) <= 3.0, 1]
            rows_cy.append((n[11:], lab[j] in ("CRC", "CHC", "SPC"), f[0], f[1] / (medlvl + 1e-9), f[1] / (np.median(loc) + 1e-9), f[2], f[3], f[4], f[5], np.log(f[6]), f[7]))

for title, rows, cols in (("HH open", rows_hh, ["r1", "r_short", "r_long", "pre", "nxt", "r1-p25", "r1/p25", "rs-p25", "r1-pre", "lvl", "busy"]),
                          ("CY crash", rows_cy, ["near", "lvl/med", "lvl/locmed", "sus_mid", "sus_hf", "tilt", "mid/hf", "logrise", "mid/snare"])):
    y = np.array([r[1] for r in rows]); X = np.array([r[2:] for r in rows], dtype=float)
    print(f"== {title}: pos={y.sum()} neg={(~y).sum()}")
    for i, c in enumerate(cols):
        v = X[:, i]; ok = ~np.isnan(v)
        print(f"   {c:11s} AUC={auc(v[ok & y], v[ok & ~y]):.3f}   pos med={np.nanmedian(v[y]):.3f} [{np.nanpercentile(v[y],20):.3f},{np.nanpercentile(v[y],80):.3f}]  neg med={np.nanmedian(v[~y]):.3f} [{np.nanpercentile(v[~y],20):.3f},{np.nanpercentile(v[~y],80):.3f}]")
    np.save(cache / f"feat_{title.split()[0]}.npy", np.column_stack([y, X]))
    songs = sorted({r[0] for r in rows})
    for s in songs:
        m = np.array([r[0] == s for r in rows])
        if title == "HH open":
            print(f"   {s:13s} n={m.sum():4d} open={int(y[m].sum()):3d}  r1 closed med={np.median(X[m & ~y, 0]) if (m & ~y).any() else -1:.2f} p90={np.percentile(X[m & ~y, 0], 90) if (m & ~y).any() else -1:.2f}  open med={np.median(X[m & y, 0]) if (m & y).any() else -1:.2f} p10={np.percentile(X[m & y, 0], 10) if (m & y).any() else -1:.2f} | rs closed={np.median(X[m & ~y, 1]) if (m & ~y).any() else -1:.2f} open={np.median(X[m & y, 1]) if (m & y).any() else -1:.2f}")
        else:
            print(f"   {s:13s} n={m.sum():4d} crash={int(y[m].sum()):3d}  near ride={np.median(X[m & ~y, 0]) if (m & ~y).any() else -1:.0f} crash={np.median(X[m & y, 0]) if (m & y).any() else -1:.0f} | lvl/loc ride={np.median(X[m & ~y, 2]) if (m & ~y).any() else -1:.2f} crash={np.median(X[m & y, 2]) if (m & y).any() else -1:.2f} | mid/hf ride={np.median(X[m & ~y, 6]) if (m & ~y).any() else -1:.2f} crash={np.median(X[m & y, 6]) if (m & y).any() else -1:.2f} | m/sn ride={np.median(X[m & ~y, 8]) if (m & ~y).any() else -1:.2f} crash={np.median(X[m & y, 8]) if (m & y).any() else -1:.2f} | sus ride={np.median(X[m & ~y, 3]) if (m & ~y).any() else -1:.2f} crash={np.median(X[m & y, 3]) if (m & y).any() else -1:.2f}")
