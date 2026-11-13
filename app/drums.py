"""鼓识别。

第一步：ADTOF Frame-RNN（在大规模真实录音上训练的鼓转录模型）给出 5 类击打概率：
        底鼓 / 军鼓 / 通鼓 / 踩镲 / 镲片。
第二步：在分离出的鼓轨上做信号分析，把 5 类细分成鼓手真正要看的谱面：
        踩镲 开/闭、镲片 → 吊镲/叮叮镲、通鼓 → 高/中/落地、军鼓 鬼音。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

FPS = 100
SR = 44100
CLASSES = ["KD", "SD", "TT", "HH", "CY"]
# 各类检出阈值：以 ADTOF 论文数值为起点，在 MDB Drums 标注集上微调
BASE_THRESHOLDS = {"KD": 0.20, "SD": 0.18, "TT": 0.38, "HH": 0.26, "CY": 0.30}

_model = None
_proc = None


def _load(device: str):
    global _model, _proc
    if _model is None:
        import contextlib
        import io

        from adtof_pytorch import (
            calculate_n_bins,
            create_frame_rnn_model,
            get_default_weights_path,
            load_pytorch_weights,
        )
        from adtof_pytorch.audio import create_adtof_processor

        _proc = create_adtof_processor()
        m = create_frame_rnn_model(calculate_n_bins())
        with contextlib.redirect_stdout(io.StringIO()):
            m = load_pytorch_weights(m, get_default_weights_path(), strict=False)
        _model = m.eval()
    return _model.to(device), _proc


def activations(mono: np.ndarray, device: str = "cuda") -> np.ndarray:
    """mono: 44.1k 单声道。返回 [T, 5]，100 帧/秒。"""
    model, proc = _load(device)
    spec = proc.apply_filterbank(proc.compute_stft(mono.astype(np.float32))).T[None, :, :, None]
    x = torch.from_numpy(np.ascontiguousarray(spec)).float().to(device)
    with torch.no_grad():
        y = model(x)[0].float().cpu().numpy()
    model.to("cpu")
    return y


def pick_peaks(act: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """对一条概率曲线做峰值检出，返回 [(时间秒, 峰值)]。算法同 madmom NotePeakPicking。"""
    a = act.astype(np.float32)
    n = len(a)
    pre_avg, post_avg, pre_max, post_max, combine = 10, 1, 2, 1, 2
    k = np.ones(pre_avg + 1 + post_avg, dtype=np.float32)
    avg = np.convolve(np.pad(a, (pre_avg, post_avg), mode="edge"), k / k.size, mode="valid")
    proc = np.maximum(0.0, a - avg)
    pad = np.pad(proc, (pre_max, post_max), mode="edge")
    wmax = np.max(np.stack([pad[i : i + n] for i in range(pre_max + post_max + 1)]), axis=0)
    idx = np.where((proc >= wmax) & (proc >= threshold))[0]
    out: list[tuple[float, float]] = []
    group: list[int] = []
    for i in idx:
        if group and i - group[-1] > combine:
            b = max(group, key=lambda j: proc[j])
            out.append((b / FPS, float(a[b])))
            group = []
        group.append(int(i))
    if group:
        b = max(group, key=lambda j: proc[j])
        out.append((b / FPS, float(a[b])))
    return out


@dataclass
class Hit:
    t: float          # 秒
    kind: str         # kick snare hh_closed hh_open crash ride tom_hi tom_mid tom_floor
    vel: float = 0.8  # 0..1 相对力度
    ghost: bool = False
    conf: float = 1.0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 鼓轨特征

def _stft_mag(mono: np.ndarray, n_fft: int = 2048, hop: int = SR // FPS) -> np.ndarray:
    import librosa

    return np.abs(librosa.stft(mono.astype(np.float32), n_fft=n_fft, hop_length=hop, center=True))


def _band(mag: np.ndarray, lo: float, hi: float, n_fft: int = 2048) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    sel = (freqs >= lo) & (freqs < hi)
    return np.sqrt(np.sum(mag[sel] ** 2, axis=0) + 1e-12)


class StemFeatures:
    def __init__(self, drum_mono: np.ndarray):
        self.mag = _stft_mag(drum_mono)
        self.n = self.mag.shape[1]
        self.low = _band(self.mag, 30, 150)
        self.mid = _band(self.mag, 150, 500)
        self.snr = _band(self.mag, 1500, 6000)
        self.hf = _band(self.mag, 7000, 16000)
        self.cym = _band(self.mag, 2000, 7000)
        self.full = _band(self.mag, 30, 16000)
        self.freqs = np.fft.rfftfreq(2048, 1 / SR)

    def peak(self, env: np.ndarray, t: float, pre: int = 1, post: int = 4) -> float:
        f = int(round(t * FPS))
        a, b = max(0, f - pre), min(self.n, f + post + 1)
        return float(env[a:b].max()) if b > a else 0.0

    def mean(self, env: np.ndarray, t: float, a_ms: float, b_ms: float) -> float:
        f = int(round(t * FPS))
        a, b = max(0, f + int(a_ms / 10)), min(self.n, f + int(b_ms / 10))
        return float(env[a:b].mean()) if b > a else 0.0

    def tom_pitch(self, t: float) -> float:
        """通鼓基频估计：击打后 30~160ms 的平均谱在 55~420Hz 内的最强峰。"""
        f = int(round(t * FPS))
        a, b = min(self.n - 1, f + 3), min(self.n, f + 16)
        if b <= a:
            return 0.0
        sp = self.mag[:, a:b].mean(axis=1)
        sel = np.where((self.freqs >= 55) & (self.freqs <= 420))[0]
        k = sel[int(np.argmax(sp[sel]))]
        # 抛物线插值
        if 0 < k < len(sp) - 1:
            y0, y1, y2 = sp[k - 1], sp[k], sp[k + 1]
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        else:
            d = 0.0
        return float((k + d) * SR / 2048)


# ---------------------------------------------------------------- 细分

def fuse(stem: np.ndarray, mix: np.ndarray) -> np.ndarray:
    """分离后的鼓轨为主、原曲为辅（MDB Drums 上实测比只用原曲高约 5 个点）。"""
    n = min(len(stem), len(mix))
    return 0.7 * stem[:n] + 0.3 * mix[:n]


def _dedupe(peaks: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """同一件鼓相隔不到 gap 秒的两个峰当作一次击打，留强的。"""
    out: list[tuple[float, float]] = []
    for t, v in peaks:
        if out and t - out[-1][0] < gap:
            if v > out[-1][1]:
                out[-1] = (t, v)
        else:
            out.append((t, v))
    return out


def detect(stem: np.ndarray, mix: np.ndarray, sensitivity: float = 0.0) -> dict[str, list[tuple[float, float]]]:
    """sensitivity ∈ [-1, 1]：正值识别出更多音（阈值降低），负值更保守。"""
    scale = float(np.clip(1.0 - 0.5 * sensitivity, 0.4, 1.8))
    n = min(len(stem), len(mix))
    stem, mix = stem[:n], mix[:n]
    acts = fuse(stem, mix)
    raw = {c: pick_peaks(acts[:, i], BASE_THRESHOLDS[c] * scale) for i, c in enumerate(CLASSES)}
    raw["KD"] = _dedupe(raw["KD"], 0.07)
    raw["SD"] = _dedupe(raw["SD"], 0.06)
    raw["TT"] = _dedupe(raw["TT"], 0.07)
    raw["CY"] = _dedupe(raw["CY"], 0.08)
    raw["HH"] = _dedupe(raw["HH"], 0.045)
    # 差一点没过线的候选，留给 fill_by_pattern 按“前后小节同一位置是否都有”来决定
    low = {c: pick_peaks(acts[:, CLASSES.index(c)], 0.45 * BASE_THRESHOLDS[c] * scale) for c in ("KD", "SD", "HH")}

    def local(arr: np.ndarray, t: float, i: int) -> float:
        k = int(round(t * FPS))
        return float(arr[max(0, k - 2) : k + 3, i].max()) if n else 0.0

    def strong(c: str) -> float:
        v = [x for _, x in raw[c] if x >= 0.3]
        return float(np.percentile(v, 75)) if len(v) >= 8 else 0.6

    # 底鼓必须在原曲里也看得到一点影子：只在分离出的鼓轨里“有”的弱底鼓，
    # 多半是别的声音（边击、贝斯）被分轨抹成了像底鼓的样子
    i_kd = CLASSES.index("KD")
    raw["KD"] = [
        (t, v) for t, v in raw["KD"]
        if local(mix, t, i_kd) >= 0.08 * scale or local(stem, t, i_kd) >= 0.7
    ]
    kick_t = np.array([t for t, _ in raw["KD"]])

    def with_kick(t: float) -> bool:
        return bool(kick_t.size and np.min(np.abs(kick_t - t)) < 0.035)

    # 模型常把一记结实的底鼓同时认成很弱的军鼓/通鼓。
    # 与底鼓同时出现的军鼓，必须在鼓轨或原曲里有一边足够确定才算数。
    i_sd, i_tt = CLASSES.index("SD"), CLASSES.index("TT")
    need = 0.7 * strong("SD") * scale
    raw["SD"] = [
        (t, v) for t, v in raw["SD"]
        if not with_kick(t) or max(local(stem, t, i_sd), local(mix, t, i_sd)) >= need
    ]
    # 通鼓是模型最不可靠的一类：要求鼓轨和原曲两边都认，且不能只是底鼓/军鼓的影子
    snare_t = np.array([t for t, _ in raw["SD"]])
    need_tt = 0.75 * strong("TT") * scale
    kept = []
    for t, v in raw["TT"]:
        s, m = local(stem, t, i_tt), local(mix, t, i_tt)
        if min(s, m) < 0.12 * scale:
            continue
        if with_kick(t) and max(s, m) < max(need_tt, 0.55 * scale):
            continue
        if snare_t.size and np.min(np.abs(snare_t - t)) < 0.035:
            sv = max(local(stem, t, i_sd), local(mix, t, i_sd))
            if v < sv:
                continue
        kept.append((t, v))
    raw["TT"] = kept
    low["KD"] = [(t, v) for t, v in low["KD"] if local(mix, t, i_kd) >= 0.08 * scale]
    low["SD"] = [(t, v) for t, v in low["SD"] if not with_kick(t)]
    raw["_low"] = low  # type: ignore[assignment]
    return raw


def fill_by_pattern(raw: dict, grid) -> dict:
    """鼓是重复的：某个位置在前后各两个小节里全都有这一下，
    而本小节只是概率差一点没过线，就把它补上（最常见的是漏掉的反拍踩镲）。
    条件放宽成“四个里有三个”在 MDB Drums 上补对的不到一半，所以要求四个全有。"""
    low = raw.pop("_low", None)
    if not low:
        return raw
    starts = np.array([s for s, _ in grid.bars])
    lens = np.array([n for _, n in grid.bars])

    def locate(t: float) -> tuple[int, int]:
        p = float(grid.pos(t))
        b = int(np.clip(np.searchsorted(starts, p + 0.125, side="right") - 1, 0, len(starts) - 1))
        return b, int(round((p - starts[b]) * 4))  # 小节序号, 小节内第几个十六分位

    for c, cands in low.items():
        have: dict[tuple[int, int], bool] = {}
        times = np.array([t for t, _ in raw[c]])
        for t in times:
            have[locate(t)] = True
        added = []
        for t, v in cands:
            if times.size and np.min(np.abs(times - t)) < 0.05:
                continue
            b, k = locate(t)
            if (b, k) in have:
                continue
            nb = [j for j in (b - 2, b - 1, b + 1, b + 2) if 0 <= j < len(starts) and lens[j] == lens[b]]
            if len(nb) == 4 and all((j, k) in have for j in nb):
                added.append((t, v))
        if added:
            raw[c] = sorted(raw[c] + added)
    return raw


def refine(raw: dict[str, list[tuple[float, float]]], feats: StemFeatures, beat_period: float = 0.5) -> list[Hit]:
    hits: list[Hit] = []

    # ---- 底鼓
    kv = [feats.peak(feats.low, t) for t, _ in raw["KD"]]
    kref = np.percentile(kv, 90) if kv else 1.0
    for (t, c), v in zip(raw["KD"], kv):
        hits.append(Hit(t, "kick", float(np.clip(v / (kref + 1e-9), 0.1, 1.0)), conf=c))

    # ---- 军鼓（含鬼音）
    sv = [feats.peak(feats.mid, t) + feats.peak(feats.snr, t) for t, _ in raw["SD"]]
    if sv:
        sref = np.percentile(sv, 90)
        st_ = np.array([t for t, _ in raw["SD"]])
        sv_ = np.array(sv)
        for (t, c), v in zip(raw["SD"], sv):
            rel = v / (sref + 1e-9)
            # 鬼音是“比周围的军鼓轻很多”，不是“比全曲最响的轻”：
            # 整段都轻（边击、主歌轻打）时它们就是主音，不该标成鬼音
            near = sv_[np.abs(st_ - t) <= 4.0]
            loc = float(np.percentile(near, 75)) if len(near) >= 4 else sref
            ghost = bool(rel < 0.22 and v < 0.4 * loc)
            hits.append(Hit(t, "snare", float(np.clip(rel, 0.1, 1.0)), ghost=ghost, conf=c))

    # ---- 踩镲 开/闭
    hh = raw["HH"]
    hh_t = np.array([t for t, _ in hh])
    hv = [feats.peak(feats.hf, t) for t in hh_t]
    href = np.percentile(hv, 90) if hv else 1.0
    for i, (t, c) in enumerate(hh):
        # 开镲的高频在击打后拖着不散；闭镲 50ms 内就收住。
        # 两个比值：击后 60~110ms（密集节奏里也量得到）和 90ms~下一击之前。
        nxt = hh_t[i + 1] - t if i + 1 < len(hh_t) else 9.0
        pk = hv[i] + 1e-9
        end = int(min(260, max(60, nxt * 1000 - 20)))
        r_long = feats.mean(feats.hf, t, 90, max(100, end)) / pk
        r_short = feats.mean(feats.hf, t, 60, 110) / pk
        is_open = r_short > 0.5 and (r_long > 0.35 or nxt <= 0.14)
        hits.append(Hit(t, "hh_open" if is_open else "hh_closed", float(np.clip(hv[i] / (href + 1e-9), 0.1, 1.0)), conf=c))

    # ---- 镲片：吊镲 / 叮叮镲
    # 四个线索合成一个分数（系数在 MDB Drums 上拟合，留一曲交叉验证 吊镲 P≈0.80 R≈0.68）：
    #   密度   叮叮镲是“打拍子”的，前后密集连续；吊镲是稀疏的重音
    #   余音   吊镲高频拖得长
    #   音色   吊镲中频（2~7k）成分更重
    #   响度   吊镲和军鼓一样响，叮叮镲明显轻
    cy = raw["CY"]
    cy_t = np.array([t for t, _ in cy])
    cmid = [feats.peak(feats.cym, t) for t in cy_t]
    cv = [feats.peak(feats.hf, t) + m for t, m in zip(cy_t, cmid)]
    cref = np.percentile(cv, 90) if cv else 1.0
    smid = [feats.peak(feats.cym, t) for t, _ in raw["SD"]]
    sref = np.percentile(smid, 75) if smid else (np.percentile(cmid, 75) if cmid else 1.0)
    win = max(1.2, 2.2 * beat_period)
    for i, (t, c) in enumerate(cy):
        near = min(int(np.sum(np.abs(cy_t - t) <= win)) - 1, 8) / 8.0
        sus = feats.mean(feats.hf, t, 200, 400) / (feats.peak(feats.hf, t) + 1e-9)
        midhf = feats.mean(feats.cym, t, 20, 120) / (feats.mean(feats.hf, t, 20, 120) + 1e-9)
        msn = cmid[i] / (sref + 1e-9)
        z = (
            2.78
            - 3.77 * near
            + 1.97 * np.log(np.clip(sus, 0.05, 8))
            + 0.77 * np.log(np.clip(midhf, 0.2, 10))
            + 2.45 * np.log(np.clip(msn, 0.1, 5))
        )
        hits.append(Hit(t, "crash" if z > 0 else "ride", float(np.clip(cv[i] / (cref + 1e-9), 0.1, 1.0)), conf=c))

    # ---- 通鼓：按音高分 高/中/落地
    tt = raw["TT"]
    if tt:
        f0 = np.array([feats.tom_pitch(t) for t, _ in tt])
        kinds = classify_toms(f0)
        tv = [feats.peak(feats.low, t) + feats.peak(feats.mid, t) for t, _ in tt]
        tref = np.percentile(tv, 90)
        for (t, c), k, v in zip(tt, kinds, tv):
            hits.append(Hit(t, k, float(np.clip(v / (tref + 1e-9), 0.1, 1.0)), conf=c))

    hits.sort(key=lambda h: h.t)
    return hits


def classify_toms(f0: np.ndarray) -> list[str]:
    """把通鼓基频聚成最多 3 组。组间至少差约 3 个半音才认为是不同的鼓。"""
    lf = np.log2(np.clip(f0, 40, 500))
    order = np.argsort(lf)
    groups: list[list[int]] = [[int(order[0])]]
    for a, b in zip(order[:-1], order[1:]):
        if lf[b] - lf[a] > 0.2:  # 0.2 个八度 ≈ 2.4 半音
            groups.append([])
        groups[-1].append(int(b))
    while len(groups) > 3:  # 合并最接近的两组
        cents = [np.mean(lf[g]) for g in groups]
        j = int(np.argmin(np.diff(cents)))
        groups[j : j + 2] = [groups[j] + groups[j + 1]]
    cents = [2 ** float(np.mean(lf[g])) for g in groups]
    if len(groups) == 3:
        names = ["tom_floor", "tom_mid", "tom_hi"]
    elif len(groups) == 2:
        names = ["tom_floor" if cents[0] < 115 else "tom_mid", "tom_hi" if cents[1] > 125 else "tom_mid"]
        if names[0] == names[1]:
            names = ["tom_floor", "tom_hi"]
    else:
        c = cents[0]
        names = ["tom_floor" if c < 100 else ("tom_mid" if c < 150 else "tom_hi")]
    out = [""] * len(f0)
    for g, nm in zip(groups, names):
        for i in g:
            out[i] = nm
    return out
