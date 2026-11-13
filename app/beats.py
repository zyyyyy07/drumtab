"""节拍与小节线：Beat This!（ISMIR 2024）识别拍点和强拍，随后整理成小节网格。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_tracker = None


def track(mono: np.ndarray, sr: int, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """返回 (拍点秒, 强拍秒)。"""
    global _tracker
    try:
        if _tracker is None:
            from beat_this.inference import Audio2Beats

            _tracker = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
        beats, downbeats = _tracker(mono.astype(np.float32), sr)
        return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)
    except Exception as e:
        print("[beats] Beat This 不可用，退回 librosa：", e)
        import librosa

        _, b = librosa.beat.beat_track(y=mono, sr=sr, units="time")
        b = np.asarray(b, dtype=float)
        return b, b[::4]


@dataclass
class Grid:
    beat_times: np.ndarray        # 长度 = 总拍数 + 1（末尾多一个收尾拍）
    bars: list[tuple[int, int]]   # (起始拍序号, 拍数)
    numerator: int                # 主拍号分子
    tempo: float                  # BPM
    anacrusis: bool               # 第一小节是否为弱起

    def pos(self, t):
        """秒 -> 连续拍位置。"""
        return np.interp(t, self.beat_times, np.arange(len(self.beat_times)))

    def time(self, p):
        return np.interp(p, np.arange(len(self.beat_times)), self.beat_times)

    def bar_time(self, i: int) -> float:
        return float(self.beat_times[self.bars[i][0]])


def _fix_phase(beats: np.ndarray, kicks: np.ndarray, glitches: np.ndarray) -> np.ndarray:
    """拍点跟踪在歌曲中途错拍（比如加了半拍的过门）后，常要好几秒才跟回来，
    这几秒里整段会差半拍，谱面就乱成一团。底鼓几乎总在拍上：
    某拍前后的底鼓若都落在“半拍”位置、没有一个在拍上，就把这拍挪到底鼓上。"""
    if len(kicks) < 16 or len(beats) < 8:
        return beats
    kicks = np.sort(kicks)
    idx = np.clip(np.searchsorted(beats, kicks), 1, len(beats) - 1)
    left, right = beats[idx - 1], beats[idx]
    r = np.where(kicks - left < right - kicks, kicks - left, kicks - right)  # 到最近拍点的有符号距离
    loc_p = np.interp(kicks, (beats[:-1] + beats[1:]) / 2, np.diff(beats))
    rel = r / loc_p
    out = beats.copy()
    for i in range(len(beats)):
        p = float(np.median(np.diff(beats[max(0, i - 4) : i + 5])))
        # 只在原始拍点本身就出过故障（间隔忽长忽短）的附近动手；
        # 节奏平稳的段落里底鼓落在反拍是正常编曲（如朋克、放克），不能挪
        if not glitches.size or np.min(np.abs(glitches - beats[i])) > 6 * p:
            continue
        m = np.abs(kicks - beats[i]) <= 1.6 * p
        if m.sum() < 3:
            continue
        rr = rel[m]
        half = np.abs(np.abs(rr) - 0.5) < 0.15
        on = np.abs(rr) < 0.15
        if on.sum() == 0 and half.mean() >= 0.75:
            out[i] = beats[i] + float(np.median(rr[half])) * p
    # 挪动后可能出现过近的两拍，交给 _regularize 合并
    return _regularize(np.sort(out))


def _snap_to_onsets(beats: np.ndarray, onsets: np.ndarray, tol: float = 0.045) -> np.ndarray:
    """拍点跟踪有 ±20ms 级的系统偏差，用鼓的实际落点做平滑校正，量化才准。"""
    if len(onsets) < 8 or len(beats) < 4:
        return beats
    idx = np.clip(np.searchsorted(onsets, beats), 1, len(onsets) - 1)
    near = np.where(np.abs(onsets[idx] - beats) < np.abs(onsets[idx - 1] - beats), onsets[idx], onsets[idx - 1])
    d = near - beats
    ok = np.abs(d) < tol
    if ok.sum() < 4:
        return beats
    d = np.where(ok, d, np.nan)
    out = beats.copy()
    for i in range(len(beats)):
        w = d[max(0, i - 4) : i + 5]
        w = w[~np.isnan(w)]
        if len(w) >= 2:
            out[i] += float(np.median(w))
    # 保证单调
    for i in range(1, len(out)):
        if out[i] <= out[i - 1] + 0.05:
            out[i] = out[i - 1] + 0.05
    return out


def _regularize(beats: np.ndarray) -> np.ndarray:
    """修掉拍点跟踪的两种常见故障：某一段掉成半速（间隔翻倍）、偶尔多出半拍。

    以全曲的主流拍长为准：间隔约为 k 倍的地方补上 k-1 拍，明显不足一拍的多余拍点删掉。
    """
    if len(beats) < 8:
        return beats
    iv = np.diff(beats)
    P = float(np.median(iv))
    ok = iv[(iv > 0.8 * P) & (iv < 1.25 * P)]
    if len(ok) >= 4:
        P = float(np.median(ok))
    out = [float(beats[0])]
    for i in range(1, len(beats)):
        t = float(beats[i])
        # 局部拍长：最近几个正常间隔的中位数，跟得上缓慢的速度变化
        recent = np.diff(out[-9:])
        recent = recent[(recent > 0.8 * P) & (recent < 1.25 * P)]
        p = float(np.median(recent)) if len(recent) >= 3 else P
        r = (t - out[-1]) / p
        if r < 0.6:
            continue
        k = int(round(r))
        if k >= 2 and abs(r - k) < 0.3:
            step = (t - out[-1]) / k
            base = out[-1]
            out.extend(base + step * j for j in range(1, k))
        out.append(t)
    return np.array(out)


def build_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
    onsets: np.ndarray | None = None,
    kicks: np.ndarray | None = None,
    beat_shift: int = 0,
    beats_per_bar: int | None = None,
    tempo_scale: float = 1.0,
) -> Grid:
    beats = np.unique(np.asarray(beats, dtype=float))
    if len(beats) < 2:
        beats = np.arange(0.0, max(duration, 2.0), 0.5)
        downbeats = beats[::4]
    downbeats = np.asarray(downbeats, dtype=float)
    iv0 = np.diff(beats)
    P0 = float(np.median(iv0))
    glitches = beats[1:][np.abs(iv0 - P0) > 0.25 * P0]
    beats = _regularize(beats)
    if kicks is not None and len(kicks):
        # 先把错位段落移回来，再找出它前后强拍的对应关系
        beats = _fix_phase(beats, np.asarray(kicks, dtype=float), glitches)
    raw_beats = beats
    # 模型输出的拍点只有 20ms 分辨率：用前后各两拍做局部平均，去掉这种抖动
    if len(beats) >= 7:
        sm = beats.copy()
        sm[2:-2] = (beats[:-4] + beats[1:-3] + beats[2:-2] + beats[3:-1] + beats[4:]) / 5.0
        beats = sm

    # 强拍对应的拍序号
    db_idx = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats})

    # 速度 ×2 / ÷2
    if tempo_scale >= 1.99:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
        db_idx = [2 * i for i in db_idx]
    elif tempo_scale <= 0.51:
        off = (db_idx[0] % 2) if db_idx else 0
        beats = beats[off::2]
        db_idx = sorted({(i - off) // 2 for i in db_idx if (i - off) % 2 == 0})

    if onsets is not None:
        beats = _snap_to_onsets(beats, np.sort(np.asarray(onsets, dtype=float)))

    # 向前、向后补拍，覆盖整首歌
    iv0 = float(np.median(np.diff(beats[:5])))
    iv1 = float(np.median(np.diff(beats[-5:])))
    pre = []
    t = beats[0] - iv0
    while t > -0.5 * iv0 and t > -0.02 - 0.0 and len(pre) < 4000:
        if t < 0:
            break
        pre.append(t)
        t -= iv0
    pre = pre[::-1]
    post = []
    t = beats[-1] + iv1
    while t < duration + iv1 and len(post) < 4000:
        post.append(t)
        t += iv1
    n_pre = len(pre)
    beats = np.concatenate([pre, beats, post])
    db_idx = [i + n_pre for i in db_idx]
    n_beats = len(beats) - 1  # 最后一个只当收尾

    # 主拍号
    diffs = np.diff(db_idx)
    if beats_per_bar:
        N = int(beats_per_bar)
    elif len(diffs):
        vals, cnt = np.unique(diffs, return_counts=True)
        N = int(vals[np.argmax(cnt)])
        if N > 12:
            N = 4
        if N <= 2:
            # 模型把半小节也标成强拍是常见错误；真正的 2/4 很少见，统一按 4 拍记
            N = 4
    else:
        N = 4

    if db_idx:
        ph = int(np.bincount(np.array(db_idx) % N, minlength=N).argmax())
    else:
        ph = 0
    if beats_per_bar or len(db_idx) < 4:
        # 规则网格：用多数强拍投票决定相位
        starts = list(range(ph, n_beats, N))
    else:
        # 以规则网格为骨架往前走；只有当后面连续 3 个强拍都支持新相位时，
        # 才认为这里真的多/少了几拍（插入一个不规则小节），否则忽略这个强拍。
        db = db_idx
        a0 = next((i for i, p in enumerate(db) if p % N == ph), 0)
        cur = db[a0]
        starts = [cur]
        for i in range(a0 + 1, len(db)):
            p = db[i]
            while p - cur >= N:
                cur += N
                starts.append(cur)
            if p == cur:
                continue
            ahead = db[i + 1 : i + 4]
            if len(ahead) == 3 and all((q - p) % N == 0 for q in ahead):
                if p - cur == 1 and len(starts) > 1:
                    starts.pop()          # 多出的 1 拍并入前一小节
                starts.append(p)
                cur = p
        # 首个强拍之前、最后强拍之后按主拍号外推
        k = starts[0] - N
        while k >= 0:
            starts.insert(0, k)
            k -= N
        k = starts[-1] + N
        while k < n_beats:
            starts.append(k)
            k += N

    starts = [s + beat_shift for s in starts]
    starts = sorted({s for s in starts if 0 <= s < n_beats})
    anacrusis = False
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
        anacrusis = len(starts) > 1 and (starts[1] - starts[0]) < N
    bars = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_beats
        if e - s >= 1:
            bars.append((s, e - s))
    # 末小节补足
    if len(bars) >= 2 and bars[-1][1] < N:
        s, n = bars[-1]
        need = N - n
        iv = beats[-1] - beats[-2]
        beats = np.concatenate([beats, beats[-1] + iv * np.arange(1, need + 1)])
        bars[-1] = (s, N)

    iv = np.sort(np.diff(raw_beats))
    k = len(iv) // 5
    core = iv[k : len(iv) - k] if len(iv) > 10 else iv
    tempo = 60.0 / float(np.mean(core))
    if tempo_scale >= 1.99:
        tempo *= 2
    elif tempo_scale <= 0.51:
        tempo /= 2
    return Grid(beat_times=beats, bars=bars, numerator=N, tempo=float(tempo), anacrusis=anacrusis)
