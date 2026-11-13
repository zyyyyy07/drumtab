"""把“时间点事件”变成乐谱：量化到拍子网格 -> 切分成可记谱的时值 -> 输出 alphaTex。

内部时间单位：每个四分音符 24 tick（可同时整除十六分、三十二分、三连音、六连音）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .beats import Grid

TPQ = 24

# ---------------------------------------------------------------- 量化


def quantize_onsets(
    pos: np.ndarray,
    keys: list,
    n_beats: int,
    allow_fine: bool = True,
    beat_period: float = 0.5,
    ternary_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """pos：每个音头的连续拍位置；keys：同 key 的两个音不应落进同一格。

    返回 (每个音头的 tick, 每拍是否为三连音拍)。
    每一拍独立在 十六分 / 三连音 / 三十二分 / 六连音 网格里选误差最小的一种。
    """
    pos = np.asarray(pos, dtype=float)
    eps = 1.0 / 16.0
    k = np.floor(pos + eps).astype(int)
    k = np.clip(k, 0, max(0, n_beats - 1))
    frac = pos - k
    by_beat: dict[int, list[int]] = {}
    for i, b in enumerate(k):
        by_beat.setdefault(int(b), []).append(i)

    fast = max(1.0, 0.45 / max(beat_period, 0.15))  # 速度越快，越不愿意用很细的格子

    def run(pen: dict[int, float]):
        grids = np.full(n_beats, 4, dtype=int)
        for b, idxs in by_beat.items():
            fr = frac[idxs]
            best, best_cost = 4, 1e9
            for G in (4, 3, 8, 6):
                if G in (8, 6) and not allow_fine:
                    continue
                # 有参考声部（鼓）时，这一拍是不是三连音由鼓说了算
                if ternary_hint is not None and (G in (3, 6)) != bool(ternary_hint[b]):
                    continue
                slot = np.round(fr * G)
                cost = float(np.abs(fr - slot / G).sum()) + pen[G]
                seen = set()
                for j, s in zip(idxs, slot):
                    key = (int(s), keys[j])
                    if key in seen:
                        cost += 0.12
                    seen.add(key)
                if cost < best_cost - 1e-9:
                    best, best_cost = G, cost
            grids[b] = best
        return grids

    pen = {4: 0.0, 3: 0.03, 8: 0.07 * fast, 6: 0.09 * fast}
    if beat_period < 0.55:
        # 速度快于约 110 BPM 时，三十二分音符几乎都是识别误差，不用；六连音也要更强的证据
        pen[8] = 9.0
        pen[6] = 0.2 * fast
    grids = run(pen)
    # 整首歌若以三连音律动为主（shuffle / 摇摆），反过来偏向三连音
    multi = [b for b, idxs in by_beat.items() if len({round(float(f), 2) for f in frac[idxs]}) >= 2]
    if multi and ternary_hint is None:
        share = np.mean([grids[b] in (3, 6) for b in multi])
        if share > 0.35:
            grids = run({4: 0.04, 3: 0.0, 8: pen[8] + 0.03 * fast, 6: min(pen[6], 0.05 * fast)})
    if ternary_hint is not None:
        # 跟随参考声部（鼓）的三连音判断，保证各声部记法一致
        for b in range(n_beats):
            if b not in by_beat:
                grids[b] = 3 if ternary_hint[b] else 4

    ticks = np.zeros(len(pos), dtype=int)
    for i in range(len(pos)):
        G = int(grids[k[i]])
        ticks[i] = k[i] * TPQ + int(round(frac[i] * G)) * (TPQ // G)
    ticks = np.clip(ticks, 0, n_beats * TPQ - 1)
    ternary = np.isin(grids, (3, 6))
    return ticks, ternary


# ---------------------------------------------------------------- 切分时值


@dataclass
class Piece:
    size: int                 # tick
    payload: object | None    # None = 休止
    tied: bool = False
    tuplet: int = 0           # 0 / 3 / 6


_BIN = [(18, (0, 6)), (12, (0, 6, 12)), (9, (0, 6, 12)), (6, (0, 3, 6, 9, 12, 15, 18)), (3, None)]
_TER = [(16, (0, 8)), (8, (0, 4, 8, 12, 16)), (4, None)]


def _snap(tick: int, ternary: np.ndarray, n_beats: int) -> int:
    b, off = divmod(tick, TPQ)
    if b >= n_beats:
        return n_beats * TPQ
    unit = 4 if ternary[b] else 3
    off = int(round(off / unit)) * unit
    return b * TPQ + min(off, TPQ)


def split_segment(s: int, e: int, payload, ternary: np.ndarray, n_beats: int, long_notes: bool) -> list[Piece]:
    """把 [s, e) 切成合法时值。s/e 为小节内 tick；ternary 为本小节每拍是否三连音。"""
    out: list[Piece] = []
    pos = s
    is_note = payload is not None
    while pos < e:
        beat, off = divmod(pos, TPQ)
        rem = e - pos
        if off == 0 and rem >= TPQ:
            nb = rem // TPQ
            if nb >= 4 and beat % 4 == 0:
                size = 4 * TPQ
            elif nb >= 3 and beat % 2 == 0 and is_note and long_notes and beat + 3 <= n_beats:
                size = 3 * TPQ
            elif nb >= 2 and beat % 2 == 0:
                size = 2 * TPQ
            elif (
                is_note and long_notes and rem >= 36 and beat + 1 < n_beats and not ternary[beat + 1]
            ):
                size = 36
            else:
                size = TPQ
            out.append(Piece(size, payload, tied=is_note and pos > s))
        else:
            limit = min(rem, TPQ - off)
            table = _TER if ternary[beat] else _BIN
            size = None
            for sz, al in table:
                if sz <= limit and (al is None or off in al):
                    size = sz
                    break
            if size is None:  # 理论上不会发生（已对齐网格），兜底
                size = limit
            out.append(Piece(size, payload, tied=is_note and pos > s, tuplet=3 if ternary[beat] else 0))
        pos += size
    return out


def engrave_bar(segments: list[tuple[int, int, object]], n_beats: int, ternary: np.ndarray, long_notes: bool) -> list[Piece]:
    """segments：小节内互不重叠的 (起, 止, 内容)。空隙自动补休止。"""
    total = n_beats * TPQ
    segs = []
    for s, e, p in sorted(segments, key=lambda x: x[0]):
        s, e = _snap(s, ternary, n_beats), _snap(e, ternary, n_beats)
        if segs and s < segs[-1][1]:
            s = segs[-1][1]
        if e <= s:
            continue
        segs.append((s, min(e, total), p))
    pieces: list[Piece] = []
    cur = 0
    for s, e, p in segs:
        if s > cur:
            pieces += split_segment(cur, s, None, ternary, n_beats, long_notes)
        pieces += split_segment(s, e, p, ternary, n_beats, long_notes)
        cur = e
    if cur < total:
        pieces += split_segment(cur, total, None, ternary, n_beats, long_notes)
    # 同一拍内只要出现六连音的十六分，这一拍统一记成 6 连音
    pos = 0
    by_beat: dict[int, list[Piece]] = {}
    for pc in pieces:
        if pc.tuplet:
            by_beat.setdefault(pos // TPQ, []).append(pc)
        pos += pc.size
    for lst in by_beat.values():
        if any(pc.size == 4 for pc in lst):
            for pc in lst:
                pc.tuplet = 6
    return pieces


_DUR = {96: (1, 0), 72: (2, 1), 48: (2, 0), 36: (4, 1), 24: (4, 0), 18: (8, 1), 12: (8, 0), 9: (16, 1), 6: (16, 0), 3: (32, 0)}
_DUR_T = {16: 4, 8: 8, 4: 16}


def piece_suffix(pc: Piece) -> str:
    if pc.tuplet:
        return f".{_DUR_T[pc.size]}{{tu {pc.tuplet}}}"
    d, dots = _DUR.get(pc.size, (16, 0))
    return f".{d}{{d}}" if dots else f".{d}"


# ---------------------------------------------------------------- 鼓

DRUM_ART = {
    "kick": "Kick",
    "snare": "Snare",
    "sidestick": "Stick",
    "hh_closed": "HH",
    "hh_open": "HHOpen",
    "hh_pedal": "HHPedal",
    "crash": "Crash",
    "ride": "Ride",
    "ride_bell": "RideBell",
    "tom_hi": "TomHi",
    "tom_mid": "TomMid",
    "tom_floor": "TomFloor",
}
# 自定义名字 -> alphaTab 内置打击乐编号（与 Guitar Pro 一致）
DRUM_ART_ID = {
    "Kick": 36, "Snare": 38, "Stick": 37, "HH": 42, "HHOpen": 46, "HHPedal": 44,
    "Crash": 49, "Ride": 51, "RideBell": 53, "TomHi": 48, "TomMid": 47, "TomFloor": 43,
}
_DRUM_ORDER = ["kick", "tom_floor", "snare", "sidestick", "tom_mid", "tom_hi", "hh_pedal", "hh_closed", "hh_open", "ride", "ride_bell", "crash"]


@dataclass
class TrackOut:
    key: str
    name: str
    short: str
    header: list[str]                       # \track 之后、小节之前的元数据
    staves: list[list[str]] = field(default_factory=list)   # 每个谱表 -> 每小节的 alphaTex
    staff_headers: list[str] = field(default_factory=list)
    note_count: int = 0


def _bar_ranges(grid: Grid):
    for s, n in grid.bars:
        yield s * TPQ, n


def drum_track(hits: list[dict], grid: Grid) -> tuple[TrackOut, np.ndarray]:
    n_beats_total = len(grid.beat_times) - 1
    beat_period = 60.0 / grid.tempo
    if hits:
        pos = grid.pos(np.array([h["t"] for h in hits]))
        keys = [h["kind"] for h in hits]
        ticks, ternary = quantize_onsets(pos, keys, n_beats_total, True, beat_period)
    else:
        ticks, ternary = np.zeros(0, dtype=int), np.zeros(n_beats_total, dtype=bool)

    slots: dict[int, dict[str, dict]] = {}
    for h, tk in zip(hits, ticks):
        d = slots.setdefault(int(tk), {})
        if h["kind"] not in d or h["vel"] > d[h["kind"]]["vel"]:
            d[h["kind"]] = h
    # 同一格里开镲盖过闭镲
    for d in slots.values():
        if "hh_open" in d and "hh_closed" in d:
            del d["hh_closed"]

    all_ticks = sorted(slots)
    bars: list[str] = []
    ai = 0
    for start, n in _bar_ranges(grid):
        end = start + n * TPQ
        segs = []
        while ai < len(all_ticks) and all_ticks[ai] < start:
            ai += 1
        j = ai
        while j < len(all_ticks) and all_ticks[j] < end:
            t = all_ticks[j]
            nxt = all_ticks[j + 1] if j + 1 < len(all_ticks) else end
            beat_end = (t // TPQ + 1) * TPQ
            e = min(nxt, beat_end, end)
            segs.append((t - start, e - start, slots[t]))
            j += 1
        tern = ternary[start // TPQ : start // TPQ + n]
        pieces = engrave_bar(segs, n, tern, long_notes=False)
        toks = []
        for pc in pieces:
            if pc.payload is None:
                toks.append("r" + piece_suffix(pc))
            else:
                names = []
                for kind in _DRUM_ORDER:
                    if kind in pc.payload:
                        h = pc.payload[kind]
                        nm = DRUM_ART[kind]
                        if h.get("ghost"):
                            nm += "{g}"
                        names.append(nm)
                toks.append("(" + " ".join(names) + ")" + piece_suffix(pc))
        bars.append(" ".join(toks))

    header = ["\\staff {score}", "\\instrument percussion", "\\clef neutral"]
    header += [f'\\articulation ("{k}" {v})' for k, v in DRUM_ART_ID.items()]
    tr = TrackOut("drums", "鼓", "鼓", [], [bars], [" ".join(header[:3]) + "\n" + "\n".join(header[3:])], note_count=len(hits))
    return tr, ternary


# ---------------------------------------------------------------- 有音高的乐器

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(p: int) -> str:
    return f"{_NAMES[p % 12]}{p // 12 - 1}"


def _quantize_notes(notes: list[dict], grid: Grid, ternary_hint: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """返回 [(起 tick, 止 tick, 音高)]（全曲绝对 tick）与每拍三连音标记。"""
    n_beats_total = len(grid.beat_times) - 1
    if not notes:
        return [], np.zeros(n_beats_total, dtype=bool)
    pos_s = grid.pos(np.array([n["start"] for n in notes]))
    pos_e = grid.pos(np.array([n["end"] for n in notes]))
    keys = [n["pitch"] for n in notes]
    ts, ternary = quantize_onsets(pos_s, keys, n_beats_total, False, 60.0 / grid.tempo, ternary_hint)
    out = []
    for n, s, pe in zip(notes, ts, pos_e):
        b = int(min(max(np.floor(pe), 0), n_beats_total - 1))
        unit = 8 if ternary[b] else 6
        e = b * TPQ + int(round((pe - b) * TPQ / unit)) * unit
        b0 = int(s) // TPQ
        min_len = 8 if ternary[b0] else 6
        e = max(e, int(s) + min_len)
        out.append((int(s), int(e), int(n["pitch"])))
    return out, ternary


def _voice_bars(qnotes: list[tuple[int, int, int]], grid: Grid, ternary: np.ndarray, fmt, max_poly: int = 6) -> list[str]:
    """单声部记谱：同时起音的音合成和弦，持续到下一个起音为止（跨拍/跨小节用连音线）。"""
    onsets: dict[int, dict[int, int]] = {}
    for s, e, p in qnotes:
        d = onsets.setdefault(s, {})
        d[p] = max(d.get(p, 0), e)
    order = sorted(onsets)
    # 全曲的“段”：(起, 止, 音高元组)
    segs: list[tuple[int, int, tuple[int, ...]]] = []
    for i, s in enumerate(order):
        nxt = order[i + 1] if i + 1 < len(order) else 10**9
        chord = onsets[s]
        e = min(max(chord.values()), nxt)
        pitches = tuple(sorted(chord)[-max_poly:])
        if e > s:
            segs.append((s, e, pitches))
    bars: list[str] = []
    si = 0
    carry: tuple[int, tuple[int, ...]] | None = None  # (绝对止点, 音高)
    state = {"prev": None}
    for start, n in _bar_ranges(grid):
        end = start + n * TPQ
        local: list[tuple[int, int, object]] = []
        if carry is not None:
            ce, cp = carry
            local.append((0, min(ce, end) - start, ("tie", cp)))
            carry = (ce, cp) if ce > end else None
        while si < len(segs) and segs[si][0] < end:
            s, e, p = segs[si]
            if s >= start:
                if local and local[-1][1] > s - start:  # 被新的起音打断
                    a, _, pl = local[-1]
                    local[-1] = (a, s - start, pl)
                    carry = None
                local.append((s - start, min(e, end) - start, ("new", p)))
                carry = (e, p) if e > end else None
            si += 1
        tern = ternary[start // TPQ : start // TPQ + n]
        pieces = engrave_bar(local, n, tern, long_notes=True)
        toks = []
        for pc in pieces:
            if pc.payload is None:
                toks.append("r" + piece_suffix(pc))
                state["prev"] = None
            else:
                kind, pitches = pc.payload
                tied = pc.tied or kind == "tie"
                toks.append(fmt(pitches, tied, state) + piece_suffix(pc))
        bars.append(" ".join(toks))
    return bars


def _fmt_score(pitches, tied, state) -> str:
    return "(" + " ".join(pitch_name(p) + ("{t}" if tied else "") for p in pitches) + ")"


class TabFingering:
    """把音高分配到弦和品：尽量贴近当前把位、跨度不超过 4 品。"""

    def __init__(self, tuning: list[int], max_fret: int = 22):
        self.tuning = tuning  # 从最细的弦到最粗的弦（1 弦在前）
        self.max_fret = max_fret
        self.hand = 3.0
        self.last: dict[int, tuple[int, int]] = {}

    def fit(self, p: int) -> int:
        lo, hi = min(self.tuning), max(self.tuning) + self.max_fret
        while p < lo:
            p += 12
        while p > hi:
            p -= 12
        return p

    def assign(self, pitches: tuple[int, ...]) -> dict[int, tuple[int, int]]:
        pitches = tuple(sorted({self.fit(p) for p in pitches}))
        cands = []
        for p in pitches:
            c = [(si + 1, p - open_) for si, open_ in enumerate(self.tuning) if 0 <= p - open_ <= self.max_fret]
            cands.append(c)
        best, best_cost = None, 1e9

        def rec(i, used, cur):
            nonlocal best, best_cost
            if i == len(pitches):
                frets = [f for _, f in cur.values() if f > 0]
                span = (max(frets) - min(frets)) if frets else 0
                if span > 5:
                    return
                center = np.mean(frets) if frets else self.hand
                mean_fret = sum(f for _, f in cur.values()) / max(1, len(cur))
                # 贴近当前把位、跨度小、尽量在低把位（自动扒谱不知道真实指法，低把位最通用）
                cost = 0.6 * abs(center - self.hand) + 1.5 * span + 0.35 * mean_fret
                cost += 1.0 * sum(1 for _, f in cur.values() if f > 12)
                if cost < best_cost:
                    best, best_cost = dict(cur), cost
                return
            for s, f in cands[i]:
                if s in used:
                    continue
                cur[pitches[i]] = (s, f)
                rec(i + 1, used | {s}, cur)
                del cur[pitches[i]]

        rec(0, frozenset(), {})
        if best is None:  # 和弦按不下来：从最低音开始逐个丢弃
            if len(pitches) > 1:
                return self.assign(pitches[1:])
            return {}
        frets = [f for _, f in best.values() if f > 0]
        if frets:
            self.hand = 0.6 * self.hand + 0.4 * float(np.mean(frets))
        return best

    def fmt(self, pitches, tied, state) -> str:
        if tied and state.get("prev"):
            a = state["prev"]
        else:
            a = self.assign(pitches)
            state["prev"] = a
        if not a:
            return "r"
        parts = [f"{f}.{s}" + ("{t}" if tied else "") for s, f in sorted(a.values())]
        return "(" + " ".join(parts) + ")"


GUITAR_TUNING = [64, 59, 55, 50, 45, 40]
BASS_TUNING = [43, 38, 33, 28]


def _tuning_tex(tuning: list[int]) -> str:
    return "\\tuning (" + " ".join(pitch_name(p) for p in tuning) + ")"


def pitched_track(key: str, notes: list[dict], grid: Grid, ternary_hint: np.ndarray, ks: str) -> TrackOut:
    q, tern = _quantize_notes(notes, grid, ternary_hint)
    if key == "bass":
        # 贝斯常被识别成高八度的泛音：折回贝斯的常用音域（E1 ~ E3）
        q = [(s, e, p - 12 * max(0, (p - 52 + 11) // 12)) for s, e, p in q]
        fing = TabFingering(BASS_TUNING, 20)
        bars = _voice_bars(q, grid, tern, fing.fmt, max_poly=1)
        return TrackOut(key, "贝斯", "贝斯", [], [bars], [f"\\staff {{tabs}} \\instrument electricbassfinger {_tuning_tex(BASS_TUNING)}"], len(q))
    if key == "guitar":
        fing = TabFingering(GUITAR_TUNING, 19)
        bars = _voice_bars(q, grid, tern, fing.fmt, max_poly=6)
        return TrackOut(key, "吉他", "吉他", [], [bars], [f"\\staff {{tabs}} \\instrument overdrivenguitar {_tuning_tex(GUITAR_TUNING)}"], len(q))
    if key == "piano":
        rh = [n for n in q if n[2] >= 60]
        lh = [n for n in q if n[2] < 60]
        b1 = _voice_bars(rh, grid, tern, _fmt_score, max_poly=5)
        b2 = _voice_bars(lh, grid, tern, _fmt_score, max_poly=4)
        return TrackOut(
            key, "键盘", "键盘", [], [b1, b2],
            [f"\\staff {{score}} \\tuning piano \\instrument acousticgrandpiano \\clef G2 \\ks {ks}",
             f"\\staff {{score}} \\tuning piano \\clef F4 \\ks {ks}"],
            len(q),
        )
    if key == "vocals":
        name = "人声"
        if q:
            med = float(np.median([n[2] for n in q]))
            if med < 57:  # 男声：按惯例高八度记在高音谱表
                q = [(s, e, p + 12) for s, e, p in q]
                name = "人声（高八度记谱）"
        bars = _voice_bars(q, grid, tern, _fmt_score, max_poly=1)
        return TrackOut(key, name, "人声", [], [bars], [f"\\staff {{score}} \\tuning piano \\instrument voiceoohs \\clef G2 \\ks {ks}"], len(q))
    # other：合成器、弦乐、管乐等
    med = float(np.median([n[2] for n in q])) if q else 60
    clef = "G2" if med >= 60 else "F4"
    bars = _voice_bars(q, grid, tern, _fmt_score, max_poly=4)
    return TrackOut(key, "其他乐器", "其他", [], [bars], [f"\\staff {{score}} \\tuning piano \\instrument stringensemble1 \\clef {clef} \\ks {ks}"], len(q))


# ---------------------------------------------------------------- 调号

_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_KS_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def estimate_key(all_notes: list[dict]) -> str:
    """Krumhansl 调性估计，返回 alphaTex 的 \\ks 值（统一用关系大调表示）。"""
    hist = np.zeros(12)
    for n in all_notes:
        hist[int(n["pitch"]) % 12] += max(0.05, n["end"] - n["start"])
    if hist.sum() <= 0:
        return "C"
    best, score = 0, -9
    for k in range(12):
        for prof, rel in ((_MAJ, 0), (_MIN, 3)):
            c = np.corrcoef(np.roll(prof, k), hist)[0, 1]
            if c > score:
                score, best = c, (k + rel) % 12
    return _KS_MAJOR[best]


# ---------------------------------------------------------------- 汇总成 alphaTex


def build_alphatex(title: str, grid: Grid, tracks: list[TrackOut], subtitle: str = "") -> str:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = [f'\\title "{esc(title)}"']
    if subtitle:
        lines.append(f'\\subtitle "{esc(subtitle)}"')
    lines.append(f"\\tempo {int(round(grid.tempo))}")
    nbars = len(grid.bars)
    first = True
    for tr in tracks:
        lines.append(f'\\track "{esc(tr.name)}" "{esc(tr.short)}"')
        for sh, bars in zip(tr.staff_headers, tr.staves):
            lines.append(sh)
            prev_n = None
            out = []
            for i in range(nbars):
                n = grid.bars[i][1]
                meta = ""
                if n != prev_n:
                    meta += f"\\ts ({n} 4) "
                    prev_n = n
                if i == 0 and grid.anacrusis:
                    meta = f"\\ts ({grid.numerator} 4) \\ac "
                    prev_n = grid.numerator
                out.append(meta + (bars[i] if i < len(bars) and bars[i] else "r.1"))
            lines.append(" |\n".join(out))
        first = False
    # 同步点：每小节一个，把谱面时间轴钉在原始音频上
    for i in range(nbars):
        ms = int(round(grid.bar_time(i) * 1000))
        lines.append(f"\\sync ({i} 0 {max(0, ms)})")
    s, n = grid.bars[-1]
    lines.append(f"\\sync ({nbars - 1} 0 {int(round(grid.beat_times[s + n] * 1000))} 1)")
    return "\n".join(lines) + "\n"
