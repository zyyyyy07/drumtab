"""整条处理流程。分成两段：
analyze()：耗时的模型推理（分轨、节拍、鼓、音高），结果存盘；
render()：从存盘结果生成乐谱，秒级完成，方便调整小节线/灵敏度后重新出谱。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio_io, beats, notation

STEM_NAMES = {"drums": "鼓", "bass": "贝斯", "guitar": "吉他", "piano": "键盘", "vocals": "人声", "other": "其他乐器"}
PITCHED = ["bass", "guitar", "piano", "vocals", "other"]
DEFAULT_OPTIONS = {
    "beat_shift": 0,          # 小节线整体挪动几拍
    "beats_per_bar": 0,       # 0 = 自动
    "tempo_scale": 1.0,       # 2 = 速度翻倍记谱，0.5 = 减半
    "drum_sensitivity": 0.0,  # -1..1
    "ghost_notes": True,
    "open_hihat": True,
}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def analyze(src: Path, out: Path, progress: Callable[[float, str], None]) -> None:
    from . import drums, pitched, separate

    dev = _device()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    progress(0.02, "读取音频")
    audio = audio_io.load_audio(src)
    duration = audio.shape[1] / audio_io.SR
    if duration < 3:
        raise RuntimeError("音频太短")
    audio_io.save_flac(out / "mix.flac", audio)

    stems = separate.separate(audio, dev, True, lambda p, m: progress(0.05 + 0.40 * p, m))
    mix_db = audio_io.rms_db(audio)
    levels = {}
    for k, v in stems.items():
        audio_io.save_flac(out / f"{k}.flac", v)
        levels[k] = audio_io.rms_db(v) - mix_db
    audio_io.save_flac(out / "nodrums.flac", audio - stems["drums"])

    progress(0.48, "识别拍子和小节")
    mono = audio.mean(0)
    b, d = beats.track(mono, audio_io.SR, dev)

    progress(0.55, "识别鼓")
    drum_mono = stems["drums"].mean(0)
    act_stem = drums.activations(drum_mono, dev)
    act_mix = drums.activations(mono, dev)
    np.savez_compressed(out / "drum_acts.npz", stem=act_stem.astype(np.float16), mix=act_mix.astype(np.float16))

    notes: dict[str, list] = {}
    for i, k in enumerate(PITCHED):
        progress(0.62 + 0.07 * i, f"识别{STEM_NAMES[k]}")
        if levels[k] < -30:  # 这件乐器在歌里基本不存在
            notes[k] = []
            continue
        try:
            notes[k] = pitched.transcribe(out / f"{k}.flac", k)
        except Exception as e:
            print(f"[pipeline] {k} 识别失败：", e)
            notes[k] = []

    analysis = {
        "duration": duration,
        "beats": [float(x) for x in b],
        "downbeats": [float(x) for x in d],
        "levels": levels,
        "notes": notes,
        "elapsed": time.time() - t0,
    }
    (out / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    progress(0.97, "生成乐谱")


def drum_hits(out: Path, options: dict, beat_period: float, grid_fn=None) -> list[dict]:
    """grid_fn(底鼓/军鼓时间) -> Grid；给了就按小节重复规律补回差一点没过线的音。"""
    from . import drums

    z = np.load(out / "drum_acts.npz")
    raw = drums.detect(z["stem"].astype(np.float32), z["mix"].astype(np.float32), float(options.get("drum_sensitivity", 0.0)))
    if grid_fn is not None:
        grid = grid_fn(
            np.array(sorted([t for t, _ in raw["KD"]] + [t for t, _ in raw["SD"]])),
            np.array([t for t, _ in raw["KD"]]),
        )
        raw = drums.fill_by_pattern(raw, grid)
    raw.pop("_low", None)
    drum_audio, _ = audio_io.read_audio_file(out / "drums.flac")
    feats = drums.StemFeatures(drum_audio.mean(0))
    hits = drums.refine(raw, feats, beat_period)
    res = []
    for h in hits:
        d = h.to_dict()
        if not options.get("ghost_notes", True):
            d["ghost"] = False
        if not options.get("open_hihat", True) and d["kind"] == "hh_open":
            d["kind"] = "hh_closed"
        res.append(d)
    return res


def fuse_acts(stem: np.ndarray, mix: np.ndarray) -> np.ndarray:
    from . import drums

    return drums.fuse(stem, mix)


def render(out: Path, title: str, options: dict | None = None, hits: list[dict] | None = None) -> dict:
    opts = dict(DEFAULT_OPTIONS)
    opts.update(options or {})
    an = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    b, d = np.array(an["beats"]), np.array(an["downbeats"])
    beat_period = float(np.median(np.diff(b))) if len(b) > 2 else 0.5
    beat_period /= float(opts["tempo_scale"]) or 1.0

    def make_grid(onsets, kicks=None):
        return beats.build_grid(
            b, d, an["duration"], onsets=onsets, kicks=kicks,
            beat_shift=int(opts["beat_shift"]),
            beats_per_bar=int(opts["beats_per_bar"]) or None,
            tempo_scale=float(opts["tempo_scale"]),
        )

    if hits is None:
        hits = drum_hits(out, opts, beat_period, make_grid)
    grid = make_grid(
        np.array([h["t"] for h in hits if h["kind"] in ("kick", "snare")]),
        np.array([h["t"] for h in hits if h["kind"] == "kick"]),
    )

    tracks = []
    dtrack, ternary = notation.drum_track(hits, grid)
    if dtrack.note_count >= 4:
        tracks.append(dtrack)
    all_notes = [n for k in PITCHED for n in an["notes"].get(k, [])]
    ks = notation.estimate_key([n for k in ("bass", "guitar", "piano", "vocals") for n in an["notes"].get(k, [])] or all_notes)
    for k in PITCHED:
        ns = an["notes"].get(k, [])
        if len(ns) < 8:
            continue
        # 有鼓时，其余声部的三连音判断跟随鼓，全谱记法一致
        tracks.append(notation.pitched_track(k, ns, grid, ternary if dtrack.note_count >= 16 else None, ks))

    sub = f"♩ = {int(round(grid.tempo))}　{grid.numerator}/4　自动扒谱"
    tex = notation.build_alphatex(title, grid, tracks, sub)
    (out / "score.atex").write_text(tex, encoding="utf-8")

    counts: dict[str, int] = {}
    for h in hits:
        counts[h["kind"]] = counts.get(h["kind"], 0) + 1
    info = {
        "tracks": [{"key": t.key, "name": t.name, "notes": t.note_count} for t in tracks],
        "tempo": grid.tempo,
        "numerator": grid.numerator,
        "bars": len(grid.bars),
        "bar_times": [grid.bar_time(i) for i in range(len(grid.bars))],
        "duration": an["duration"],
        "key": ks,
        "drum_counts": counts,
        "options": opts,
    }
    (out / "score.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info
