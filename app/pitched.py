"""有音高乐器的识别：Spotify basic-pitch（NeuralNote 用的同一个模型），在分离后的单乐器音轨上运行。"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.getLogger().setLevel(logging.ERROR)

_model = None

# (最低频率, 最高频率, 起音阈值, 持续阈值, 最短音长 ms, 是否单音)
PARAMS = {
    "bass":   (30.0, 420.0, 0.5, 0.3, 90.0, True),
    "vocals": (75.0, 1100.0, 0.55, 0.35, 110.0, True),
    "guitar": (78.0, 1400.0, 0.5, 0.3, 70.0, False),
    "piano":  (27.5, 4200.0, 0.5, 0.3, 70.0, False),
    "other":  (40.0, 2500.0, 0.55, 0.35, 90.0, False),
}


def _get_model():
    global _model
    if _model is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        _model = Model(ICASSP_2022_MODEL_PATH)
    return _model


def _monophonic(notes: list[dict]) -> list[dict]:
    notes = sorted(notes, key=lambda n: (n["start"], -n["amp"]))
    out: list[dict] = []
    for n in notes:
        if out:
            p = out[-1]
            if n["start"] < p["end"] - 0.02:
                # 重叠：响的那个留下
                if n["amp"] >= 0.8 * p["amp"] or n["start"] - p["start"] > 0.6 * (p["end"] - p["start"]):
                    p["end"] = n["start"]
                    if p["end"] - p["start"] < 0.05:
                        out.pop()
                else:
                    continue
        out.append(dict(n))
    return out


def transcribe(stem_path: str | Path, key: str) -> list[dict]:
    """返回 [{start, end, pitch, amp}]。"""
    from basic_pitch.inference import predict

    fmin, fmax, on_thr, fr_thr, min_len, mono = PARAMS[key]
    _, _, events = predict(
        str(stem_path),
        _get_model(),
        onset_threshold=on_thr,
        frame_threshold=fr_thr,
        minimum_note_length=min_len,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )
    notes = [
        {"start": float(s), "end": float(e), "pitch": int(p), "amp": float(a)}
        for s, e, p, a, *_ in events
    ]
    if not notes:
        return []
    # 去掉明显是串音的弱音
    amps = np.array([n["amp"] for n in notes])
    floor = max(0.12, 0.35 * float(np.percentile(amps, 75)))
    notes = [n for n in notes if n["amp"] >= floor]
    if mono:
        notes = _monophonic(notes)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes
