"""造一个“已完成”的测试项目：鼓用人工标注，其余走真实流程。用于前端联调。"""
import sys, os, json, time, shutil
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from app import audio_io, beats, pitched, pipeline, separate

here = Path(__file__).parent
root = here / "testdata" / "MDBDrums" / "MDB Drums"
name = sys.argv[1] if len(sys.argv) > 1 else "MusicDelta_Rock"
pid = sys.argv[2] if len(sys.argv) > 2 else "0123456789ab"
out = (Path(__file__).resolve().parent.parent / "projects") / pid
out.mkdir(parents=True, exist_ok=True)

MAP = {"KD": "kick", "SD": "snare", "SDG": "snare", "SDB": "snare", "SDF": "snare", "SDD": "snare", "SDNS": "snare",
       "SST": "sidestick", "CHH": "hh_closed", "OHH": "hh_open", "PHH": "hh_pedal", "RDC": "ride", "RDB": "ride_bell",
       "CRC": "crash", "CHC": "crash", "SPC": "crash", "HIT": "tom_hi", "MHT": "tom_mid", "HFT": "tom_floor", "LFT": "tom_floor"}
hits = []
for line in open(root / "annotations" / "subclass" / f"{name}_subclass.txt"):
    p = line.split()
    if len(p) == 2 and p[1] in MAP:
        hits.append({"t": float(p[0]), "kind": MAP[p[1]], "vel": 0.8, "ghost": p[1] == "SDG"})

a = audio_io.load_audio(root / "audio" / "full_mix" / f"{name}_MIX.wav")
dur = a.shape[1] / 44100
audio_io.save_flac(out / "mix.flac", a)
st = separate.separate(a)
mix_db = audio_io.rms_db(a)
levels = {}
for k, v in st.items():
    audio_io.save_flac(out / f"{k}.flac", v)
    levels[k] = audio_io.rms_db(v) - mix_db
audio_io.save_flac(out / "nodrums.flac", a - st["drums"])
b, d = beats.track(a.mean(0), 44100)
notes = {k: (pitched.transcribe(out / f"{k}.flac", k) if levels[k] >= -30 else []) for k in pipeline.PITCHED}
(out / "analysis.json").write_text(json.dumps({"duration": dur, "beats": list(map(float, b)), "downbeats": list(map(float, d)), "levels": levels, "notes": notes}), encoding="utf-8")
info = pipeline.render(out, name, None, hits=hits)
meta = {"id": pid, "title": name, "filename": name + ".wav", "created": time.time(), "status": "done", "progress": 1.0, "message": "完成", "error": "", "options": info["options"]}
(out / "project.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in info.items() if k != "bar_times"}, ensure_ascii=False))
print("levels", {k: round(v, 1) for k, v in levels.items()})
