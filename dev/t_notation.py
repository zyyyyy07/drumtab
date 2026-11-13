"""用人工标注的鼓 + 实际识别的音高乐器，生成一份 alphaTex，验证记谱引擎。"""
import sys, os, json, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from app import audio_io, beats, notation, pitched

here = os.path.dirname(__file__)
root = os.path.join(here, "testdata", "MDBDrums", "MDB Drums")
name = sys.argv[1] if len(sys.argv) > 1 else "MusicDelta_Rock"
out = os.path.join(here, "out", name)

MAP = {"KD": "kick", "SD": "snare", "SDG": "snare", "SDB": "snare", "SDF": "snare", "SDD": "snare", "SDNS": "snare",
       "SST": "sidestick", "CHH": "hh_closed", "OHH": "hh_open", "PHH": "hh_pedal", "RDC": "ride", "RDB": "ride_bell",
       "CRC": "crash", "CHC": "crash", "SPC": "crash", "HIT": "tom_hi", "MHT": "tom_mid", "HFT": "tom_floor", "LFT": "tom_floor"}
hits = []
for line in open(os.path.join(root, "annotations", "subclass", name + "_subclass.txt")):
    p = line.split()
    if len(p) == 2 and p[1] in MAP:
        hits.append({"t": float(p[0]), "kind": MAP[p[1]], "vel": 0.8, "ghost": p[1] == "SDG"})

a = audio_io.load_audio(os.path.join(root, "audio", "full_mix", name + "_MIX.wav"))
dur = a.shape[1] / 44100
b, d = beats.track(a.mean(0), 44100)
grid = beats.build_grid(b, d, dur, onsets=np.array([h["t"] for h in hits if h["kind"] in ("kick", "snare")]))
print("tempo", grid.tempo, "bars", grid.bars)
tr, tern = notation.drum_track(hits, grid)
tracks = [tr]
notes = {}
for k in ["bass", "guitar", "piano", "vocals", "other"]:
    f = os.path.join(out, k + ".flac")
    t0 = time.time()
    notes[k] = pitched.transcribe(f, k) if os.path.exists(f) else []
    print(k, len(notes[k]), "notes", round(time.time() - t0, 1), "s")
ks = notation.estimate_key([n for v in notes.values() for n in v])
print("key", ks)
for k, v in notes.items():
    if len(v) >= 8:
        tracks.append(notation.pitched_track(k, v, grid, tern, ks))
tex = notation.build_alphatex(name, grid, tracks, "test")
open(os.path.join(out, "score.atex"), "w", encoding="utf-8").write(tex)
print(tex[:3000])
