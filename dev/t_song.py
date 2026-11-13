"""诊断某个已处理项目：拍点规律性、小节划分、鼓谱记法统计，并抽看小节。
用法：python dev/t_song.py <项目id> [起始小节 结束小节]
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app import beats, pipeline

d = Path(__file__).resolve().parent.parent / "projects" / sys.argv[1]
a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
b, db = np.array(a["beats"]), np.array(a["downbeats"])
iv = np.diff(b)
med = float(np.median(iv))
bad = [(round(float(b[i]), 1), round(float(iv[i]), 2)) for i in range(len(iv)) if abs(iv[i] - med) > 0.15 * med]
print(f"原始拍点 {len(b)} 个, 中位间隔 {med:.3f}s ({60/med:.1f} BPM), 不规则间隔 {len(bad)} 处: {bad[:12]}")

meta = json.loads((d / "project.json").read_text(encoding="utf-8"))
info = pipeline.render(d, meta["title"], meta.get("options"))
print({k: v for k, v in info.items() if k not in ("bar_times", "options", "tracks")})
bt = np.array(info["bar_times"])
bl = np.diff(bt)
print("小节时长: 中位 %.2fs, 异常小节(偏差>15%%): %s" % (np.median(bl), [(i + 1, round(float(x), 2)) for i, x in enumerate(bl) if abs(x - np.median(bl)) > 0.15 * np.median(bl)][:30]))
tex = (d / "score.atex").read_text(encoding="utf-8")
drum = tex.split("\\track")[1]
body = drum[drum.rfind("\\articulation"):]
body = body[body.find("\n") + 1:]
bars = body.split("|")
print("鼓谱小节数", len(bars), "| 拍号变化", re.findall(r"\\ts \((\d+) 4\)", drum))
print("三连音", drum.count("tu 3"), "六连音", drum.count("tu 6"), "32分", drum.count(".32"), "附点", drum.count("{d}"), "16分", drum.count(".16"), "8分", drum.count(".8"), "鬼音", drum.count("{g}"))
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1, 0)
for i in range(lo - 1, min(hi, len(bars))):
    print(f"{i+1:3d} [{bt[i]:6.1f}s] {bars[i].strip()}")
