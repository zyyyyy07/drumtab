"""用本机 Edge 无头模式打开页面，截图 + 收集控制台报错 + 检查播放同步。"""
import sys, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright

here = Path(__file__).parent
shots = here / "shots"
shots.mkdir(exist_ok=True)
pid = sys.argv[1] if len(sys.argv) > 1 else "0123456789ab"
steps = sys.argv[2].split(",") if len(sys.argv) > 2 else ["load"]

with sync_playwright() as p:
    br = p.chromium.launch(channel="msedge", headless=True, args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
    pg = br.new_page(viewport={"width": 1440, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    pg.goto("http://127.0.0.1:8765/")
    pg.wait_for_selector("#list li")
    pg.screenshot(path=str(shots / "00_home.png"))
    pg.evaluate(f"document.querySelectorAll('#list li').forEach(li => 0)")
    pg.click(f"#list li:has-text('{sys.argv[3] if len(sys.argv) > 3 else 'Rock'}')")
    pg.wait_for_selector("#viewer:not([hidden])", timeout=15000)
    pg.wait_for_function("document.querySelector('#loading').hidden && document.querySelectorAll('#at svg').length > 0", timeout=30000)
    time.sleep(1.0)
    pg.screenshot(path=str(shots / "01_drums.png"))
    info = pg.evaluate("""() => ({tracks: __app.score.tracks.map(t => t.name), bars: __app.score.masterBars.length,
        svgs: document.querySelectorAll('#at svg').length, dur: document.querySelector('#audio').duration,
        sync: __app.score.masterBars.filter(b => b.syncPoints && b.syncPoints.length).length})""")
    print("info", json.dumps(info, ensure_ascii=False))
    if "all" in steps:
        pg.click("#tracks button:has-text('总谱')")
        pg.wait_for_function("document.querySelector('#loading').hidden", timeout=30000)
        time.sleep(1.2)
        pg.screenshot(path=str(shots / "02_all.png"))
        for i, nm in enumerate(info["tracks"][1:]):
            pg.click(f"#tracks button:has-text('{nm[:2]}')")
            pg.wait_for_function("document.querySelector('#loading').hidden", timeout=30000)
            time.sleep(1.0)
            pg.screenshot(path=str(shots / f"03_track{i}.png"))
        pg.click("#tracks button:has-text('鼓')")
        time.sleep(1.0)
    if "play" in steps:
        pg.click("#play")
        res = []
        for i in range(8):
            time.sleep(1.0)
            res.append(pg.evaluate("""() => { const a = document.querySelector('#audio'); const c = document.querySelector('.at-cursor-bar'); const b = document.querySelector('.at-cursor-beat');
                const r = c ? c.getBoundingClientRect() : null; const rb = b ? b.getBoundingClientRect() : null;
                return {t: +a.currentTime.toFixed(2), paused: a.paused, bar: document.querySelector('#barno').textContent,
                        cur: r ? [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] : null,
                        beat: rb ? [Math.round(rb.x), Math.round(rb.y)] : null, scroll: document.querySelector('#scroll').scrollTop}; }"""))
        for r in res:
            print(json.dumps(r, ensure_ascii=False))
        pg.screenshot(path=str(shots / "04_playing.png"))
        pg.click("#play")
    if "print" in steps:
        with pg.expect_popup() as pop:
            pg.click("#btn-print")
        pp = pop.value
        pp.wait_for_load_state()
        time.sleep(6)
        print("popup svgs", pp.evaluate("document.querySelectorAll('svg').length"), "title", pp.title())
        pp.pdf(path=str(shots / "print.pdf"), format="A4")
        pp.set_viewport_size({"width": 900, "height": 1270})
        pp.screenshot(path=str(shots / "06_print.png"))
        import re
        print("pdf pages", len(re.findall(rb"/Type\s*/Page[^s]", (shots / "print.pdf").read_bytes())))
        pp.close()
    if "export" in steps:
        for x in ("gp", "midi", "atex"):
            pg.click("#btn-export")
            with pg.expect_download() as dl:
                pg.click(f"#menu-export button[data-x='{x}']")
            d = dl.value
            f = shots / ("export_" + d.suggested_filename)
            d.save_as(str(f))
            print("download", d.suggested_filename, f.stat().st_size)
    if "fix" in steps:
        pg.click("#btn-fix")
        time.sleep(0.3)
        pg.screenshot(path=str(shots / "05_fix.png"))
    print("LOGS", len(logs))
    for l in logs[:25]:
        print(l[:400])
    br.close()
