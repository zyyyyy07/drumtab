"""端到端：浏览器里上传中文名 MP3 -> 等处理 -> 自动打开 -> 减速播放检查同步 -> 切换音源 -> 校正重出谱。"""
import sys, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright

here = Path(__file__).parent
shots = here / "shots"
with sync_playwright() as p:
    br = p.chromium.launch(channel="msedge", headless=True, args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
    pg = br.new_page(viewport={"width": 1440, "height": 900})
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    pg.on("dialog", lambda d: d.accept())
    pg.goto("http://127.0.0.1:8765/")
    pg.wait_for_selector("#list li")
    # 删掉文件名乱码的那条
    n0 = pg.locator("#list li").count()
    bad = pg.locator("#list li", has_text="²âÊÔ")
    if bad.count():
        bad.first.hover(); bad.first.locator(".del").click(); time.sleep(1)
    print("list", n0, "->", pg.locator("#list li").count())
    t0 = time.time()
    if pg.locator("#list li", has_text="长测试曲").count():
        pg.locator("#list li", has_text="长测试曲").first.click()   # 已经传过就直接打开
    else:
        pg.set_input_files("#file", str(here / "长测试曲.mp3"))
        pg.wait_for_selector("#working:not([hidden])", timeout=20000)
        time.sleep(8)
        pg.screenshot(path=str(shots / "10_working.png"))
        print("working msg:", pg.inner_text("#w-msg"))
    pg.wait_for_selector("#viewer:not([hidden])", timeout=300000)
    pg.wait_for_function("document.querySelector('#loading').hidden && document.querySelectorAll('#at svg').length > 0", timeout=60000)
    print("processed+opened in", round(time.time() - t0), "s; title:", pg.inner_text("#title"), "| chips:", pg.inner_text("#chips").replace("\n", " "))
    print("tracks:", pg.inner_text("#tracks").replace("\n", " "))
    time.sleep(1)

    def state():
        return pg.evaluate("""() => { const a = document.querySelector('#audio'); const c = document.querySelector('.at-cursor-bar');
            const r = c ? c.getBoundingClientRect() : null;
            return {t: +a.currentTime.toFixed(2), rate: a.playbackRate, src: a.src.split('/').pop(), bar: document.querySelector('#barno').textContent,
                    cur: r ? [Math.round(r.x), Math.round(r.y + document.querySelector('#scroll').scrollTop), Math.round(r.width)] : null,
                    scroll: Math.round(document.querySelector('#scroll').scrollTop)}; }""")

    # 正常速度：记录 t≈60s 附近光标位置
    pg.evaluate("document.querySelector('#audio').currentTime = 60")
    pg.click("#play"); time.sleep(2.5); pg.click("#play"); time.sleep(0.3)
    s1 = state(); print("100%:", json.dumps(s1, ensure_ascii=False))
    # 50% 速度回到同一位置
    pg.select_option("#speed", "0.5")
    pg.evaluate("document.querySelector('#audio').currentTime = 60")
    pg.click("#play"); time.sleep(5.0); pg.click("#play"); time.sleep(0.3)
    s2 = state(); print(" 50%:", json.dumps(s2, ensure_ascii=False))
    # 切到去鼓伴奏，位置应保持
    pg.select_option("#listen", "nodrums"); time.sleep(1.5)
    s3 = state(); print("去鼓:", json.dumps(s3, ensure_ascii=False))
    # 50% 速度下点击谱面跳转：点第 1 行某小节，音频应跳到对应时间（而不是一半/两倍）
    pg.evaluate("document.querySelector('#scroll').scrollTop = 0"); time.sleep(0.4)
    box = pg.evaluate("""() => { const b = __app.api.renderer.boundsLookup.findMasterBarByIndex(5).visualBounds; const r = document.querySelector('#at').getBoundingClientRect(); return [r.x + b.x + b.w * 0.5, r.y + b.y + b.h * 0.5]; }""")
    pg.mouse.click(box[0], box[1]); time.sleep(0.8)
    s5 = state(); print("点第6小节(50%):", json.dumps(s5, ensure_ascii=False), "期望时间≈", pg.evaluate("__app.cur.score.bar_times.slice(5,7)"))
    pg.select_option("#speed", "1")
    # 远处播放：应自动滚动
    pg.evaluate("document.querySelector('#audio').currentTime = 200")
    pg.click("#play"); time.sleep(3);
    s4 = state(); print("t=200:", json.dumps(s4, ensure_ascii=False))
    pg.screenshot(path=str(shots / "11_scrolled.png"))
    pg.click("#play")
    # 校正：小节线推后一拍
    pg.click("#btn-fix"); pg.click("[data-shift='1']"); pg.click("#fix-apply")
    time.sleep(1); pg.wait_for_function("document.querySelector('#loading').hidden", timeout=60000); time.sleep(1.5)
    print("after fix: shift =", pg.evaluate("__app.cur.score.options.beat_shift"), "bars =", pg.evaluate("__app.cur.score.bars"), "score bars =", pg.evaluate("__app.score.masterBars.length"))
    pg.click("#btn-fix"); pg.click("[data-shift='-1']"); pg.click("#fix-apply")
    time.sleep(1); pg.wait_for_function("document.querySelector('#loading').hidden", timeout=60000)
    pg.click("#legend summary"); time.sleep(2)
    pg.evaluate("document.querySelector('#scroll').scrollTop = 0"); time.sleep(0.5)
    pg.screenshot(path=str(shots / "12_legend.png"))
    print("LOGS", len(logs)); [print(l[:300]) for l in logs[:10]]
    br.close()
