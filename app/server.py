"""本地网页服务：上传音频 -> 排队处理 -> 提供乐谱和各音轨音频。"""
from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = ROOT / "projects"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="鼓谱")
_jobs: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()
_ID = re.compile(r"^[0-9a-f]{12}$")
_MEDIA = {"mix", "nodrums", "drums", "bass", "guitar", "piano", "vocals", "other"}


def _dir(pid: str) -> Path:
    if not _ID.match(pid):
        raise HTTPException(404, "项目不存在")
    d = DATA / pid
    if not (d / "project.json").exists():
        raise HTTPException(404, "项目不存在")
    return d


def _read(d: Path) -> dict:
    return json.loads((d / "project.json").read_text(encoding="utf-8"))


def _write(d: Path, meta: dict) -> None:
    with _lock:
        tmp = d / "project.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "project.json")


def _update(d: Path, **kw) -> dict:
    meta = _read(d)
    meta.update(kw)
    _write(d, meta)
    return meta


def _full(d: Path) -> dict:
    meta = _read(d)
    sj = d / "score.json"
    if meta.get("status") == "done" and sj.exists():
        meta["score"] = json.loads(sj.read_text(encoding="utf-8"))
        meta["media"] = sorted(m for m in _MEDIA if (d / f"{m}.flac").exists())
    return meta


def _worker() -> None:
    while True:
        pid = _jobs.get()
        d = DATA / pid
        try:
            if not (d / "project.json").exists():
                continue
            meta = _read(d)
            _update(d, status="processing", progress=0.0, message="准备中", error="")
            last = [0.0]

            def progress(p: float, msg: str) -> None:
                if time.time() - last[0] > 0.3 or p >= 0.97:
                    last[0] = time.time()
                    _update(d, progress=round(float(p), 3), message=msg)

            src = next(d.glob("source.*"))
            pipeline.analyze(src, d, progress)
            pipeline.render(d, meta["title"], meta.get("options"))
            _update(d, status="done", progress=1.0, message="完成")
        except Exception as e:
            traceback.print_exc()
            try:
                _update(d, status="error", error=str(e)[:500], message="处理失败")
            except Exception:
                pass


@app.on_event("startup")
def _startup() -> None:
    # 上次没跑完的任务重新排队
    for d in sorted(DATA.iterdir()):
        try:
            if (d / "project.json").exists() and _read(d).get("status") in ("queued", "processing"):
                _update(d, status="queued", progress=0.0, message="排队中")
                _jobs.put(d.name)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


@app.get("/api/projects")
def list_projects():
    out = []
    for d in DATA.iterdir():
        if (d / "project.json").exists():
            try:
                out.append(_read(d))
            except Exception:
                pass
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


@app.post("/api/projects")
async def create_project(file: UploadFile = File(...)):
    name = Path(file.filename or "audio").name
    if name and all(ord(c) < 256 for c in name) and any(ord(c) > 127 for c in name):
        # 个别客户端用本地编码发文件名，尝试还原中文
        for enc in ("utf-8", "gbk"):
            try:
                name = name.encode("latin-1").decode(enc)
                break
            except UnicodeError:
                pass
    ext = Path(name).suffix.lower()
    if ext not in {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".mov", ".mkv", ".webm", ".aiff", ".aif", ".ape"}:
        raise HTTPException(400, "不支持的文件格式")
    pid = uuid.uuid4().hex[:12]
    d = DATA / pid
    d.mkdir(parents=True)
    with open(d / f"source{ext}", "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    meta = {
        "id": pid, "title": Path(name).stem, "filename": name, "created": time.time(),
        "status": "queued", "progress": 0.0, "message": "排队中", "error": "", "options": dict(pipeline.DEFAULT_OPTIONS),
    }
    _write(d, meta)
    _jobs.put(pid)
    return meta


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _full(_dir(pid))


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    d = _dir(pid)
    if _read(d).get("status") == "processing":
        raise HTTPException(409, "正在处理中，稍后再删")
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/projects/{pid}/render")
def rerender(pid: str, body: dict = Body(...)):
    d = _dir(pid)
    meta = _read(d)
    if meta.get("status") != "done":
        raise HTTPException(409, "尚未处理完成")
    opts = dict(pipeline.DEFAULT_OPTIONS)
    opts.update(meta.get("options") or {})
    for k, v in (body.get("options") or {}).items():
        if k in pipeline.DEFAULT_OPTIONS:
            opts[k] = type(pipeline.DEFAULT_OPTIONS[k])(v)
    title = str(body.get("title") or meta["title"])[:120]
    pipeline.render(d, title, opts)
    _update(d, options=opts, title=title)
    return _full(d)


@app.post("/api/projects/{pid}/retry")
def retry(pid: str):
    d = _dir(pid)
    if _read(d).get("status") != "error":
        raise HTTPException(409, "只有失败的任务可以重试")
    _update(d, status="queued", progress=0.0, message="排队中", error="")
    _jobs.put(pid)
    return _read(d)


@app.get("/api/projects/{pid}/score", response_class=PlainTextResponse)
def get_score(pid: str):
    f = _dir(pid) / "score.atex"
    if not f.exists():
        raise HTTPException(404, "乐谱尚未生成")
    return PlainTextResponse(f.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/media/{pid}/{name}.flac")
def media(pid: str, name: str):
    if name not in _MEDIA:
        raise HTTPException(404)
    f = _dir(pid) / f"{name}.flac"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/flac")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=str(WEB)), name="web")
