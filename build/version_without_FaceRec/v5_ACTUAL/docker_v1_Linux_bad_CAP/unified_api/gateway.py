# unified_api/gateway.py
from __future__ import annotations

import os
import cv2
import time
import threading
import subprocess
import sys
import re
from pathlib import Path
from typing import Generator, Optional, List

import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ------------------------- Paths & ENV --------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OPT_DIR = BASE_DIR / "optimizer_part"
TEMPLATES_DIR = OPT_DIR / "templates"  # твой index.html лежит тут
STATIC_DIR = OPT_DIR / "static"

# Для RTSP через TCP и коротких таймаутов (headless)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;5000000"
)

# ------------------------- FastAPI app --------------------------------------
app = FastAPI(title="🧠 Unified API (Gateway)")

# Статика/шаблоны твоего UI
if STATIC_DIR.exists():
    app.mount("/optimizer/static", StaticFiles(directory=str(STATIC_DIR)), name="optimizer-static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ------------------------- ВСПОМОГАТЕЛЬНЫЕ СКРИПТЫ -------------------------
# Эти функции запускают твои скрипты из optimizer_part/
def _run(cmd: List[str]) -> dict:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(OPT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return {"ok": True, "out": result.stdout, "err": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "out": e.stdout or "", "err": e.stderr or ""}

def run_base_model_script(model_name: str):
    r = _run([sys.executable, "base.py", "--model", model_name])
    if r["ok"]:
        out = r["out"].splitlines()
        filtered, in_block = [], False
        for line in out:
            if re.match(r"=+", line):
                in_block = not in_block
                filtered.append(line)
            elif in_block or "запуск" in line.lower() or "модель" in line.lower():
                filtered.append(line)
        return {"status": "Скрипт base.py выполнен", "output": "\n".join(filtered)}
    return {"status": "Ошибка при выполнении base.py", "error": r["err"] + ("\n" + r["out"] if r["out"] else "")}

def run_calibration_script(model="insightface", fps_threshold=10.0, cpu_threshold=50.0):
    r = _run([
        sys.executable, "mecho_algo_V2.py",
        "--model", model,
        "--fps_threshold", str(fps_threshold),
        "--cpu_threshold", str(cpu_threshold),
    ])
    if r["ok"]:
        out = r["out"].splitlines()
        keep, in_block = [], False
        for line in out:
            if re.match(r"=+", line):
                in_block = not in_block; keep.append(line)
            elif (in_block or
                  "Калибровка" in line or "Средний FPS" in line or "Средняя загрузка CPU" in line or
                  "нагрузка" in line or "Калибровка прошла успешно" in line or
                  line.startswith("[КАЛИБРОВКА]") or line.startswith("[CALIBRATION]")):
                keep.append(line)
        return {"status": "Калибровка завершена", "output": "\n".join(keep)}
    return {"status": "Ошибка при выполнении скрипта", "error": r["err"] + ("\n" + r["out"] if r["out"] else "")}

def run_optimization_script():
    r = _run([sys.executable, "algo_V4_fps.py"])
    if r["ok"]:
        out = r["out"].splitlines()
        keep, in_block = [], False
        for line in out:
            if "Applied providers:" in line:
                continue
            if re.match(r"=+", line):
                in_block = not in_block; keep.append(line)
            elif in_block or any(k in line.lower() for k in ("оптимизация", "fps", "cpu", "статус")):
                keep.append(line)
        return {"status": "Оптимизация завершена", "output": "\n".join(keep)}
    return {"status": "Ошибка при выполнении скрипта оптимизации", "error": r["err"] + ("\n" + r["out"] if r["out"] else "")}

# ------------------------- СОСТОЯНИЕ UI ------------------------------------
from enum import Enum
from typing import Literal

class ModeEnum(int, Enum):
    calibrate = 1
    optimize = 2
    select_model = 3

selected_model = {"current": None}

# ------------------------- ТВОЙ UI (главная в /optimizer/) ------------------
@app.get("/optimizer/", response_class=HTMLResponse)
def optimizer_home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "current_model": selected_model["current"]}
    )

@app.get("/optimizer/select-model-ui", response_class=HTMLResponse)
def select_model_get(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "current_model": selected_model["current"], "view": "select"}
    )

@app.get("/optimizer/run-mode-ui", response_class=HTMLResponse)
def run_mode_get(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "current_model": selected_model["current"], "view": "run"}
    )

@app.post("/optimizer/run-mode-ui")
def run_mode_ui(
    request: Request,
    mode: int = Form(...),
    fps_threshold: float = Form(10.0),
    cpu_threshold: float = Form(50.0),
):
    if mode == 1:
        model = "insightface"
        result = run_calibration_script(model, fps_threshold, cpu_threshold)
    elif mode == 2:
        result = run_optimization_script()
    elif mode == 3:
        model_map = {
            "yolov8": "yolov8",
            "dlib": "dlib",
            "mtcnn": "mtcnn",
            "insightface": "insightface",
            "retinaFace": "retinaface",
            "mediapipe": "mediapipe",
            "haarcascade": "haarcascade",
            "ssd": "ssd",
        }
        sel = selected_model["current"]
        if sel is None or sel not in model_map:
            result = {"status": "Сначала выберите модель через /optimizer/select-model-ui",
                      "available_models": list(model_map.keys())}
        else:
            result = run_base_model_script(model_map[sel])
    else:
        result = {"error": "Некорректный режим"}

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]}
    )

@app.post("/optimizer/select-model-ui")
def select_model_ui(request: Request, model_name: str = Form(...)):
    selected_model["current"] = model_name
    result = run_base_model_script(model_name)
    result.update({"selected_model": model_name})
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]}
    )

# маленький 204, чтобы логику браузера не засоряло
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# ------------------------- MJPEG ПРЕВЬЮ (НЕ ломает UI) ---------------------
_CAM_URL = os.getenv("CAM_URL", "0")
_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()
_bad_reads = 0

def _open_capture() -> Optional[cv2.VideoCapture]:
    global _cap
    try:
        if _cap is not None:
            _cap.release()
    except Exception:
        pass

    if _CAM_URL.isdigit():
        _cap = cv2.VideoCapture(int(_CAM_URL))
    else:
        _cap = cv2.VideoCapture(_CAM_URL, cv2.CAP_FFMPEG)

    try:
        _cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    return _cap

def _get_capture(retries: int = 12, delay: float = 0.5) -> Optional[cv2.VideoCapture]:
    with _cap_lock:
        cap = _open_capture()
        for _ in range(retries):
            if cap and cap.isOpened():
                return cap
            time.sleep(delay)
            cap = _open_capture()
    return None

def _jpeg_chunk(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

def _placeholder(text: str) -> bytes:
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return _jpeg_chunk(canvas)

def mjpeg_generator() -> Generator[bytes, None, None]:
    global _bad_reads
    cap = _get_capture()
    if not cap or not cap.isOpened():
        start = time.time()
        while time.time() - start < 8:
            yield _placeholder("Waiting for RTSP...")
            time.sleep(1.0)
            cap = _get_capture(retries=1, delay=0.5)
            if cap and cap.isOpened():
                break
        if not cap or not cap.isOpened():
            yield _placeholder("Camera not available")
            return

    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            _bad_reads += 1
            if _bad_reads >= 10:
                _bad_reads = 0
                cap = _get_capture(retries=5, delay=0.5)
                if not cap or not cap.isOpened():
                    yield _placeholder("Reconnecting...")
                    time.sleep(1.0)
                    continue
            time.sleep(0.02)
            continue

        _bad_reads = 0
        now = time.time()
        fps = 1.0 / max(1e-3, (now - last))
        last = now
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
        yield _jpeg_chunk(frame)

@app.get("/optimizer/stream.mjpg")
def stream_mjpg():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/optimizer/preview", response_class=HTMLResponse)
def preview_page(_: Request):
    html = f"""
    <!doctype html><meta charset="utf-8"/>
    <title>Optimizer Preview</title>
    <style>
      body{{margin:0;background:#0b0b0b;color:#eee;font:14px system-ui}}
      .wrap{{max-width:980px;margin:0 auto;padding:12px}}
      img{{width:100%;border:1px solid #333;border-radius:8px}}
      .pill{{display:inline-block;padding:6px 10px;margin:6px 6px 10px 0;border:1px solid #333;border-radius:9999px;background:#111}}
      a{{color:#9ad}}
    </style>
    <div class="wrap">
      <div>
        <span class="pill">CAM_URL: <b>{_CAM_URL}</b></span>
        <a class="pill" href="/optimizer/cam-status">status</a>
        <a class="pill" href="/optimizer/">back to UI</a>
      </div>
      <img src="/optimizer/stream.mjpg" alt="MJPEG stream"/>
    </div>
    """
    return HTMLResponse(html)

@app.get("/optimizer/cam-status")
def cam_status():
    with _cap_lock:
        opened = bool(_cap and _cap.isOpened())
    read_ok = False
    shape = None
    if opened:
        read_ok, frame = _cap.read()
        if read_ok and frame is not None:
            shape = list(frame.shape)
    return JSONResponse({"cam_url": _CAM_URL, "opened": opened, "read_ok": bool(read_ok), "shape": shape})
