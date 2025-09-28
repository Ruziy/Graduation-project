# main_V3.py
import os
import time
import cv2
import numpy as np
import torch
import dlib
import mediapipe as mp

from pathlib import Path
from enum import Enum
from typing import Literal, Generator, Dict, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from ultralytics.utils import LOGGER
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
try:
    from face_detection import RetinaFace
except Exception:
    from retinaface import RetinaFace
import logging
import subprocess
import sys
import re

LOGGER.setLevel(logging.ERROR)

# ---------------- App & paths ----------------
app = FastAPI(title="🧠 Умный API: Калибровка, Оптимизация, Выбор модели")
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------- Globals ----------------
selected_model: Dict[str, Optional[str]] = {"current": None}  # 'yolov8'/'insightface'/... or None
device = "cuda" if torch.cuda.is_available() else "cpu"
insight_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # fallback на CPU
detectors_cache: Dict[str, object] = {}

# Чтобы RTSP не вис, помогаем FFMPEG-бэкенду
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;3000000|max_delay;0")

global_cap = None

# ---------------- Detectors ----------------
def get_detector(name: str):
    """Ленивая инициализация и кеш детекторов."""
    if name in detectors_cache:
        return detectors_cache[name]

    if name == "yolov8":
        det = YOLO(str(BASE_DIR / "weights/yolov8n-face.pt"))
    elif name == "mtcnn":
        det = MTCNN(keep_all=True, device=device)
    elif name == "insightface":
        det = FaceAnalysis(name="buffalo_l", providers=insight_providers)
        det.prepare(ctx_id=(0 if device == "cuda" else -1))
    elif name == "retinaface":
        det = RetinaFace()
    elif name == "mediapipe":
        det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    elif name == "dlib":
        det = dlib.get_frontal_face_detector()
    elif name == "haarcascade":
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    elif name == "ssd":
        prototxt = str(BASE_DIR / "weights/deploy.prototxt.txt")
        caffemodel = str(BASE_DIR / "weights/res10_300x300_ssd_iter_140000.caffemodel")
        det = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    else:
        det = None

    detectors_cache[name] = det
    return det

def detect_faces(frame, model, detector=None, net=None, conf=0.5):
    """Унифицированная детекция для превью."""
    h, w = frame.shape[:2]

    if model == "yolov8":
        results = detector(frame)[0]
        return [b for b in results.boxes.xyxy.cpu().numpy()]

    elif model == "mtcnn":
        boxes, _ = detector.detect(frame)
        return [] if boxes is None else boxes

    elif model == "insightface":
        faces = detector.get(frame)
        return [f.bbox for f in faces]

    elif model == "retinaface":
        faces = detector(frame)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b, lm, score in faces if score >= conf]

    elif model == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        boxes = []
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1, y1 = int(bb.xmin * w), int(bb.ymin * h)
                ww, hh = int(bb.width * w), int(bb.height * h)
                boxes.append([x1, y1, x1 + ww, y1 + hh])
        return boxes

    elif model == "dlib":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        return [[f.left(), f.top(), f.right(), f.bottom()] for f in faces]

    elif model == "haarcascade":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [[x, y, x + w, y + h] for (x, y, w, h) in faces]

    elif model == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        out = []
        for i in range(detections.shape[2]):
            confd = detections[0, 0, i, 2]
            if confd >= conf:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                out.append(box.astype(int).tolist())
        return out

    return []

# ---------------- App lifecycle ----------------
@app.on_event("startup")
def _open_cam():
    global global_cap
    src = os.getenv("CAM_URL", "0")
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    global_cap = cap

@app.on_event("shutdown")
def _close_cam():
    if global_cap:
        global_cap.release()

# ---------------- Subprocess helpers ----------------
def run_base_model_script(model_name: str):
    try:
        script = BASE_DIR / "base.py"
        result = subprocess.run(
            [sys.executable, str(script), "--model", model_name],  # base.py сам headless/ограничен по времени
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = result.stdout.splitlines()
        filtered_lines, in_block = [], False
        for line in output_lines:
            if re.match(r"=+", line):  # просто «красиво» режем лог, если есть разделители
                in_block = not in_block
                filtered_lines.append(line)
            elif in_block or "запуск" in line.lower() or "модель" in line.lower():
                filtered_lines.append(line)
        return {"status": "Скрипт base.py выполнен", "output": "\n".join(filtered_lines) or result.stdout}
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + "\n" + (e.stdout or "")
        return {"status": "Ошибка при выполнении base.py", "error": err}
    except FileNotFoundError as e:
        return {"status": "Файл не найден", "error": f"{script} ({e})"}

def run_calibration_script(model="insightface", fps_threshold=10.0, cpu_threshold=50.0):
    script = BASE_DIR / "mecho_algo_V2.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script),
             "--model", model,
             "--fps_threshold", str(fps_threshold),
             "--cpu_threshold", str(cpu_threshold)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        out = result.stdout
        lines = out.splitlines()
        cal_lines, in_block = [], False
        for line in lines:
            if re.match(r"=+", line):
                in_block = not in_block
                cal_lines.append(line)
            elif (in_block or
                  "Калибровка" in line or
                  "Средний FPS" in line or
                  "Средняя загрузка CPU" in line or
                  "нагрузка" in line or
                  "Калибровка прошла успешно" in line or
                  line.startswith("[КАЛИБРОВКА]") or
                  line.startswith("[CALIBRATION]")):
                cal_lines.append(line)
        return {"status": "Калибровка завершена", "output": "\n".join(cal_lines) or out}
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + "\n" + (e.stdout or "")
        return {"status": "Ошибка при выполнении скрипта", "error": err}
    except FileNotFoundError as e:
        return {"status": "Файл не найден", "error": f"{script} ({e})"}

def run_optimization_script():
    try:
        script = BASE_DIR / "algo_V4_fps.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        out = result.stdout
        lines = out.splitlines()
        filtered, in_block = [], False
        for line in lines:
            if "Applied providers:" in line:
                continue
            if re.match(r"=+", line):
                in_block = not in_block
                filtered.append(line)
            elif in_block or any(s in line.lower() for s in ("оптимизация", "fps", "cpu", "статус")):
                filtered.append(line)
        return {"status": "Оптимизация завершена", "output": "\n".join(filtered) or out}
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + "\n" + (e.stdout or "")
        return {"status": "Ошибка при выполнении скрипта оптимизации", "error": err}
    except FileNotFoundError as e:
        return {"status": "Файл не найден", "error": f"{script} ({e})"}

# ---------------- UI routes ----------------
class ModeEnum(int, Enum):
    calibrate = 1
    optimize = 2
    select_model = 3

class ModeRequest(BaseModel):
    mode: ModeEnum

class ModelChoice(BaseModel):
    selected_model_name: Literal[
        "yolov8", "insightface", "dlib", "mtcnn",
        "retinaface", "mediapipe", "haarcascade", "ssd"
    ]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",
        {"request": request, "current_model": selected_model["current"]})

@app.get("/select-model-ui", response_class=HTMLResponse)
def select_model_get(request: Request):
    return templates.TemplateResponse("index.html",
        {"request": request, "current_model": selected_model["current"], "view": "select"})

@app.get("/run-mode-ui", response_class=HTMLResponse)
def run_mode_get(request: Request):
    return templates.TemplateResponse("index.html",
        {"request": request, "current_model": selected_model["current"], "view": "run"})

@app.post("/run-mode-ui")
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
        selected = selected_model["current"]
        if selected is None or selected not in model_map:
            result = {"status": "Сначала выберите модель через /select-model-ui",
                      "available_models": list(model_map.keys())}
        else:
            result = run_base_model_script(model_map[selected])
    else:
        result = {"error": "Некорректный режим"}

    return templates.TemplateResponse("index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]})

@app.post("/select-model-ui")
def select_model_ui(request: Request, model_name: str = Form(...)):
    selected_model["current"] = model_name
    # Вызов скрипта оставляем (он headless и быстро завершается)
    result = run_base_model_script(model_name)
    result.update({"selected_model": model_name})
    return templates.TemplateResponse("index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]})

# ---------------- MJPEG stream ----------------
def mjpeg_generator() -> Generator[bytes, None, None]:
    cap = global_cap
    if cap is None or not cap.isOpened():
        # Заглушка
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.time()
        fps = 1.0 / max(1e-3, (now - prev))
        prev = now

        # Если выбрана модель — рисуем боксы
        model = selected_model["current"]
        if model:
            det = get_detector(model)
            if model == "ssd":
                boxes = detect_faces(frame, model, net=det)
            else:
                boxes = detect_faces(frame, model, detector=det)
            for b in boxes or []:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

        hud = f"Model: {model or '—'} | FPS: {fps:.1f}"
        cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (20, 220, 20), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.005)

@app.get("/stream.mjpg")
def stream_mjpg():
    return StreamingResponse(mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/preview", response_class=HTMLResponse)
def preview_page(request: Request):
    html = f"""
<!doctype html>
<html><head>
  <meta charset="utf-8"/>
  <title>Optimizer Preview</title>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;padding:16px;background:#0b0b0b;color:#eee}}
    .wrap{{max-width:980px;margin:0 auto}}
    img{{max-width:100%;border:1px solid #333;border-radius:8px;display:block}}
    .row{{display:flex;gap:12px;align-items:center;margin-bottom:12px}}
    .pill{{padding:6px 10px;border:1px solid #333;border-radius:999px;background:#111}}
    a{{color:#9ad}}
  </style>
</head><body>
  <div class="wrap">
    <div class="row">
      <div class="pill">model: <b>{selected_model['current'] or "—"}</b></div>
      <div class="pill"><a href="/">Back to UI</a></div>
    </div>
    <img src="/stream.mjpg" alt="MJPEG stream"/>
  </div>
</body></html>
"""
    return HTMLResponse(html)
