from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from enum import Enum
from typing import Literal, Generator
import uvicorn
import subprocess
from ultralytics.utils import LOGGER
import logging
import sys
import re
import time
from pathlib import Path
import cv2
import numpy as np

LOGGER.setLevel(logging.ERROR)

app = FastAPI(title="🧠 Умный API: Калибровка, Оптимизация, Выбор модели")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def run_base_model_script(model_name: str):
    try:
        script = BASE_DIR / "base.py"
        result = subprocess.run(
            [sys.executable, str(script), "--model", model_name],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = (result.stdout or "").splitlines()
        filtered_lines = []
        in_block = False

        for line in output_lines:
            low = line.lower()
            if re.match(r"=+", line):
                in_block = not in_block
                filtered_lines.append(line)
            elif (
                in_block
                or "запуск" in low
                or "модель" in low
                or "model" in low
                or "selected" in low
                or "init" in low
                or "starting" in low
            ):
                filtered_lines.append(line)

        filtered_output = "\n".join(filtered_lines).strip()
        if not filtered_output:
            filtered_output = ""

        return {
            "status": "Штатный запуск завершён",
            "model": model_name,
            "output": f"⚠️ Алгоритм штатной работы остановлен пользователем\n🧩 Выбранная модель: {model_name}\n{filtered_output}",
        }

    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + "\n" + (e.stdout or "")
        return {"status": "Ошибка при выполнении base.py", "error": err, "model": model_name}
    except FileNotFoundError as e:
        return {"status": "Файл не найден", "error": f"{script} ({e})", "model": model_name}


def run_calibration_script(model="insightface", fps_threshold=10.0, cpu_threshold=50.0):
    try:
        script = BASE_DIR / "mecho_algo_V2.py"
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

        output_lines = result.stdout.splitlines()
        calibration_lines = []
        in_block = False
        for line in output_lines:
            if re.match(r"=+", line):
                in_block = not in_block
                calibration_lines.append(line)
            elif (
                in_block
                or "Активная модель" in line
                or "Калибровка завершена успешно" in line
                or "Калибровка завершена" in line
                or "Параметры пока не достигнуты" in line
                or "Перейдём к более лёгкой модели" in line
                or "подобрать модель не удалось" in line
                or "детекторов по заданным ограничениям не найдено" in line
                or "Используемая модель:" in line
                or "Средний FPS" in line
                or "Средняя загрузка CPU" in line
            ):
                calibration_lines.append(line)

        filtered_output = "\n".join(calibration_lines)

        success = bool(re.search(r"(Калибровка завершена успешно|Калибровка прошла успешно)", filtered_output))
        failure = bool(re.search(r"(подобрать модель не удалось|детекторов .* не найдено)", filtered_output))

        model_match = re.search(r"(Используемая модель|Модель удовлетворяющая условиям):\s*([^\n\r]+)", filtered_output)
        avg_fps_match = re.search(r"Средний FPS:\s*([0-9]+(?:\.[0-9]+)?)", filtered_output)
        avg_cpu_match = re.search(r"Средняя загрузка CPU:\s*([0-9]+(?:\.[0-9]+)?)%", filtered_output)

        payload = {
            "status": "Калибровка завершена" if success and not failure else (
                      "Подходящих детекторов не найдено" if failure else "Калибровка завершена"),
            "output": filtered_output
        }
        if model_match:
            payload["model"] = model_match.group(2).strip()
        if avg_fps_match:
            payload["avg_fps"] = float(avg_fps_match.group(1))
        if avg_cpu_match:
            payload["avg_cpu"] = float(avg_cpu_match.group(1))

        return payload

    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + "\n" + (e.stdout or "")
        return {"status": "Ошибка при выполнении скрипта", "error": err}
    except FileNotFoundError as e:
        return {"status": "Файл не найден", "error": f"{script} ({e})"}


def run_optimization_script():
    script = BASE_DIR / "algo_V4_fps.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False, 
        )

        stdout = (result.stdout or "")
        stderr = (result.stderr or "")

        noise_patterns = [
            re.compile(r"^\s*$"),
            re.compile(r"^utf-?8$", re.IGNORECASE),
            re.compile(r"^Applied providers:"),
            re.compile(r"^find model:\s", re.IGNORECASE),
            re.compile(r"^set det-size:", re.IGNORECASE),
            re.compile(r"^\s*\d+:\s+\d+x\d+\s"),  
            re.compile(r"^Speed:\s"),              
            re.compile(r".*CPUExecutionProvider.*"),
            re.compile(r"^I tensorflow/|^W tensorflow/|^E tensorflow/", re.IGNORECASE),
        ]

        def is_noise(line: str) -> bool:
            s = line.strip()
            return any(rx.search(s) for rx in noise_patterns)

        filtered_lines = [ln for ln in stdout.splitlines() if not is_noise(ln)]

        if result.returncode != 0:
            return {
                "status": "Оптимизация остановлена пользователем",
                "output": "⚠️ Алгоритм оптимизации остановлен пользователем",
            }

        if not filtered_lines:
            return {
                "status": "Оптимизация остановлена пользователем",
                "output": "⚠️ Алгоритм оптимизации остановлен пользователем",
            }

        return {
            "status": "Оптимизация завершена",
            "output": "\n".join(filtered_lines),
        }

    except subprocess.CalledProcessError:
        return {
            "status": "Оптимизация остановлена пользователем",
            "output": "⚠️ Алгоритм оптимизации остановлен пользователем",
        }
    except FileNotFoundError as e:
        return {"status": "Файл не найден", "error": f"{script} ({e})"}

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

selected_model = {"current": None}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "current_model": selected_model["current"]}
    )

@app.post("/select-model-ui")
def select_model_ui(request: Request, model_name: str = Form(...)):
    selected_model["current"] = model_name
    result = run_base_model_script(model_name)
    if isinstance(result, dict):
        result["selected_model"] = model_name
    else:
        result = {
            "status": "Штатный запуск завершён",
            "output": str(result),
            "selected_model": model_name,
        }
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]}
    )



@app.get("/run-mode-ui", response_class=HTMLResponse)
def run_mode_get(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "current_model": selected_model["current"], "view": "run"}
    )

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
            result = {
                "status": "Сначала выберите модель через /select-model-ui",
                "available_models": list(model_map.keys()),
            }
        else:
            model_name = model_map[selected]
            result = run_base_model_script(model_name)
    else:
        result = {"error": "Некорректный режим"}

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]}
    )

@app.post("/select-model-ui")
def select_model_ui(request: Request, model_name: str = Form(...)):
    selected_model["current"] = model_name
    result = run_base_model_script(model_name)
    result.update({"selected_model": model_name})
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "current_model": selected_model["current"]}
    )

def mjpeg_generator(cam_index: int = 0) -> Generator[bytes, None, None]:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera not available", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            fps = 1.0 / max(1e-3, (now - prev))
            prev = now
            hud = f"Model: {selected_model['current'] or '-'}  FPS: {fps:.1f}"
            cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (20, 220, 20), 2, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            chunk = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
            time.sleep(0.01)
    finally:
        cap.release()

@app.get("/stream.mjpg")
def stream_mjpg():
    return StreamingResponse(
        mjpeg_generator(cam_index=0),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

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
          <div class="pill"><a href="/optimizer/">Back to UI</a></div>
        </div>
        <img src="/optimizer/stream.mjpg" alt="MJPEG stream"/>
      </div>
    </body></html>
    """
    return HTMLResponse(html)

if __name__ == "__main__":
    uvicorn.run("main_V3:app", host="0.0.0.0", port=8010, reload=True)
