from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from enum import Enum
from typing import Literal
from fastapi.templating import Jinja2Templates
import uvicorn
import subprocess
from ultralytics.utils import LOGGER
import logging

# Отключить все логи ниже ERROR
LOGGER.setLevel(logging.ERROR)

app = FastAPI(title="🧠 Умный API: Калибровка, Оптимизация, Выбор модели")

# Подключаем шаблоны и статику
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

import re
def run_base_model_script(model_name: str):
    try:
        result = subprocess.run(
            ["python", "base.py", "--model", model_name],
            capture_output=True,
            text=True,
            check=True
        )

        # Отбор вывода при необходимости
        output_lines = result.stdout.splitlines()
        filtered_lines = []
        in_block = False
        for line in output_lines:
            if re.match(r"=+", line):
                in_block = not in_block
                filtered_lines.append(line)
            elif in_block or "запуск" in line.lower() or "модель" in line.lower():
                filtered_lines.append(line)

        filtered_output = "\n".join(filtered_lines)
        return {"status": "Скрипт base.py выполнен", "output": filtered_output}

    except subprocess.CalledProcessError as e:
        return {"status": "Ошибка при выполнении base.py", "error": e.stderr}


def run_calibration_script(model="yolov8", fps_threshold=10.0, cpu_threshold=50.0):
    try:
        result = subprocess.run(
            [
                "python", "mecho_algo_V2.py",
                "--model", model,
                "--fps_threshold", str(fps_threshold),
                "--cpu_threshold", str(cpu_threshold)
            ],
            capture_output=True,
            text=True,
            check=True
        )

        # Фильтрация только строк с калибровкой
        output_lines = result.stdout.splitlines()
        calibration_lines = []
        in_block = False
        for line in output_lines:
            if re.match(r"=+", line):
                in_block = not in_block
                calibration_lines.append(line)
            elif in_block or "Калибровка завершена" in line or "Средний FPS" in line or "Средняя загрузка CPU" in line or "нагрузка" in line or "Калибровка прошла успешно" in line:
                calibration_lines.append(line)

        filtered_output = "\n".join(calibration_lines)
        return {"status": "Калибровка завершена", "output": filtered_output}

    except subprocess.CalledProcessError as e:
        return {"status": "Ошибка при выполнении скрипта", "error": e.stderr}

def run_optimization_script():
    try:
        result = subprocess.run(
            ["python", "algo_V4_fps.py"],
            capture_output=True,
            text=True,
            check=True
        )

        # Фильтрация полезных строк (если нужно — добавь свою логику)
        output_lines = result.stdout.splitlines()
        filtered_lines = []
        in_block = False
        for line in output_lines:
            if re.match(r"=+", line):
                in_block = not in_block
                filtered_lines.append(line)
            elif in_block or "оптимизация" in line.lower() or "fps" in line.lower() or "cpu" in line.lower():
                filtered_lines.append(line)

        filtered_output = "\n".join(filtered_lines)
        return {"status": "Оптимизация завершена", "output": filtered_output}

    except subprocess.CalledProcessError as e:
        return {"status": "Ошибка при выполнении скрипта оптимизации", "error": e.stderr}


# Перечисление режимов
class ModeEnum(int, Enum):
    calibrate = 1
    optimize = 2
    select_model = 3

# Запросы
class ModeRequest(BaseModel):
    mode: ModeEnum

class ModelChoice(BaseModel):
    selected_model_name: Literal[
        "yolov8", "dlib", "mtcnn", "insightface", 
        "retinaFace", "mediapipe", "haarcascade", "ssd"
    ]



selected_model = {"current": None}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "current_model": selected_model["current"]})

@app.post("/run-mode-ui")
def run_mode_ui(request: Request, mode: int = Form(...)):
    if mode == 1:
        # Калибровка
        model = "yolov8"  # Модель по умолчанию для калибровки
        fps_threshold = 10.0
        cpu_threshold = 50.0
        result = run_calibration_script(model, fps_threshold, cpu_threshold)

    elif mode == 2:
        # Оптимизация
        result = run_optimization_script()

    elif mode == 3:
        # Выбор модели и запуск base.py
        model_map = {
            "yolov8": "yolov8",
            "dlib": "dlib",
            "mtcnn": "mtcnn",
            "insightface": "insightface",
            "retinaFace": "retinaface",
            "mediapipe": "mediapipe",
            "haarcascade": "haarcascade",
            "ssd": "ssd"
        }

        selected = selected_model["current"]

        if selected is None or selected not in model_map:
            result = {
                "status": "Сначала выберите модель через /select-model-ui",
                "available_models": list(model_map.keys())
            }
        else:
            model_name = model_map[selected]
            result = run_base_model_script(model_name)

    else:
        # Некорректный режим
        result = {"error": "Некорректный режим"}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "current_model": selected_model["current"]
    })


@app.post("/select-model-ui")
def select_model_ui(request: Request, model_name: str = Form(...)):
    selected_model["current"] = model_name

    # Запускаем base.py сразу после выбора модели
    result = run_base_model_script(model_name)
    result.update({"selected_model": model_name})  # добавим к результату info о модели

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "current_model": selected_model["current"]
    })


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)