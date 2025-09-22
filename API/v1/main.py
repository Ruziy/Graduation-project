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



# Перечисление режимов
class ModeEnum(int, Enum):
    calibrate = 1
    optimize = 2
    select_model = 3

# Запросы
class ModeRequest(BaseModel):
    mode: ModeEnum

class ModelChoice(BaseModel):
    model_name: Literal["Model A", "Model B", "Model C"]

selected_model = {"current": None}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "current_model": selected_model["current"]})

@app.post("/run-mode-ui")
def run_mode_ui(request: Request, mode: int = Form(...)):
    if mode == 1:
        # Здесь можно задать значения по умолчанию или получить их от пользователя
        model = "yolov8"
        fps_threshold = 10.0
        cpu_threshold = 50.0
        result = run_calibration_script(model, fps_threshold, cpu_threshold)
    elif mode == 2:
        result = {"status": "Оптимизация завершена (заглушка)"}
    elif mode == 3:
        result = {
            "status": "Выберите модель через /select-model/",
            "available_models": ["Model A", "Model B", "Model C"]
        }
    else:
        result = {"error": "Некорректный режим"}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "current_model": selected_model["current"]
    })

@app.post("/select-model-ui")
def select_model_ui(request: Request, model_name: str = Form(...)):
    selected_model["current"] = model_name
    result = {"selected_model": model_name, "status": "Модель успешно выбрана"}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "current_model": selected_model["current"]
    })

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)