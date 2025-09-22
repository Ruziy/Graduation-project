from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from enum import Enum
from typing import Literal
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="🧠 Умный API: Калибровка, Оптимизация, Выбор модели")

# Подключаем шаблоны и статику
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
        result = {"status": "Калибровка завершена (заглушка)"}
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