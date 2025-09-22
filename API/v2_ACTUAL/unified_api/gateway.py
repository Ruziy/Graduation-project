from __future__ import annotations
import os, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
import importlib

BASE_DIR = Path(__file__).resolve().parent
# важно: добавить обе подпапки в sys.path
sys.path.insert(0, str(BASE_DIR / "optimizer_part"))
sys.path.insert(0, str(BASE_DIR / "faceRec_part"))

# ИМПОРТИРУЕМ ПАТЧНУТЫЙ МОДУЛЬ (main_V3_local), а не main_V3
opt_module = importlib.import_module("main_V3")
optimizer_app = getattr(opt_module, "app")

face_module = importlib.import_module("app")
face_app = getattr(face_module, "app")

app = FastAPI(title="🧩 Unified API Hub")

jinja_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)

@app.get("/", response_class=HTMLResponse)
async def hub():
    html = jinja_env.get_template("hub.html").render()
    return HTMLResponse(html)

app.mount("/optimizer", optimizer_app)
app.mount("/face", face_app)

@app.get("/healthz", response_class=HTMLResponse)
async def healthz():
    return HTMLResponse("ok")
