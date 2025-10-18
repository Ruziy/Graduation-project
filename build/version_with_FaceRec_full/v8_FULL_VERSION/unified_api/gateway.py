from __future__ import annotations
import os, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader, select_autoescape
import importlib

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "optimizer_part"))
sys.path.insert(0, str(BASE_DIR / "faceRec_part"))

opt_module = importlib.import_module("main_V3")
optimizer_app = getattr(opt_module, "app")

face_module = importlib.import_module("app")
face_app = getattr(face_module, "app")       

app = FastAPI(title="🧩 Unified API Hub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jinja_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)

@app.get("/", response_class=HTMLResponse)
async def hub():
    html = jinja_env.get_template("hub.html").render()
    return HTMLResponse(html)
from fastapi.responses import RedirectResponse

@app.api_route("/select-model-ui", methods=["GET", "POST"], include_in_schema=False)
def _redir_select_model_ui():
    return RedirectResponse(url="/optimizer/select-model-ui", status_code=307)

@app.api_route("/run-mode-ui", methods=["GET", "POST"], include_in_schema=False)
def _redir_run_mode_ui():
    return RedirectResponse(url="/optimizer/run-mode-ui", status_code=307)

app.mount("/optimizer", optimizer_app)
app.mount("/face", face_app)
app.mount("/api",  face_app)  

app.add_api_websocket_route("/ws", face_module.ws_endpoint)

static_root = BASE_DIR / "static"
if static_root.exists():
    app.mount("/static", StaticFiles(directory=str(static_root), html=False), name="root_static")

@app.get("/healthz", response_class=HTMLResponse)
async def healthz():
    return HTMLResponse("ok")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
