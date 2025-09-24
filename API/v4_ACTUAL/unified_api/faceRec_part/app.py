# faceRec_part/app.py
from __future__ import annotations
import os, base64
from typing import List, Tuple, Dict
from pathlib import Path

import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="FaceRec Service (stable)")

# ---------- ПУТИ (жёстко) ----------
BASE_DIR  = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
STATIC_DIR= BASE_DIR / "static"
KNOWN_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ПАРАМЕТРЫ ----------
TOLERANCE     = 0.62
DETECT_SCALE  = 0.5
UPSAMPLE      = 1
DETECTOR      = "hog"      # есть CUDA/dlib -> "cnn"
ENC_MODEL     = "small"    # "large" точнее, но медленнее
JITTER        = 0
MIN_FACE      = 24

# ---------- ПАМЯТЬ ----------
KNOWN_ENCS: Dict[str, List[np.ndarray]] = {}
_loaded_once = False

def _safe(name: str) -> str:
    return str(name).strip().replace("/", "_").replace("\\", "_")

def _encodings_from_file(img_path: Path) -> List[np.ndarray]:
    try:
        img = face_recognition.load_image_file(str(img_path))   # RGB
        locs = face_recognition.face_locations(
            img, number_of_times_to_upsample=max(1, UPSAMPLE), model=DETECTOR
        )
        if not locs:
            locs = face_recognition.face_locations(img, number_of_times_to_upsample=2, model=DETECTOR)
        if not locs:
            return []
        encs = face_recognition.face_encodings(
            img, known_face_locations=locs, num_jitters=max(0, JITTER), model=ENC_MODEL
        )
        return encs or []
    except Exception:
        return []

def _load_known() -> None:
    global KNOWN_ENCS
    KNOWN_ENCS = {}
    if not KNOWN_DIR.exists():
        return
    for person_dir in KNOWN_DIR.iterdir():
        if person_dir.is_dir():
            encs: List[np.ndarray] = []
            for img in person_dir.glob("*.*"):
                encs.extend(_encodings_from_file(img))
            if encs:
                KNOWN_ENCS[person_dir.name] = encs
    print("[LOAD] KNOWN_DIR:", KNOWN_DIR)
    print("[LOAD] people:", sorted(KNOWN_ENCS.keys()))
    print("[LOAD] counts:", {k: len(v) for k, v in KNOWN_ENCS.items()})

def _ensure_loaded():
    global _loaded_once
    if not _loaded_once:
        _load_known()
        _loaded_once = True

# Инициализация при импорте и на старте
_load_known()
_loaded_once = True

@app.on_event("startup")
def _on_startup():
    _load_known()

# ---------- API (без дублей) ----------
class PeopleResponse(BaseModel):
    people: List[str]

@app.get("/people", response_model=PeopleResponse)
def api_people():
    # читаем папки из KNOWN_DIR каждый раз — так точно видны заранее добавленные люди
    people = sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()])
    return PeopleResponse(people=people)

@app.get("/people_v2", response_model=PeopleResponse)
def people_v2():
    """Альтернативный формат: объект { 'people': [...] }."""
    if not KNOWN_DIR.exists():
        return PeopleResponse(people=[])
    return PeopleResponse(people=sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()]))

@app.get("/people_status")
def people_status():
    """Короткий статус: имена + число энкодингов в памяти."""
    people = sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()]) if KNOWN_DIR.exists() else []
    counts = {k: len(v) for k, v in KNOWN_ENCS.items()}
    for name in people:
        counts.setdefault(name, 0)
    return {"people": people, "counts": counts}

@app.get("/people_refresh")
def people_refresh():
    """Принудительно перечитать known_faces и вернуть people_status."""
    _load_known()
    return people_status()

@app.get("/debug/encodings")
def debug_encodings():
    _ensure_loaded()
    return {k: len(v) for k, v in KNOWN_ENCS.items()}

@app.get("/debug/fs")
def debug_fs():
    tree = {}
    if KNOWN_DIR.exists():
        for p in KNOWN_DIR.iterdir():
            if p.is_dir():
                tree[p.name] = [f.name for f in p.glob("*.*")]
    return tree

@app.post("/reload")
def reload_db():
    _load_known()
    return {"ok": True, "counts": {k: len(v) for k, v in KNOWN_ENCS.items()}}

@app.post("/known_faces")
async def add_known(person: str = Form(...), files: List[UploadFile] = File(...)):
    _ensure_loaded()
    person = _safe(person)
    target = KNOWN_DIR / person
    target.mkdir(parents=True, exist_ok=True)

    saved = []
    for uf in files:
        data = await uf.read()
        if not data:
            continue
        ext  = os.path.splitext(uf.filename or "")[1] or ".jpg"
        name = os.path.splitext(os.path.basename(uf.filename or "img"))[0] + ext
        out  = target / name
        with open(out, "wb") as f:
            f.write(data)
        saved.append(name)

    encs: List[np.ndarray] = []
    for img in target.glob("*.*"):
        encs.extend(_encodings_from_file(img))
    if encs:
        KNOWN_ENCS[person] = encs

    return {"ok": True, "person": person, "saved": saved, "encodings": len(encs)}

# ---------- RECOGNITION / WS ----------
def _b64_to_ndarray(data_url: str) -> np.ndarray:
    comma = data_url.find(',')
    b64 = data_url[comma + 1:] if comma != -1 else data_url
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR

def _recognize(frame_bgr: np.ndarray):
    small = cv2.resize(frame_bgr, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs_small = face_recognition.face_locations(
        rgb_small, number_of_times_to_upsample=UPSAMPLE, model=DETECTOR
    )
    inv = 1.0 / float(DETECT_SCALE)
    locs_full = []
    for (t, r, b, l) in locs_small:
        T, R, B, L = int(t * inv), int(r * inv), int(b * inv), int(l * inv)
        if min(B - T, R - L) >= MIN_FACE:
            locs_full.append((T, R, B, L))

    boxes, names, dists = [], [], []
    if not locs_full:
        return boxes, names, dists

    rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(
        rgb_full, known_face_locations=locs_full, num_jitters=JITTER, model=ENC_MODEL
    )

    all_names: List[str] = []
    all_encs: List[np.ndarray] = []
    for n, lst in KNOWN_ENCS.items():
        for e in lst:
            all_names.append(n)
            all_encs.append(e)

    if encs and all_encs:
        known_mat = np.stack(all_encs, axis=0)
        for enc, (T, R, B, L) in zip(encs, locs_full):
            dif = known_mat - enc
            dist = np.linalg.norm(dif, axis=1)
            i = int(np.argmin(dist))
            best = float(dist[i])
            name = all_names[i] if best <= TOLERANCE else "Unknown"
            boxes.append((L, T, R, B))
            names.append(name)
            dists.append(best)
    else:
        for (T, R, B, L) in locs_full:
            boxes.append((L, T, R, B))
            names.append("Unknown")
            dists.append(1.0)

    return boxes, names, dists

@app.websocket("/")
async def ws_root(ws: WebSocket):
    await ws_endpoint(ws)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "frame":
                frame = _b64_to_ndarray(data.get("data", ""))
                boxes, names, dists = _recognize(frame)
                await ws.send_json({
                    "type": "result",
                    "boxes": boxes,
                    "names": names,
                    "dists": [round(float(x), 4) for x in dists],
                    "detections": [
                        {"x": int(L), "y": int(T), "w": int(max(0, R-L)), "h": int(max(0, B-T)),
                         "name": nm, "dist": round(float(ds), 4)}
                        for (L, T, R, B), nm, ds in zip(boxes, names, dists)
                    ]
                })
            else:
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass

# ---------- UI / статика ----------
@app.get("/", include_in_schema=False)
def face_root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"ok": True, "message": "face module up, no index.html"})

app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="face_static")

