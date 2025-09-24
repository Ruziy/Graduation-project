# faceRec_part/app.py
from __future__ import annotations
import os, base64, shutil
from typing import List, Tuple, Dict
from pathlib import Path

import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="FaceRec Service")

# --- настройки ---
try:
    from .settings import KNOWN_DIR, STATIC_DIR, TOLERANCE, DETECT_SCALE, UPSAMPLE, DETECTOR, ENC_MODEL_LIVE, JITTER_LIVE, MIN_FACE
except Exception:
    BASE_DIR = Path(__file__).resolve().parent
    KNOWN_DIR = (BASE_DIR / "known_faces").resolve()
    STATIC_DIR = (BASE_DIR / "static").resolve()
    TOLERANCE = 0.5
    DETECT_SCALE = 0.25
    UPSAMPLE = 1
    DETECTOR = "hog"
    ENC_MODEL_LIVE = "small"
    JITTER_LIVE = 0
    MIN_FACE = 40

KNOWN_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# --- память с энкодингами ---
KNOWN_ENCS: Dict[str, List[np.ndarray]] = {}

def _safe_name(name: str) -> str:
    return name.strip().replace("/", "_").replace("\\", "_")

def _load_known_faces() -> None:
    """Загрузить энкодинги из KNOWN_DIR/* в память."""
    global KNOWN_ENCS
    KNOWN_ENCS = {}
    if not KNOWN_DIR.exists():
        return
    for person_dir in KNOWN_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        enc_list: List[np.ndarray] = []
        for img_path in person_dir.glob("*.*"):
            try:
                img = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img, model="large", num_jitters=1)
                enc_list.extend(encs)
            except Exception:
                continue
        if enc_list:
            KNOWN_ENCS[name] = enc_list

@app.on_event("startup")
def _on_startup():
    _load_known_faces()
    try:
        print("[INIT] Loaded people:", sorted(KNOWN_ENCS.keys()))
    except Exception:
        pass

# --- API ---
from pydantic import BaseModel
from typing import List

class PeopleResponse(BaseModel):
    people: List[str]

@app.get("/people", response_model=PeopleResponse)
def api_people():
    # читаем папки из KNOWN_DIR каждый раз — так точно видны заранее добавленные люди
    people = sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()])
    return PeopleResponse(people=people)

@app.post("/reload")
def api_reload():
    _load_known_faces()
    return {"ok": True, "people": sorted(KNOWN_ENCS.keys())}

@app.post("/known_faces")
async def api_known_faces(person: str = Form(...), files: List[UploadFile] = File(...)):
    safe_person = _safe_name(person)
    target = KNOWN_DIR / safe_person
    target.mkdir(parents=True, exist_ok=True)

    saved = []
    for uf in files:
        content = await uf.read()
        if not content:
            continue
        ext = os.path.splitext(uf.filename or "")[1] or ".jpg"
        out_name = os.path.splitext(os.path.basename(uf.filename or "img"))[0] + ext
        out_path = target / out_name
        with open(out_path, "wb") as f:
            f.write(content)
        saved.append(out_path.name)

    # перезагрузим энкодинги только для этого человека
    enc_list: List[np.ndarray] = []
    for img_path in target.glob("*.*"):
        try:
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img, model="large", num_jitters=1)
            enc_list.extend(encs)
        except Exception:
            continue
    if enc_list:
        KNOWN_ENCS[safe_person] = enc_list

    return {"ok": True, "person": safe_person, "saved": saved, "encodings": len(enc_list)}

# --- WS ---
def _b64_to_ndarray(data_url: str) -> np.ndarray:
    comma = data_url.find(',')
    b64 = data_url[comma + 1:] if comma != -1 else data_url
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return frame

def _recognize_frame(frame_bgr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
    small = cv2.resize(frame_bgr, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs_small = face_recognition.face_locations(
        rgb_small, number_of_times_to_upsample=UPSAMPLE, model=DETECTOR
    )
    inv = 1.0 / float(DETECT_SCALE)
    locs_full = []
    for (top, right, bottom, left) in locs_small:
        t, r, b, l = int(top * inv), int(right * inv), int(bottom * inv), int(left * inv)
        if min(b - t, r - l) >= MIN_FACE:
            locs_full.append((t, r, b, l))

    boxes, names, dists = [], [], []
    if not locs_full:
        return boxes, names, dists

    rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(
        rgb_full, known_face_locations=locs_full,
        num_jitters=JITTER_LIVE, model=ENC_MODEL_LIVE
    )

    all_names: List[str] = []
    all_encs: List[np.ndarray] = []
    for n, lst in KNOWN_ENCS.items():
        for e in lst:
            all_names.append(n)
            all_encs.append(e)

    if encs and all_encs:
        known_mat = np.asarray(all_encs, dtype=np.float32)
        for enc, (t, r, b, l) in zip(encs, locs_full):
            dif = known_mat - enc
            dist = np.sqrt(np.sum(dif * dif, axis=1))
            idx = int(np.argmin(dist))
            best = float(dist[idx])
            name = all_names[idx] if best <= TOLERANCE else "Unknown"
            boxes.append((l, t, r, b))
            names.append(name)
            dists.append(best)
    else:
        for (t, r, b, l) in locs_full:
            boxes.append((l, t, r, b))
            names.append("Unknown")
            dists.append(1.0)

    return boxes, names, dists
@app.websocket("/")
async def ws_root(ws: WebSocket):
    # просто прокинем на твой уже существующий обработчик
    await ws_endpoint(ws)
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "frame":
                frame_b64 = data.get("data")
                if not frame_b64:
                    await ws.send_json({"type": "error", "message": "no frame"})
                    continue

                frame = _b64_to_ndarray(frame_b64)
                boxes, names, dists = _recognize_frame(frame)  # boxes = [(l,t,r,b), ...]

                # старый формат (сохраняем на всякий)
                resp = {
                    "type": "result",
                    "boxes": boxes,
                    "names": names,
                    "dists": [round(float(x), 4) for x in dists],
                }

                # новый удобный формат для многих UI: список объектов с x,y,w,h
                objs = []
                for (l, t, r, b), name, dist in zip(boxes, names, dists):
                    w = max(0, r - l)
                    h = max(0, b - t)
                    objs.append({
                        "x": int(l),
                        "y": int(t),
                        "w": int(w),
                        "h": int(h),
                        "name": name,
                        "dist": round(float(dist), 4),
                    })
                resp["detections"] = objs

                await ws.send_json(resp)
            else:
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


# --- UI: / -> index.html и статика по /static ---
@app.get("/", include_in_schema=False)
def face_root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"ok": True, "message": "face module up, no index.html"}

app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="face_static")
