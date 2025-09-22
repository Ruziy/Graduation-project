import io
import os
import base64
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import face_recognition

from settings import KNOWN_DIR, STATIC_DIR, TOLERANCE, DETECT_SCALE, UPSAMPLE, DETECTOR, ENC_MODEL_LIVE, JITTER_LIVE, MIN_FACE


app = FastAPI(title="Face Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Ensure known dir exists
KNOWN_DIR.mkdir(parents=True, exist_ok=True)

# In-memory encodings: name -> list[np.ndarray(128,)]
KNOWN_ENCS: Dict[str, List[np.ndarray]] = {}


def load_known_faces():
    """Load encodings from known_faces/* folders."""
    global KNOWN_ENCS
    KNOWN_ENCS = {}
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
    print(f"[INIT] Loaded people: {list(KNOWN_ENCS.keys())}")


load_known_faces()


def b64_to_ndarray(data_url: str) -> np.ndarray:
    """Decode data:image/jpeg;base64,... -> BGR ndarray."""
    comma = data_url.find(',')
    if comma != -1:
        b64 = data_url[comma + 1:]
    else:
        b64 = data_url
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return frame


def recognize_frame(frame_bgr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
    # Detect on downscaled
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

    # Flatten known encs for distance calc
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/people")
async def list_people():
    return {"people": sorted(KNOWN_ENCS.keys())}


@app.post("/api/known_faces")
async def add_person(person: str = Form(...), files: List[UploadFile] = File(...)):
    # Make person folder
    safe_person = person.strip().replace("/", "_").replace("\\", "_")
    target = KNOWN_DIR / safe_person
    target.mkdir(parents=True, exist_ok=True)

    saved = []
    for uf in files:
        content = await uf.read()
        if not content:
            continue
        # Guess extension
        ext = os.path.splitext(uf.filename or "")[1] or ".jpg"
        out_path = target / f"{os.path.splitext(os.path.basename(uf.filename or 'img'))[0]}{ext}"
        with open(out_path, "wb") as f:
            f.write(content)
        saved.append(str(out_path.name))

    # Reload encodings for this person only
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "frame":
                frame_b64 = data.get("data")
                frame = b64_to_ndarray(frame_b64)
                boxes, names, dists = recognize_frame(frame)
                await ws.send_json({
                    "type": "result",
                    "boxes": boxes,  # [l,t,r,b]
                    "names": names,
                    "dists": [round(float(x), 4) for x in dists],
                })
            else:
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
