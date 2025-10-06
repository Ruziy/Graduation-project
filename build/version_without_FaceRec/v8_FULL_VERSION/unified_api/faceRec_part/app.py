from __future__ import annotations
import os, io, base64, json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import time
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="FaceRec Service (pluggable)")

BASE_DIR   = Path(__file__).resolve().parent
KNOWN_DIR  = BASE_DIR / "known_faces"
STATIC_DIR = BASE_DIR / "static"
KNOWN_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_DETECTORS = {"yolov8","mtcnn","insightface","retinaface","mediapipe","dlib","haarcascade","ssd"}

def _read_detector_from_txt(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        val = path.read_text(encoding="utf-8").strip().lower()
        return val if val in ALLOWED_DETECTORS and val else None
    except Exception:
        return None
global DETECTOR, _detector
DET_FILE = BASE_DIR / "log.txt"            
det_from_txt = _read_detector_from_txt(DET_FILE)

DETECTOR = det_from_txt or "mediapipe"

ENCODER  = os.getenv("ENCODER", "dlib").lower()  

#!!!!!!! ВАЖНЫЕ ПАРАМЕТРЫ
DETECT_SCALE = float(os.getenv("DETECT_SCALE", "0.5"))     
UPSAMPLE     = int(os.getenv("UPSAMPLE", "1"))
ENC_MODEL    = os.getenv("DLIB_ENCODER_MODEL", "small")     
JITTER       = int(os.getenv("DLIB_JITTER", "0"))
MIN_FACE     = int(os.getenv("MIN_FACE", "24"))
TOL_DLIB       = float(os.getenv("TOLERANCE_DLIB", "0.62"))
TOL_ARCFACE_CS = float(os.getenv("TOLERANCE_ARCFACE_COS", "0.35"))

KNOWN_ENCS: Dict[str, List[np.ndarray]] = {}
_loaded_once = False
_detector = None
_encoder  = None

def _safe(name: str) -> str:
    return str(name).strip().replace('/', '_').replace('\\', '_')

def _to_trbl_from_xywh(x: int, y: int, w: int, h: int) -> Tuple[int,int,int,int]:
    T, L = y, x
    B, R = y + h, x + w
    return (T, R, B, L)

class FaceDetector:
    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """Return list of boxes in (top, right, bottom, left) on ORIGINAL image scale."""
        raise NotImplementedError

class DlibDetector(FaceDetector):
    def __init__(self, mode: str = "hog", upsample: int = 1):
        import face_recognition  
        self.fr = face_recognition
        self.mode = mode
        self.upsample = max(1, int(upsample))

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locs = self.fr.face_locations(rgb, number_of_times_to_upsample=self.upsample, model=self.mode)
        return [(t, r, b, l) for (t, r, b, l) in locs]

class MediaPipeDetector(FaceDetector):
    def __init__(self, min_conf: float = 0.5):
        try:
            import mediapipe as mp
        except Exception as e:
            raise RuntimeError("mediapipe не установлен. pip install mediapipe") from e
        self.mp = mp
        self.min_conf = float(min_conf)

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        h, w = img_bgr.shape[:2]
        with self.mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=self.min_conf) as fd:
            res = fd.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        boxes = []
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                ww = max(0, int(bb.width * w))
                hh = max(0, int(bb.height * h))
                boxes.append(_to_trbl_from_xywh(x, y, ww, hh))
        return boxes

class HaarDetector(FaceDetector):
    def __init__(self, cascade_path: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent.parent / "optimizer_part" / "weights"
        default_cascade = base_dir / "haarcascade_frontalface_default.xml"
        cascade_path = (
            cascade_path
            or os.getenv("HAAR_CASCADE")
            or (str(default_cascade) if default_cascade.exists() else cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        )
        self.clf = cv2.CascadeClassifier(cascade_path)
        if self.clf.empty():
            raise RuntimeError(f"Не удалось загрузить Haar каскад: {cascade_path}")

        print(f"[INFO] Haar каскад успешно загружен: {cascade_path}")  

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = self.clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(16, 16))
        boxes = []
        for (x, y, w, h) in rects:
            boxes.append(_to_trbl_from_xywh(int(x), int(y), int(w), int(h)))
        return boxes

class SSDDetector(FaceDetector):
    def __init__(self, prototxt: Optional[str] = None, weights: Optional[str] = None, conf: float = 0.5):
        base_dir = Path(__file__).resolve().parent.parent / "optimizer_part" / "weights"
        default_prototxt = base_dir / "deploy.prototxt.txt"
        default_weights  = base_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        prototxt = prototxt or os.getenv("SSD_PROTOTXT") or str(default_prototxt)
        weights  = weights  or os.getenv("SSD_WEIGHTS")  or str(default_weights)

        if not Path(prototxt).exists() or not Path(weights).exists():
            raise RuntimeError(
                f"Файлы модели не найдены.\n"
                f"prototxt: {prototxt}\nweights: {weights}\n"
                "Укажите правильные пути или переменные среды SSD_PROTOTXT и SSD_WEIGHTS."
            )

        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.conf = float(conf)

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score >= self.conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x = max(0, x1)
                y = max(0, y1)
                ww = max(0, x2 - x1)
                hh = max(0, y2 - y1)
                boxes.append(_to_trbl_from_xywh(x, y, ww, hh))
        return boxes


class MTCNNDetector(FaceDetector):
    def __init__(self, min_conf: float = 0.5):
        try:
            from mtcnn import MTCNN
        except Exception as e:
            raise RuntimeError("mtcnn не установлен. pip install mtcnn") from e
        self.det = MTCNN()
        self.min_conf = float(min_conf)

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        res = self.det.detect_faces(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        boxes = []
        for r in res:
            if r.get('confidence', 0) >= self.min_conf and 'box' in r:
                x,y,w,h = r['box']
                boxes.append(_to_trbl_from_xywh(int(x), int(y), int(w), int(h)))
        return boxes

class YOLOv8Detector(FaceDetector):
    def __init__(self, model_path: Optional[str] = None, conf: float = 0.3):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("ultralytics не установлен. pip install ultralytics") from e

        default_model_path = (
            Path(__file__).resolve().parent.parent / "optimizer_part" / "weights" / "yolov8n-face.pt"
        )

        model_path = (
            model_path
            or os.getenv("DETECTOR_MODEL")
            or str(default_model_path)
        )

        if not Path(model_path).exists():
            raise RuntimeError(f"Файл модели не найден по пути: {model_path}")

        self.model = YOLO(model_path)
        self.conf = float(conf)

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = img_bgr.shape[:2]
        res = self.model.predict(source=img_bgr[..., ::-1], verbose=False, conf=self.conf)
        boxes = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b.astype(int)
                x = max(0, x1)
                y = max(0, y1)
                ww = max(0, x2 - x1)
                hh = max(0, y2 - y1)
                boxes.append(_to_trbl_from_xywh(x, y, ww, hh))
        return boxes

class RetinaFaceDetector(FaceDetector):
    def __init__(self, provider: str = "auto"):
        try:
            from retinaface import RetinaFace
        except Exception as e:
            raise RuntimeError("retinaface не установлен. pip install retinaface") from e
        self.api = RetinaFace

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        data = self.api.detect_faces(img_bgr)
        boxes = []
        if isinstance(data, dict):
            for k, v in data.items():
                (x1,y1,x2,y2) = v['facial_area']
                x = int(x1); y = int(y1); w = int(x2-x1); h = int(y2-y1)
                boxes.append(_to_trbl_from_xywh(x,y,w,h))
        return boxes

class InsightFaceDetector(FaceDetector):
    def __init__(self, provider: str = "cpu"):
        try:
            import insightface
        except Exception as e:
            raise RuntimeError("insightface не установлен. pip install insightface onnxruntime") from e
        from insightface.app import FaceAnalysis
        self.FaceAnalysis = FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0 if provider=="gpu" else -1)

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = self.app.get(img_bgr)
        boxes = []
        for f in faces:
            x1,y1,x2,y2 = f.bbox.astype(int)
            x = int(x1); y = int(y1); w = int(x2-x1); h = int(y2-y1)
            boxes.append(_to_trbl_from_xywh(x,y,w,h))
        return boxes

class FaceEncoder:
    def encode(self, img_bgr: np.ndarray, boxes_trbl: List[Tuple[int,int,int,int]]) -> List[np.ndarray]:
        raise NotImplementedError

class DlibEncoder(FaceEncoder):
    def __init__(self, model: str = "small", jitter: int = 0):
        import face_recognition
        self.fr = face_recognition
        self.model = model
        self.jitter = int(jitter)

    def encode(self, img_bgr: np.ndarray, boxes_trbl: List[Tuple[int,int,int,int]]) -> List[np.ndarray]:
        if not boxes_trbl:
            return []
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        encs = self.fr.face_encodings(rgb, known_face_locations=boxes_trbl, num_jitters=self.jitter, model=self.model)
        return [np.asarray(e, dtype=np.float32) for e in encs]

class InsightFaceEncoder(FaceEncoder):
    def __init__(self, provider: str = "cpu"):
        try:
            import insightface
        except Exception as e:
            raise RuntimeError("insightface не установлен. pip install insightface onnxruntime") from e
        from insightface.app import FaceAnalysis
        self.FaceAnalysis = FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0 if provider=="gpu" else -1)

    def encode(self, img_bgr: np.ndarray, boxes_trbl: List[Tuple[int,int,int,int]]) -> List[np.ndarray]:
        faces = self.app.get(img_bgr)
        out = []
        for f in faces:
            if boxes_trbl:
                x1,y1,x2,y2 = map(int, f.bbox)
                def trbl_to_xyxy(b):
                    t,r,bottom,l = b
                    return (l, t, r, bottom)  
                ok = False
                for b in boxes_trbl:
                    bx1,by1,bx2,by2 = trbl_to_xyxy(b)
                    xx1, yy1 = max(x1,bx1), max(y1,by1)
                    xx2, yy2 = min(x2,bx2), min(y2,by2)
                    inter = max(0, xx2-xx1) * max(0, yy2-yy1)
                    area1 = (x2-x1)*(y2-y1)
                    area2 = (bx2-bx1)*(by2-by1)
                    if inter > 0 and inter >= 0.1*min(area1, area2):
                        ok = True; break
                if not ok:
                    continue
            emb = np.asarray(f.embedding, dtype=np.float32)
            n = np.linalg.norm(emb) + 1e-9
            out.append(emb / n)
        return out

DET_ALIASES = {
    "yolov8": "yolov8",
    "mtcnn": "mtcnn",
    "insightface": "insightface",
    "retinaface": "retinaface",
    "mediapipe": "mediapipe",
    "dlib": "dlib",
    "haarcascade": "haarcascade",
    "ssd": "ssd",
}

ENC_ALIASES = {
    "dlib": "dlib",
    "insightface": "insightface",
}

def make_detector(name: str) -> FaceDetector:
    n = DET_ALIASES.get(name.lower(), name.lower())
    if n == "dlib":
        backend = os.getenv("DLIB_BACKEND", "hog").lower()
        return DlibDetector(mode="cnn" if backend=="cnn" else "hog", upsample=UPSAMPLE)
    if n == "mediapipe":
        return MediaPipeDetector()
    if n == "haarcascade":
        return HaarDetector()
    if n == "ssd":
        return SSDDetector()
    if n == "mtcnn":
        return MTCNNDetector()
    if n == "yolov8":
        return YOLOv8Detector()
    if n == "retinaface":
        return RetinaFaceDetector()
    if n == "insightface":
        return InsightFaceDetector()
    raise ValueError(f"Неизвестный детектор: {name}")

def make_encoder(name: str) -> FaceEncoder:
    n = ENC_ALIASES.get(name.lower(), name.lower())
    if n == "dlib":
        return DlibEncoder(model=ENC_MODEL, jitter=JITTER)
    if n == "insightface":
        return InsightFaceEncoder()
    raise ValueError(f"Неизвестный энкодер: {name}")

def _encodings_from_file(img_path: Path) -> List[np.ndarray]:
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    boxes = _detector.detect(img)
    if not boxes:
        return []
    return _encoder.encode(img, boxes)

def _load_known() -> None:
    global KNOWN_ENCS
    KNOWN_ENCS = {}
    if not KNOWN_DIR.exists():
        return
    for person_dir in KNOWN_DIR.iterdir():
        if person_dir.is_dir():
            encs: List[np.ndarray] = []
            for img in sorted(person_dir.glob("*.*")):
                try:
                    encs.extend(_encodings_from_file(img))
                except Exception:
                    continue
            if encs:
                KNOWN_ENCS[person_dir.name] = encs
    print("[LOAD] people:", sorted(KNOWN_ENCS.keys()))
    print("[LOAD] counts:", {k: len(v) for k, v in KNOWN_ENCS.items()})

def _ensure_initialized(detector: Optional[str] = None, encoder: Optional[str] = None):
    global _loaded_once, _detector, _encoder
    if _detector is None or (detector and detector.lower()!=DETECTOR):
        _detector = make_detector(detector or DETECTOR)
    if _encoder is None or (encoder and encoder.lower()!=ENCODER):
        _encoder  = make_encoder(encoder or ENCODER)
    if not _loaded_once:
        _load_known()
        _loaded_once = True

_ensure_initialized()

@app.on_event("startup")
def _on_startup():
    _ensure_initialized()

class PeopleResponse(BaseModel):
    people: List[str]

@app.get("/people", response_model=PeopleResponse)
def api_people():
    people = sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()]) if KNOWN_DIR.exists() else []
    return PeopleResponse(people=people)

@app.get("/people_status")
def people_status():
    counts = {k: len(v) for k, v in KNOWN_ENCS.items()}
    people = sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()]) if KNOWN_DIR.exists() else []
    for name in people:
        counts.setdefault(name, 0)
    return {"people": people, "counts": counts, "detector": DETECTOR, "encoder": ENCODER}

@app.get("/people_refresh")
def people_refresh(detector: Optional[str]=Query(None), encoder: Optional[str]=Query(None)):
    _ensure_initialized(detector=detector, encoder=encoder)
    _load_known()
    return people_status()

@app.get("/debug/encodings")
def debug_encodings():
    return {k: len(v) for k, v in KNOWN_ENCS.items()}

@app.get("/debug/fs")
def debug_fs():
    tree: Dict[str, List[str]] = {}
    if KNOWN_DIR.exists():
        for p in KNOWN_DIR.iterdir():
            if p.is_dir():
                tree[p.name] = [f.name for f in p.glob("*.*")]
    return tree

@app.post("/reload")
def reload_db(detector: Optional[str]=Query(None), encoder: Optional[str]=Query(None)):
    _ensure_initialized(detector=detector, encoder=encoder)
    _load_known()
    return {"ok": True, "counts": {k: len(v) for k, v in KNOWN_ENCS.items()}, "detector": detector or DETECTOR, "encoder": encoder or ENCODER}

@app.post("/known_faces")
async def add_known(person: str = Form(...), files: List[UploadFile] = File(...)):
    _ensure_initialized()
    person = _safe(person)
    target = KNOWN_DIR / person
    target.mkdir(parents=True, exist_ok=True)

    saved = []
    for uf in files:
        data = await uf.read()
        if not data:
            continue
        name = os.path.basename(uf.filename or "img.jpg")
        out  = target / name
        with open(out, "wb") as f:
            f.write(data)
        saved.append(name)

    encs: List[np.ndarray] = []
    for img in target.glob("*.*"):
        try:
            encs.extend(_encodings_from_file(img))
        except Exception:
            pass
    if encs:
        KNOWN_ENCS[person] = encs

    return {"ok": True, "person": person, "saved": saved, "encodings": len(encs)}

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

class Matcher:
    def __init__(self, mode: str = ENCODER):
        self.mode = mode
        self._refresh_bank()

    def _refresh_bank(self):
        self.names: List[str] = []
        vecs: List[np.ndarray] = []
        for n, lst in KNOWN_ENCS.items():
            for e in lst:
                self.names.append(n)
                vecs.append(np.asarray(e, dtype=np.float32))
        self.bank = np.stack(vecs, axis=0) if vecs else None

    def match(self, encs: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        if not encs:
            return [], []
        if self.bank is None or self.bank.size == 0:
            return ["Unknown"]*len(encs), [1.0]*len(encs)

        names: List[str] = []
        scores: List[float] = []
        if self.mode == "dlib":
            for e in encs:
                dif = self.bank - e
                dist = np.linalg.norm(dif, axis=1)
                i = int(np.argmin(dist))
                best = float(dist[i])
                name = self.names[i] if best <= TOL_DLIB else "Unknown"
                names.append(name); scores.append(best)
        else:
            for e in encs:
                sims = self.bank @ e / (np.linalg.norm(self.bank, axis=1)* (np.linalg.norm(e)+1e-9) + 1e-9)
                i = int(np.argmax(sims))
                best_sim = float(sims[i])
                best_dist = 1.0 - best_sim
                name = self.names[i] if best_dist <= TOL_ARCFACE_CS else "Unknown"
                names.append(name); scores.append(best_dist)
        return names, scores


#!!!!!!! ВАЖНЫЕ ПАРАМЕТРЫ
DET_EVERY_N      = int(os.getenv("DET_EVERY_N", "4"))       # детект каждые N кадров
RATE_LIMIT_FPS   = float(os.getenv("RATE_LIMIT_FPS", "0"))  # 0 = без лимита; иначе, напр., 15
MAX_TRACK_FACES  = int(os.getenv("MAX_TRACK_FACES", "5"))   # максимум лиц для трекинга/энкодинга на кадр

def _create_kcf():
    try:
        from cv2 import legacy as cv2_legacy
        return cv2_legacy.TrackerKCF_create()
    except Exception:
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        try:
            from cv2 import legacy as cv2_legacy
            return cv2_legacy.TrackerCSRT_create()
        except Exception:
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
    raise RuntimeError("Ни KCF, ни CSRT трекер недоступны в текущей сборке OpenCV.")

def _b64_to_ndarray(data_url: str) -> np.ndarray:
    comma = data_url.find(',')
    b64 = data_url[comma + 1:] if comma != -1 else data_url
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.websocket("/")
async def ws_root(ws: WebSocket):
    await ws_endpoint(ws)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    q = ws.scope.get("query_string", b"").decode()
    params = dict([p.split("=") for p in q.split("&") if "=" in p]) if q else {}
    det = params.get("detector")
    enc = params.get("encoder")
    try:
        _ensure_initialized(detector=det, encoder=enc)
    except Exception as e:
        await ws.accept()
        await ws.send_json({"type":"error", "message": str(e)})
        await ws.close()
        return

    await ws.accept()
    matcher = Matcher(mode=ENCODER if enc is None else enc)

    frame_id = 0
    trackers = []  
    last_send_t = 0.0

    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") != "frame":
                await ws.send_json({"type": "pong"})
                continue

            global DETECTOR, _detector

            if RATE_LIMIT_FPS > 0:
                min_dt = 1.0 / RATE_LIMIT_FPS
                now = time.monotonic()
                if now - last_send_t < min_dt:
                    continue
                last_send_t = now

            frame = _b64_to_ndarray(data.get("data", ""))

            inv = 1.0
            small = frame
            if 0 < DETECT_SCALE < 1.0:
                small = cv2.resize(frame, (0,0), fx=DETECT_SCALE, fy=DETECT_SCALE)
                inv = 1.0/DETECT_SCALE

            file_det = None
            if det is None:
                file_det = _read_detector_from_txt(DET_FILE)
                if file_det and file_det != DETECTOR:
                    print(f"[CONFIG] Переключаю детектор: {DETECTOR} -> {file_det}", flush=True)
                    DETECTOR = file_det
                    _detector = make_detector(DETECTOR)

            if _detector is None:
                _detector = make_detector(DETECTOR)

            source = "query" if det is not None else ("log.txt" if file_det else "default")
            print(
                f"[CONFIG] detector_active={DETECTOR} | "
                f"detector_in_file={file_det or _read_detector_from_txt(DET_FILE) or '—'} | "
                f"source={source}",
                flush=True
            )

            frame_id += 1
            use_detect = (frame_id % DET_EVERY_N == 1) or (len(trackers) == 0)

            boxes_full: List[Tuple[int,int,int,int]] = []

            if use_detect:
                boxes_small = _detector.detect(small)
                for (t,r,b,l) in boxes_small:
                    T,R,B,L = int(t*inv), int(r*inv), int(b*inv), int(l*inv)
                    if min(B-T, R-L) >= MIN_FACE:
                        boxes_full.append((T,R,B,L))

                if len(boxes_full) > MAX_TRACK_FACES:
                    boxes_full = boxes_full[:MAX_TRACK_FACES]

                trackers = []
                for (T,R,B,L) in boxes_full:
                    tr = _create_kcf()
                    tr.init(frame, (int(L), int(T), int(R-L), int(B-T)))
                    trackers.append({"tr": tr, "name": "Unknown", "score": 1.0})

                if boxes_full:
                    encs = _encoder.encode(frame, boxes_full)
                else:
                    encs = []

                matcher._refresh_bank()
                names, scores = matcher.match(encs)

                for i, (nm, sc) in enumerate(zip(names, scores)):
                    if i < len(trackers):
                        trackers[i]["name"] = nm
                        trackers[i]["score"] = float(sc)
            else:
                alive = []
                for trk in trackers:
                    ok, box = trk["tr"].update(frame)
                    if not ok:
                        continue
                    L, T, W, H = box
                    L, T, W, H = int(L), int(T), int(W), int(H)
                    T_, R_, B_, L_ = T, L+W, T+H, L
                    if min(B_-T_, R_-L_) >= MIN_FACE:
                        boxes_full.append((T_, R_, B_, L_))
                        alive.append(trk)
                trackers = alive
                names  = [t["name"] for t in trackers]
                scores = [float(t["score"]) for t in trackers]

            dets = []
            N = min(len(boxes_full), len(names), len(scores))
            for i in range(N):
                (T,R,B,L) = boxes_full[i]
                nm, sc = names[i], scores[i]
                dets.append({
                    "x": int(L), "y": int(T), "w": int(max(0, R-L)), "h": int(max(0, B-T)),
                    "name": nm,
                    "score": round(float(sc), 4),
                })

            await ws.send_json({
                "type": "result",
                "boxes": [(int(L),int(T),int(R),int(B)) for (T,R,B,L) in boxes_full[:N]],
                "names": names[:N],
                "scores": [round(float(s), 4) for s in scores[:N]],
                "detections": dets,
                "detector": det or DETECTOR,
                "encoder": enc or ENCODER,
            })

    except WebSocketDisconnect:
        pass


@app.get("/", include_in_schema=False)
def face_root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"ok": True, "message": "face module up, no index.html"})

@app.get("/people-ui", response_class=HTMLResponse)
def people_ui():
    names = sorted([p.name for p in KNOWN_DIR.iterdir() if p.is_dir()]) if KNOWN_DIR.exists() else []
    items = "\n".join(f"<li>{n}</li>" for n in names) or '<li class="muted">Список пуст</li>'
    return HTMLResponse(f"""<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Известные лица</title>
  <style>
    :root{{--bg:#0b0f14;--card:rgba(255,255,255,.06);--bd:rgba(255,255,255,.16);--tx:#e7eef7;--muted:#9fb0c6;--brand:#6ea8fe;--r:16px}}
    @media (prefers-color-scheme:light){{:root{{--bg:#f7f9fc;--card:#fff;--bd:rgba(16,24,40,.08);--tx:#0f172a;--muted:#4a5568;--brand:#2563eb}}}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--tx);font:14px/1.5 system-ui,-apple-system,Segoe UI,Inter,Roboto,Arial}}
    .wrap{{max-width:860px;margin:32px auto;padding:0 16px}}
    .card{{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);padding:18px}}
    h1{{margin:0 0 12px;font-size:20px}}
    .row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}}
    .btn{{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:12px;border:1px solid var(--bd);color:#fff;background:linear-gradient(135deg,var(--brand),#00d4ff);text-decoration:none}}
    .btn:hover{{filter:brightness(1.05)}}
    ul{{list-style:none;margin:0;padding:0;display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}}
    li{{background:rgba(255,255,255,.04);border:1px solid var(--bd);border-radius:12px;padding:10px 12px}}
    .muted{{color:var(--muted)}}
    .footer{{margin-top:12px;color:var(--muted)}}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>👥 Известные лица</h1>
      <div class=\"row\">
        <a class=\"btn\" href=\"/people\" target=\"_blank\" rel=\"noopener\">Открыть как JSON</a>
        <a class=\"btn\" href=\"/face/\" rel=\"noopener\">Перейти к распознаванию</a>
      </div>
      <ul>{items}</ul>
      <div class=\"footer\">Папка источника: <code>{KNOWN_DIR}</code></div>
    </div>
  </div>
</body>
</html>""")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="face_static")
