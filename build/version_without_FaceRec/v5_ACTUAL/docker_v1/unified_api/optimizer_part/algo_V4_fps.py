# algo_V4_fps.py  (headless smart selector)
import os, time, sys
import cv2, numpy as np, torch, dlib, mediapipe as mp
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
try:
    from face_detection import RetinaFace
except Exception:
    from retinaface import RetinaFace

def calculate_image_metrics(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(hsv[..., 2].mean())
    contrast   = float(gray.std())
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise      = float(np.mean(np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0))))
    return dict(brightness=brightness, contrast=contrast, sharpness=sharpness, noise=noise)

def normalize(value, lo, hi):
    if hi == lo: return 0.0
    x = (value - lo) / (hi - lo)
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def analyze_scene_profile(m):
    p = []
    if m["brightness"] < 40:   p.append("dark")
    elif m["brightness"] > 180: p.append("overexposed")
    if m["contrast"] < 30:     p.append("flat")
    elif m["contrast"] > 100:  p.append("high-contrast")
    if m["sharpness"] < 50:     p.append("blurry")
    if m["noise"] > 50:         p.append("noisy")
    return " ".join(p)

def compute_detector_scores(metrics):
    n = {
        k: normalize(metrics[k], *rng) for k, rng in {
            "brightness": (0, 255),
            "contrast": (0, 100),
            "sharpness": (0, 500),
            "noise": (0, 100),
        }.items()
    }
    base = {"yolov8":0.9,"retinaface":1.0,"insightface":0.8,"mtcnn":0.3,
            "mediapipe":0.5,"ssd":0.4,"dlib":0.2,"haarcascade":0.5}
    return {
        "yolov8":      3.9*n["contrast"] + 3.9*n["brightness"] - 2.1*n["noise"] + 3.6*n["sharpness"] + base["yolov8"],
        "retinaface":  3.6*n["contrast"] + 3.6*n["brightness"] - 1.2*n["noise"] + 3.6*n["sharpness"] + base["retinaface"],
        "insightface": 3.3*n["contrast"] + 3.3*n["brightness"] - 0.6*n["noise"] + 3.0*n["sharpness"] + base["insightface"],
        "mtcnn":       1.5*n["contrast"] + 1.5*n["brightness"] - 2.1*n["noise"] + 2.1*n["sharpness"] + base["mtcnn"],
        "mediapipe":   0.9*n["contrast"] + 2.7*n["brightness"] - 3.6*n["noise"] + 1.5*n["sharpness"] + base["mediapipe"],
        "ssd":         1.2*n["contrast"] + 1.2*n["brightness"] - 2.4*n["noise"] + 2.1*n["sharpness"] + base["ssd"],
        "dlib":        1.2*n["contrast"] + 1.2*n["brightness"] - 3.0*n["noise"] + 1.2*n["sharpness"] + base["dlib"],
        "haarcascade": 2.1*n["contrast"] + 2.1*n["brightness"] - 1.5*n["noise"] + 2.4*n["sharpness"] + base["haarcascade"],
    }

def init_detectors(device, weights_dir):
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
    dets = {
        "yolov8": YOLO(str(weights_dir / "yolov8n-face.pt")),
        "mtcnn": MTCNN(keep_all=True, device=device),
        "retinaface": RetinaFace(),
        "mediapipe": mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5),
        "dlib": dlib.get_frontal_face_detector(),
        "insightface": FaceAnalysis(name="buffalo_l", providers=providers),
        "ssd": cv2.dnn.readNetFromCaffe(str(weights_dir / "deploy.prototxt.txt"),
                                        str(weights_dir / "res10_300x300_ssd_iter_140000.caffemodel"))
    }
    dets["insightface"].prepare(ctx_id=(0 if device=="cuda" else -1))
    return dets

def detect_faces(frame, model_name, detector=None, net=None, conf=0.5):
    h, w = frame.shape[:2]
    if model_name == "yolov8":
        res = detector(frame)[0]
        return [b for b in res.boxes.xyxy.cpu().numpy()]
    elif model_name == "dlib":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [[f.left(), f.top(), f.right(), f.bottom()] for f in detector(gray)]
    elif model_name == "mtcnn":
        boxes, _ = detector.detect(frame)
        return [] if boxes is None else boxes
    elif model_name == "insightface":
        return [f.bbox for f in detector.get(frame)]
    elif model_name == "retinaface":
        faces = detector(frame)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b, lm, score in faces if score >= conf]
    elif model_name == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        out = []
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1, y1 = int(bb.xmin*w), int(bb.ymin*h)
                ww, hh = int(bb.width*w), int(bb.height*h)
                out.append([x1, y1, x1+ww, y1+hh])
        return out
    elif model_name == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104,177,123), swapRB=False, crop=False)
        net.setInput(blob)
        dets = net.forward()
        out = []
        for i in range(dets.shape[2]):
            c = dets[0,0,i,2]
            if c >= conf:
                box = dets[0,0,i,3:7] * np.array([w,h,w,h])
                out.append(box.astype(int).tolist())
        return out
    return []

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_dir = Path(__file__).resolve().parent / "weights"

    # Камера / RTSP
    src = os.getenv("CAM_URL", "0")
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Камера недоступна", flush=True); return

    dets = init_detectors(device, weights_dir)
    s_time = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue

            metrics = calculate_image_metrics(frame)
            profile = analyze_scene_profile(metrics)
            scores = compute_detector_scores(metrics)
            best = max(scores, key=scores.get)

            detector = dets.get(best)
            net = dets["ssd"] if best == "ssd" else None
            t0 = time.time()
            boxes = detect_faces(frame, best, detector=detector, net=net)
            dt = (time.time() - t0) * 1000
            frames += 1
            fps = frames / max(1e-3, (time.time()-s_time))

            # Лог в консоль (для API-обёртки)
            print(f"[SMART] best={best} profile='{profile}' boxes={len(boxes) if boxes is not None else 0} "
                  f"infer={dt:.1f}ms fps={fps:.1f}", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("⚠️ Статус: Оптимизация завершена пользователем!", flush=True)

if __name__ == "__main__":
    from pathlib import Path
    main()
