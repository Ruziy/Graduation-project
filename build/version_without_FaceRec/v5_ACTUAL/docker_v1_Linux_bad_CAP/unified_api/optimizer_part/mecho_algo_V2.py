# mecho_algo_V2.py (headless calibration)
import os, sys, time, argparse
import cv2, torch, dlib, mediapipe as mp
import psutil
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
try:
    from face_detection import RetinaFace
except Exception:
    from retinaface import RetinaFace

def measure_cpu_usage(interval=0.0):
    return psutil.cpu_percent(interval=interval)

def get_lighter_detector(current):
    order = ["insightface","retinaface","yolov8","mtcnn","ssd","dlib","mediapipe","haarcascade"]
    try:
        i = order.index(current)
        return order[i+1] if i+1 < len(order) else current
    except ValueError:
        return current

def detect_faces(frame, model, detector=None, net=None):
    import numpy as np
    h, w = frame.shape[:2]
    boxes = []
    if model == "yolov8":
        res = detector(frame)[0]
        for b in res.boxes.xyxy.cpu().numpy():
            boxes.append(b)
    elif model == "mtcnn":
        b, _ = detector.detect(frame)
        if b is not None: boxes.extend(b)
    elif model == "insightface":
        for f in detector.get(frame):
            boxes.append(f.bbox)
    elif model == "retinaface":
        faces = detector(frame)
        for box, lm, score in faces:
            x1,y1,x2,y2 = [int(v) for v in box]
            boxes.append([x1,y1,x2,y2])
    elif model == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = detector.process(rgb)
        if r.detections:
            for d in r.detections:
                bb = d.location_data.relative_bounding_box
                x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
                ww,hh = int(bb.width*w), int(bb.height*h)
                boxes.append([x1,y1,x1+ww,y1+hh])
    elif model == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123))
        net.setInput(blob)
        dets = net.forward()
        for i in range(dets.shape[2]):
            c = dets[0,0,i,2]
            if c > 0.5:
                x1,y1,x2,y2 = (dets[0,0,i,3:7] * [w,h,w,h]).astype(int)
                boxes.append([x1,y1,x2,y2])
    elif model == "dlib":
        for d in detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):
            boxes.append([d.left(), d.top(), d.right(), d.bottom()])
    elif model == "haarcascade":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        for (x,y,ww,hh) in face_cascade.detectMultiScale(gray,1.1,5):
            boxes.append([x,y,x+ww,y+hh])
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolov8",
        choices=["yolov8","insightface","retinaface","mtcnn","mediapipe","ssd","dlib","haarcascade"])
    ap.add_argument("--fps_threshold", type=float, default=10.0)
    ap.add_argument("--cpu_threshold", type=float, default=50.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = os.getenv("CAM_URL", "0")
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Камера недоступна", flush=True); sys.exit(1)

    detector, net = None, None
    if args.model == "yolov8":
        detector = YOLO("weights/yolov8n-face.pt")
    elif args.model == "mtcnn":
        detector = MTCNN(keep_all=True, device=device)
    elif args.model == "insightface":
        detector = FaceAnalysis(providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        detector.prepare(ctx_id=(0 if device=="cuda" else -1))
    elif args.model == "retinaface":
        detector = RetinaFace()
    elif args.model == "mediapipe":
        detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    elif args.model == "dlib":
        detector = dlib.get_frontal_face_detector()
    elif args.model == "ssd":
        net = cv2.dnn.readNetFromCaffe("weights/deploy.prototxt.txt","weights/res10_300x300_ssd_iter_140000.caffemodel")

    start = time.time()
    last = start
    frame_count = 0
    cpus = []

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        if args.model == "ssd":
            _ = detect_faces(frame, args.model, net=net)
        else:
            _ = detect_faces(frame, args.model, detector=detector)

        frame_count += 1
        cpus.append(measure_cpu_usage(interval=0.0))

        now = time.time()
        if now - start > 20.0:
            avg_fps = frame_count / (now - start)
            avg_cpu = sum(cpus)/max(1, len(cpus))

            if avg_fps < args.fps_threshold or avg_cpu > args.cpu_threshold:
                print(f"[КАЛИБРОВКА] Активная модель: {args.model}", flush=True)
                print(f"⚠️ Нагрузка высокая (FPS<{args.fps_threshold} или CPU>{args.cpu_threshold}%) — переключаемся", flush=True)
                args.model = get_lighter_detector(args.model)
                # реинициализация
                return  # даём API перезапустить скрипт с обновлённой моделью при необходимости
            else:
                print("✅ Калибровка прошла успешно. Требования к FPS и CPU соблюдены.", flush=True)
                print(f"🔧 Калибровка завершена: модель {args.model}", flush=True)
                print(f"📊 Средний FPS: {avg_fps:.2f}", flush=True)
                print(f"💻 Средняя загрузка CPU: {avg_cpu:.1f}%", flush=True)
                break

    cap.release()

if __name__ == "__main__":
    main()
