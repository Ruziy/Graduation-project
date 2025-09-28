# base.py (short headless run of a chosen model)
import os, sys, time, argparse
import cv2, torch, dlib, mediapipe as mp
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
try:
    from face_detection import RetinaFace
except Exception:
    from retinaface import RetinaFace
import numpy as np

def detect_faces(frame, model_name, detector=None, net=None, conf=0.5):
    h, w = frame.shape[:2]
    if model_name == "yolov8":
        res = detector(frame)[0]
        return [b for b in res.boxes.xyxy.cpu().numpy()]
    elif model_name == "mtcnn":
        b, _ = detector.detect(frame); return [] if b is None else b
    elif model_name == "insightface":
        return [f.bbox for f in detector.get(frame)]
    elif model_name == "retinaface":
        faces = detector(frame)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b,lm,sc in faces if sc >= conf]
    elif model_name == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = detector.process(rgb); out=[]
        if r.detections:
            for d in r.detections:
                bb = d.location_data.relative_bounding_box
                x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
                ww,hh = int(bb.width*w), int(bb.height*h)
                out.append([x1,y1,x1+ww,y1+hh])
        return out
    elif model_name == "dlib":
        return [[f.left(), f.top(), f.right(), f.bottom()] for f in detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))]
    elif model_name == "haarcascade":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        return [[x,y,x+w,y+h] for (x,y,w,h) in faces]
    elif model_name == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123))
        net.setInput(blob); dets = net.forward()
        out=[]
        for i in range(dets.shape[2]):
            c=dets[0,0,i,2]
            if c>=conf:
                box=(dets[0,0,i,3:7]*[w,h,w,h]).astype(int)
                out.append(box.tolist())
        return out
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
        choices=["yolov8","mtcnn","insightface","retinaface","mediapipe","dlib","haarcascade","ssd"])
    ap.add_argument("--duration", type=float, default=5.0)
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
    elif args.model == "haarcascade":
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    elif args.model == "ssd":
        net = cv2.dnn.readNetFromCaffe("weights/deploy.prototxt.txt","weights/res10_300x300_ssd_iter_140000.caffemodel")

    print(f"== Запуск модели: {args.model} ==", flush=True)
    start = time.time()
    frames = 0
    while time.time() - start < args.duration:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        if args.model == "ssd":
            boxes = detect_faces(frame, args.model, net=net)
        else:
            boxes = detect_faces(frame, args.model, detector=detector)
        frames += 1
    cap.release()
    print(f"Готово. Кадров обработано: {frames}", flush=True)

if __name__ == "__main__":
    main()
