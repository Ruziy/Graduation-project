import cv2
import torch
import argparse
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from face_detection import RetinaFace
import mediapipe as mp
import dlib

# Загружаем классификатор для распознавания лиц (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame, model_name, detector=None, net=None, confidence_threshold=0.5):
    h, w, _ = frame.shape

    if model_name == "yolov8":
        results = detector(frame, device=0)
        return [r.boxes.xyxy.cpu().numpy() for r in results][0]
    elif model_name == "dlib":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        return [[face.left(), face.top(), face.right(), face.bottom()] for face in faces]
    elif model_name == "mtcnn":
        boxes, _ = detector.detect(frame)
        return boxes if boxes is not None else []
    elif model_name == "insightface":
        faces = detector.get(frame)
        return [f.bbox for f in faces]
    elif model_name == "retinaface":
        faces = detector(frame)
        return [[int(v) for v in box] for box, _, score in faces if score >= confidence_threshold]
    elif model_name == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        return [[int(bbox.xmin * w), int(bbox.ymin * h), int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)] for detection in results.detections] if results.detections else []
    elif model_name == "haarcascade":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [[x, y, x + w, y + h] for (x, y, w, h) in faces]
    elif model_name == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        return [[int(d[3] * w), int(d[4] * h), int(d[5] * w), int(d[6] * h)] for d in detections[0, 0] if d[2] > confidence_threshold]
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=["yolov8", "mtcnn", "insightface", "retinaface", "mediapipe", "dlib", "haarcascade", "ssd"], help="Выберите модель для распознавания лиц")
    args = parser.parse_args()
    args.input_dir = r"C:\Users\Alex\Desktop\diplom\Graduation-project\speed_test\data\random_5000"

    if args.model == "yolov8":
        detector = YOLO(r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\yolov8n-face.pt")
    elif args.model == "mtcnn":
        detector = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    elif args.model == "insightface":
        detector = FaceAnalysis(providers=['CUDAExecutionProvider'])
        detector.prepare(ctx_id=0)
    elif args.model == "retinaface":
        detector = RetinaFace()
    elif args.model == "mediapipe":
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    elif args.model == "dlib":
        detector = dlib.get_frontal_face_detector()
    elif args.model == "ssd":
        prototxt = r"C:\Users\Alex\Desktop\diplom\Graduation-project\CAFFE_DNN\deploy.prototxt.txt"
        weights = r"C:\Users\Alex\Desktop\diplom\Graduation-project\CAFFE_DNN\res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    else:
        detector = None

    start_time = time.time()
    total_faces = 0
    image_count = 0

    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_count += 1
            img_path = os.path.join(args.input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {filename}")
                continue
            boxes = detect_faces(img, args.model, detector=detector, net=net if args.model == "ssd" else None)
            if boxes is not None:
                total_faces += len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.input_dir, f"processed_{filename}"), img)

    elapsed_time = time.time() - start_time
    print(f"Обработано {image_count} изображений. Найдено {total_faces} лиц. Время обработки: {elapsed_time:.2f} секунд.")

if __name__ == "__main__":
    main()
