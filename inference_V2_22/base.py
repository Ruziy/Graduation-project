import cv2
import torch
import argparse
import numpy as np
import os
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
    h, w, _ = frame.shape  # Получаем размеры изображения

    if model_name == "yolov8":
        # Используем YOLOv8 для обнаружения лиц
        results = detector(frame, device=device)
        return [r.boxes.xyxy.cpu().numpy() for r in results][0]
    
    elif model_name == "dlib":
        # Используем Dlib для обнаружения лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        boxes = []
        for face in faces:
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            boxes.append([x1, y1, x2, y2])
        return boxes

    elif model_name == "mtcnn":
        # Используем MTCNN для обнаружения лиц
        boxes, _ = detector.detect(frame)
        return boxes

    elif model_name == "insightface":
        # Используем InsightFace для обнаружения лиц
        faces = detector.get(frame)
        return [f.bbox for f in faces]

    elif model_name == "retinaface":
        # Используем RetinaFace для обнаружения лиц
        faces = detector(frame)
        boxes = []
        for box, landmarks, score in faces:
            if score < confidence_threshold:
                continue  # Пропустить слабые детекции

            x1, y1, x2, y2 = [int(v) for v in box]
            boxes.append([x1, y1, x2, y2])
        return boxes 

    elif model_name == "mediapipe":
        # Используем MediaPipe для обнаружения лиц
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        boxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w_, h_ = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                boxes.append([x, y, x + w_, y + h_])
        return boxes

    elif model_name == "haarcascade":
        # Используем Haar Cascade для обнаружения лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        return boxes
    
    elif model_name == "ssd":
        # Используем SSD для обнаружения лиц
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append([startX, startY, endX, endY])
        return boxes

    return []  # Если модель не распознана, возвращаем пустой список


import time  # Добавьте в список импортов

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=["yolov8", "mtcnn", "insightface", "retinaface", "mediapipe", "dlib", "haarcascade", "ssd"])
    

    args = parser.parse_args()
    args.model = args.model.lower()
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'  # Принудительно CPU

    if device == 'cpu':
        print("Процесс детекции будет реализован на CPU, в связи с этим детекторы UNKNOWN будут недоступны!")
        print("Постарайтесь решить вопросы с зависимостями GPU/CUDA, чтобы пользоваться всем функционалом.")
    else:
        print("Процесс детекции будет реализован на GPU!")

    cap = cv2.VideoCapture(0)

    # Выбор модели
    if args.model == "yolov8":
        detector = YOLO(r"weights/yolov8n-face.pt")
    elif args.model == "mtcnn":
        detector = MTCNN(keep_all=True, device=device)
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
    elif args.model == "haarcascade":
        detector = None
    elif args.model == "ssd":
        path_prototxt = r"weights/deploy.prototxt.txt"
        path_model_weights = r"weights/res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model_weights)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение лиц
        if args.model == "ssd":
            boxes = detect_faces(frame, args.model, net=net)
        else:
            boxes = detect_faces(frame, args.model, detector)

        # Отрисовка прямоугольников
        if boxes is not None:
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Расчёт и отображение FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Показ кадра
        cv2.imshow(f"Face Detection - {args.model}", frame)

        import sys

        # ...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Выход по клавише q")
            sys.exit(0)  # Без ошибки


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
