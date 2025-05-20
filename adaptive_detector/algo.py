import cv2
import numpy as np
import argparse
import torch
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

# Haar Cascade для резервного детектора
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + r'haarcascade_frontalface_default.xml')

# --- Метрики качества ---
def calculate_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

def calculate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.std()

def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_noise(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return np.mean(np.abs(gray - blur))

def calculate_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- Выбор детектора на основе анализа ---
def select_detector(brightness, contrast, sharpness, blur):
    if brightness < 60 or blur < 50:
        return 'retinaface'
    elif sharpness < 100:
        return 'haarcascade'
    elif contrast > 70:
        return 'yolov8'
    else:
        return 'mtcnn'  # по умолчанию

# --- Детекция лиц ---
def detect_faces(frame, model_name, detector=None, net=None, confidence_threshold=0.5):
    h, w, _ = frame.shape

    if model_name == "yolov8":
        results = detector(frame, device=0)
        return [r.boxes.xyxy.cpu().numpy() for r in results][0]

    elif model_name == "dlib":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        return [[f.left(), f.top(), f.right(), f.bottom()] for f in faces]

    elif model_name == "mtcnn":
        boxes, _ = detector.detect(frame)
        return boxes

    elif model_name == "insightface":
        faces = detector.get(frame)
        return [f.bbox for f in faces]

    elif model_name == "retinaface":
        faces = detector(frame)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b, lm, score in faces if score > confidence_threshold]

    elif model_name == "mediapipe":
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        return [[x, y, x + w, y + h] for (x, y, w, h) in faces]

    elif model_name == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int").tolist())
        return boxes

    return []

# --- Инициализация всех детекторов ---
def init_all_detectors(device):
    return {
        "yolov8": YOLO(r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\yolov8n-face.pt"),
        "mtcnn": MTCNN(keep_all=True, device=device),
        "retinaface": RetinaFace(),
        "mediapipe": mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5),
        "dlib": dlib.get_frontal_face_detector(),
        "insightface": FaceAnalysis(providers=['CUDAExecutionProvider']),
        "haarcascade": None,
        "ssd": cv2.dnn.readNetFromCaffe(r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\deploy.prototxt.txt", r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\res10_300x300_ssd_iter_140000.caffemodel")
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detectors = init_all_detectors(device)
    if 'insightface' in detectors:
        detectors['insightface'].prepare(ctx_id=0)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Анализ качества
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        sharpness = calculate_sharpness(frame)
        blur = calculate_blur(frame)

        selected_model = select_detector(brightness, contrast, sharpness, blur)

        # Получение подходящего детектора
        detector = detectors[selected_model] if selected_model not in ['ssd', 'haarcascade'] else None
        net = detectors['ssd'] if selected_model == 'ssd' else None

        boxes = detect_faces(frame, selected_model, detector=detector, net=net)

        # Отображение рамок
        if boxes is not None:
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Вывод информации
        text = f"{selected_model.upper()} | Brightness: {brightness:.1f} | Contrast: {contrast:.1f} | Blur: {blur:.1f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Face Detection (Auto)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
