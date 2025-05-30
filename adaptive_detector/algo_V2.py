import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from face_detection import RetinaFace
import mediapipe as mp
import dlib

# Подавление логов
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Метрики качества изображения ---
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
    return calculate_sharpness(frame)

# --- Выбор лучшего детектора по условиям ---
def select_best_detector(brightness, contrast, sharpness, noise_level):
    """
    Основано на анализе распознавания:
    - YOLOv8 лучше всего справляется с контрастом и засветами.
    - RetinaFace устойчив к блюру.
    - InsightFace — универсальный, но уступает при шумах.
    - Dlib и Haar — слабые, не использовать.
    """

    if contrast > 60 and brightness > 80:
        return 'yolov8'
    elif sharpness < 60 or noise_level > 15:
        return 'retinaface'
    elif brightness < 50 or contrast < 30:
        return 'insightface'
    elif sharpness > 100 and noise_level < 10:
        return 'mtcnn'
    else:
        return 'mediapipe'  # универсальный fallback

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
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                boxes.append([x, y, x + width, y + height])
        return boxes

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
        "ssd": cv2.dnn.readNetFromCaffe(
            r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\deploy.prototxt.txt",
            r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\res10_300x300_ssd_iter_140000.caffemodel"
        )
    }

# --- Основной цикл ---
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

        # Расчёт метрик
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        sharpness = calculate_sharpness(frame)
        noise_level = calculate_noise(frame)

        # Выбор детектора
        selected_model = select_best_detector(brightness, contrast, sharpness, noise_level)

        detector = detectors.get(selected_model, None)
        net = detectors['ssd'] if selected_model == 'ssd' else None

        boxes = detect_faces(frame, selected_model, detector=detector, net=net)

        # Рисуем лица
        if boxes is not None:
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Отладочная информация
        info = f"{selected_model.upper()} | Bright: {brightness:.1f} | Contrast: {contrast:.1f} | Noise: {noise_level:.1f}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Face Detection (Smart)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
