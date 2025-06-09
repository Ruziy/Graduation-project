import os
import cv2
import numpy as np
import torch
import tensorflow as tf
import time
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from face_detection import RetinaFace
import mediapipe as mp
import dlib


# Подавление логов TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Нормализация метрик
def normalize(val, min_val, max_val):
    return max(0.0, min(1.0, (val - min_val) / (max_val - min_val + 1e-5)))


def calculate_image_metrics(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = hsv[..., 2].mean()
    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.mean(np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0)))

    metrics = {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise": noise
    }
    return metrics

# Профили искажений и веса для детекторов
def analyze_scene_profile(metrics):
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    sharpness = metrics["sharpness"]
    noise = metrics["noise"]

    profile = ""
    if brightness < 40:
        profile += "dark "
    elif brightness > 180:
        profile += "overexposed "
    if contrast < 30:
        profile += "flat "
    elif contrast > 100:
        profile += "high-contrast "
    if sharpness < 50:
        profile += "blurry "
    if noise > 50:
        profile += "noisy "


    return profile.strip()

def normalize(value, min_val, max_val):
    """Нормализация значения в диапазон [0, 1] с защитой от выхода за пределы"""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def compute_detector_scores(metrics):
    norm = {
        metric: normalize(metrics[metric], *limits)
        for metric, limits in {
            "brightness": (0, 255),
            "contrast": (0, 100),
            "sharpness": (0, 500),
            "noise": (0, 100)
        }.items()
    }

    rotate_base = {
    "yolov8":      0.9,
    "retinaface":  1.0,
    "insightface": 0.8,
    "mtcnn":       0.3,
    "mediapipe":   0.5,
    "ssd":         0.4,
    "dlib":        0.2,
    "haarcascade": 0.5
}

    # Новые веса устойчивости по результатам:
    # weights = {
    #     "yolov8":      1.3 * norm["contrast"] + 1.3 * norm["brightness"] - 0.7 * norm["noise"] + 1.2 * norm["sharpness"] + rotate_base["yolov8"],
    #     "retinaface":  1.2 * norm["contrast"] + 1.2 * norm["brightness"] - 0.4 * norm["noise"] + 1.2 * norm["sharpness"] + rotate_base["retinaface"],
    #     "insightface": 1.1 * norm["contrast"] + 1.1 * norm["brightness"] - 0.2 * norm["noise"] + 1.0 * norm["sharpness"] + rotate_base["insightface"],
    #     "mtcnn":       0.5 * norm["contrast"] + 0.5 * norm["brightness"] - 0.7 * norm["noise"] + 0.7 * norm["sharpness"] + rotate_base["mtcnn"],
    #     "mediapipe":   0.3 * norm["contrast"] + 0.9 * norm["brightness"] - 1.2 * norm["noise"] + 0.5 * norm["sharpness"] + rotate_base["mediapipe"],
    #     "ssd":         0.4 * norm["contrast"] + 0.4 * norm["brightness"] - 0.8 * norm["noise"] + 0.7 * norm["sharpness"] + rotate_base["ssd"],
    #     "dlib":        0.4 * norm["contrast"] + 0.4 * norm["brightness"] - 1.0 * norm["noise"] + 0.4 * norm["sharpness"] + rotate_base["dlib"],
    #     "haarcascade": 0.7 * norm["contrast"] + 0.7 * norm["brightness"] - 0.5 * norm["noise"] + 0.8 * norm["sharpness"] + rotate_base["haarcascade"]
    # }
    # Усиленные коэффициенты — контраст, яркость, резкость и шум влияют сильнее
    weights = {
        "yolov8":      3.9 * norm["contrast"] + 3.9 * norm["brightness"] - 2.1 * norm["noise"] + 3.6 * norm["sharpness"] + rotate_base["yolov8"],
        "retinaface":  3.6 * norm["contrast"] + 3.6 * norm["brightness"] - 1.2 * norm["noise"] + 3.6 * norm["sharpness"] + rotate_base["retinaface"],
        "insightface": 3.3 * norm["contrast"] + 3.3 * norm["brightness"] - 0.6 * norm["noise"] + 3.0 * norm["sharpness"] + rotate_base["insightface"],
        "mtcnn":       1.5 * norm["contrast"] + 1.5 * norm["brightness"] - 2.1 * norm["noise"] + 2.1 * norm["sharpness"] + rotate_base["mtcnn"],
        "mediapipe":   0.9 * norm["contrast"] + 2.7 * norm["brightness"] - 3.6 * norm["noise"] + 1.5 * norm["sharpness"] + rotate_base["mediapipe"],
        "ssd":         1.2 * norm["contrast"] + 1.2 * norm["brightness"] - 2.4 * norm["noise"] + 2.1 * norm["sharpness"] + rotate_base["ssd"],
        "dlib":        1.2 * norm["contrast"] + 1.2 * norm["brightness"] - 3.0 * norm["noise"] + 1.2 * norm["sharpness"] + rotate_base["dlib"],
        "haarcascade": 2.1 * norm["contrast"] + 2.1 * norm["brightness"] - 1.5 * norm["noise"] + 2.4 * norm["sharpness"] + rotate_base["haarcascade"]
    }

    return weights


def select_best_detector(metrics):
    profile = analyze_scene_profile(metrics)
    scores = compute_detector_scores(metrics)
    best_model = max(scores, key=scores.get)
    return best_model, profile, scores

# Обёртка над всеми детекторами
def detect_faces(frame, model_name, detector=None, net=None, conf=0.5):
    h, w = frame.shape[:2]

    if model_name == "yolov8":
        results = detector(frame)
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
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b, lm, score in faces if score > conf]

    elif model_name == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        boxes = []
        if results.detections:
            for d in results.detections:
                bbox = d.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                w_box, h_box = int(bbox.width * w), int(bbox.height * h)
                boxes.append([x1, y1, x1 + w_box, y1 + h_box])
        return boxes

    elif model_name == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int").tolist())
        return boxes

    return []

# Инициализация
def init_detectors(device):
    return {
        "yolov8": YOLO(r"weights/yolov8n-face.pt"),
        "mtcnn": MTCNN(keep_all=True, device=device),
        "retinaface": RetinaFace(),
        "mediapipe": mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5),
        "dlib": dlib.get_frontal_face_detector(),
        "insightface": FaceAnalysis(providers=['CUDAExecutionProvider']),
        "ssd": cv2.dnn.readNetFromCaffe(
            r"weights/deploy.prototxt.txt",
            r"weights/res10_300x300_ssd_iter_140000.caffemodel"
        )
    }

# Главная функция
def main():
    import sys
    print(sys.stdout.encoding)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detectors = init_detectors(device)
    if 'insightface' in detectors:
        detectors['insightface'].prepare(ctx_id=0)

    cap = cv2.VideoCapture(0)
    global profile
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        metrics = calculate_image_metrics(frame)
        best_model, profile, scores = select_best_detector(metrics)

        detector = detectors.get(best_model, None)
        net = detectors['ssd'] if best_model == 'ssd' else None

        faces = detect_faces(frame, best_model, detector, net)

        # Рисуем
        if faces is not None:
            for box in faces:
                if box is None: continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Визуализация
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        info = f"{best_model.upper()} | Profile: {profile} | Time: {elapsed*1000:.1f} ms | FPS: {fps:.1f}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        y_offset = 50
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            text = f"{name}: {score:.2f}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_offset += 20

        cv2.imshow("Smart Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
