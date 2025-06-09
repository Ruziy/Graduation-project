import cv2
import time
import psutil
import argparse
import torch
import dlib
import platform

from ultralytics import YOLO
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from face_detection import RetinaFace
import mediapipe as mp
from ultralytics.utils import LOGGER
import logging

# Отключить все логи ниже ERROR
LOGGER.setLevel(logging.ERROR)
def measure_cpu_usage(interval=1.0):
    """Измеряет загрузку CPU за указанное время (в процентах)"""
    return psutil.cpu_percent(interval=interval)

def get_lighter_detector(current):
    hierarchy = [
        "insightface",  # максимально тяжёлая
        "retinaface",    
        "yolov8",
        "mtcnn",
        "ssd",
        "dlib",
        "mediapipe",
        "haarcascade"   # минимально тяжёлая
    ]
    try:
        idx = hierarchy.index(current)
        return hierarchy[idx + 1] if idx + 1 < len(hierarchy) else current
    except ValueError:
        return current


def detect_faces(frame, model, detector=None, net=None):
    """Основная функция детекции лиц"""
    boxes = []

    if model == "yolov8":
        results = detector(frame)[0]
        for result in results.boxes.xyxy.cpu().numpy():
            boxes.append(result)
    elif model == "mtcnn":
        boxes_tensor, _ = detector.detect(frame)
        if boxes_tensor is not None:
            boxes.extend(boxes_tensor)
    elif model == "insightface":
        faces = detector.get(frame)
        for face in faces:
            boxes.append(face.bbox)
    elif model == "retinaface":
        # Используем RetinaFace для обнаружения лиц
        faces = detector(frame)
        boxes = []
        for box, landmarks, score in faces:
            x1, y1, x2, y2 = [int(v) for v in box]
            boxes.append([x1, y1, x2, y2])

    elif model == "mediapipe":
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)
                boxes.append((x1, y1, x2, y2))
    elif model == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                boxes.append(box.astype(int))
    elif model == "dlib":
        dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for d in dets:
            boxes.append([d.left(), d.top(), d.right(), d.bottom()])
    elif model == "haarcascade":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
    return boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8",
                        choices=["yolov8", "insightface", "retinaface", "mtcnn", "mediapipe", "ssd", "dlib", "haarcascade"])
    parser.add_argument("--fps_threshold", type=float, default=10.0,
                    help="Порог загрузки FPS  для смены модели")
    parser.add_argument("--cpu_threshold", type=float, default=50.0,
                    help="Порог загрузки CPU (%) для смены модели")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        print("Определён CUDA!! Обработка будет через GPU")
    cap = cv2.VideoCapture(0)

    # Инициализация модели
    detector, net = None, None
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
    elif args.model == "ssd":
        net = cv2.dnn.readNetFromCaffe(r"weights/deploy.prototxt.txt",
                                       r"weights/res10_300x300_ssd_iter_140000.caffemodel")

    prev_time = time.time()
    start_time = time.time()
    frame_count = 0
    avg_fps = 0
    calibrated = False
    cpu_usages = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.model == "ssd":
            boxes = detect_faces(frame, args.model, net=net)
        else:
            boxes = detect_faces(frame, args.model, detector)

        if boxes is not None:
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        frame_count += 1
        cpu_usages.append(measure_cpu_usage(interval=0))

        avg_cpu_now = cpu_usages[-1] if cpu_usages else 0
        cv2.putText(frame, f"Model: {args.model}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"CPU: {avg_cpu_now:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f"Face Detection - {args.model}", frame)

        if not calibrated and (current_time - start_time) > 20:
            avg_fps = frame_count / (current_time - start_time)
            avg_cpu = sum(cpu_usages) / len(cpu_usages)

            # Сбор логов в список
            calibration_log = []
            calibration_log.append(f"🔧 Калибровка завершена: модель {args.model}")
            calibration_log.append(f"📊 Средний FPS: {avg_fps:.2f}")
            calibration_log.append(f"💻 Средняя загрузка CPU: {avg_cpu:.1f}%")

            if avg_fps < args.fps_threshold or avg_cpu > args.cpu_threshold:
                calibration_log.append(
                    f"⚠️ Обнаружена высокая нагрузка (FPS < {args.fps_threshold} или CPU > {args.cpu_threshold}%)! Переключаемся на более лёгкий детектор."
                )
                args.model = get_lighter_detector(args.model)

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
                    net = cv2.dnn.readNetFromCaffe(
                        r"weights/deploy.prototxt.txt",
                        r"weights/res10_300x300_ssd_iter_140000.caffemodel"
                    )

                # Сброс счётчиков
                start_time = time.time()
                frame_count = 0
                cpu_usages = []
                calibrated = False

            else:
                calibrated = True
                calibration_log.append("✅ Калибровка прошла успешно. Требования к FPS и CPU соблюдены.")
                cap.release()
                cv2.destroyAllWindows()
                calibration_result = "\n".join(calibration_log)
                print(calibration_result)
                response = {"status": "Калибровка завершена", "output": calibration_result}
                exit(0)

            # Вывод логов в консоль и в ответ
            calibration_result = "\n".join(calibration_log)
            print(calibration_result)
            response = {"status": "Калибровка завершена", "output": calibration_result}



        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
