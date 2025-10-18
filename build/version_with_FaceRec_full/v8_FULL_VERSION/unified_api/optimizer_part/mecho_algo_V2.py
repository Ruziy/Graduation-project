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

LOGGER.setLevel(logging.ERROR)

def measure_cpu_usage(interval=1.0):
    return psutil.cpu_percent(interval=interval)

def get_lighter_detector(current):
    hierarchy = [
        "insightface",
        "retinaface",
        "yolov8",
        "mtcnn",
        "ssd",
        "dlib",
        "mediapipe",
        "haarcascade"
    ]
    try:
        idx = hierarchy.index(current)
        return hierarchy[idx + 1] if idx + 1 < len(hierarchy) else current
    except ValueError:
        return current

def detect_faces(frame, model, detector=None, net=None, thresholds=None):
    boxes = []
    th = 0.0
    if thresholds:
        th = thresholds.get(model, 0.0)

    if model == "yolov8":
        results = detector(frame)[0]
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, "conf") else [1.0] * len(xyxy)
        for b, c in zip(xyxy, conf):
            if c >= th:
                boxes.append(b)

    elif model == "mtcnn":
        boxes_tensor, probs = detector.detect(frame)
        if boxes_tensor is not None:
            for b, p in zip(boxes_tensor, probs):
                if p is None or p >= th:
                    boxes.append(b)

    elif model == "insightface":
        faces = detector.get(frame)
        for face in faces:
            score = getattr(face, "det_score", 1.0)
            if score is None or score >= th:
                boxes.append(face.bbox)

    elif model == "retinaface":
        faces = detector(frame)
        for box, landmarks, score in faces:
            if score >= th:
                x1, y1, x2, y2 = [int(v) for v in box]
                boxes.append([x1, y1, x2, y2])

    elif model == "mediapipe":
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                score = detection.score[0] if detection.score else 1.0
                if score >= th:
                    bboxC = detection.location_data.relative_bounding_box
                    x1 = int(bboxC.xmin * w)
                    y1 = int(bboxC.ymin * h)
                    x2 = int((bboxC.xmin + bboxC.width) * w)
                    y2 = int((bboxC.ymin + bboxC.height) * h)
                    boxes.append((x1, y1, x2, y2))

    elif model == "ssd":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence >= th:
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
    parser.add_argument("--fps_threshold", type=float, default=10.0, help="Порог FPS для смены модели")
    parser.add_argument("--cpu_threshold", type=float, default=50.0, help="Порог загрузки CPU (%) для смены модели")
    parser.add_argument("--window_sec", type=float, default=20.0, help="Окно оценки в секундах для каждой модели")

    parser.add_argument("--th_yolo", type=float, default=0.25, help="Порог уверенности для YOLOv8")
    parser.add_argument("--th_retinaface", type=float, default=0.80, help="Порог уверенности для RetinaFace")
    parser.add_argument("--th_mtcnn", type=float, default=0.90, help="Порог уверенности для MTCNN")
    parser.add_argument("--th_insightface", type=float, default=0.60, help="Порог уверенности для InsightFace")
    parser.add_argument("--th_ssd", type=float, default=0.50, help="Порог уверенности для SSD")
    parser.add_argument("--th_mediapipe", type=float, default=0.50, help="Минимальная уверенность для MediaPipe")
    args = parser.parse_args()

    thresholds = {
        "yolov8": args.th_yolo,
        "retinaface": args.th_retinaface,
        "mtcnn": args.th_mtcnn,
        "insightface": args.th_insightface,
        "ssd": args.th_ssd,
        "mediapipe": args.th_mediapipe
    }

    def load_detector(name, device):
        det, dnn = None, None
        if name == "yolov8":
            det = YOLO(r"weights/yolov8n-face.pt")
        elif name == "mtcnn":
            det = MTCNN(keep_all=True, device=device)
        elif name == "insightface":
            det = FaceAnalysis(providers=['CUDAExecutionProvider'] if device == "cuda" else None)
            det.prepare(ctx_id=0 if device == "cuda" else -1)
        elif name == "retinaface":
            det = RetinaFace()
        elif name == "mediapipe":
            mp_face = mp.solutions.face_detection
            det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=args.th_mediapipe)
        elif name == "dlib":
            det = dlib.get_frontal_face_detector()
        elif name == "ssd":
            dnn = cv2.dnn.readNetFromCaffe(
                r"weights/deploy.prototxt.txt",
                r"weights/res10_300x300_ssd_iter_140000.caffemodel"
            )
        elif name == "haarcascade":
            det = None
        return det, dnn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        print("Обнаружен CUDA. Будет использовано ускорение GPU.")
    cap = cv2.VideoCapture(0)

    models_order = ["yolov8", "insightface", "retinaface", "mtcnn", "mediapipe", "ssd", "dlib", "haarcascade"]
    current_index = models_order.index(args.model)
    last_index = len(models_order) - 1

    last_announced_model = None
    def announce_model(m):
        nonlocal last_announced_model
        if m != last_announced_model:
            print(f"🧪 Активная модель: {m}")
            last_announced_model = m

    def switch_to(index):
        m = models_order[index]
        det, dnn = load_detector(m, device)
        announce_model(m)
        return m, det, dnn, time.time(), 0, []

    args.model, detector, net, start_time, frame_count, cpu_usages = switch_to(current_index)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.model == "ssd":
            boxes = detect_faces(frame, args.model, detector=None, net=net, thresholds=thresholds)
        else:
            boxes = detect_faces(frame, args.model, detector=detector, net=None, thresholds=thresholds)

        if boxes is not None:
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        frame_count += 1
        cpu_usages.append(measure_cpu_usage(interval=0))
        cpu_now = cpu_usages[-1] if cpu_usages else 0

        cv2.putText(frame, f"Model: {args.model}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"CPU: {cpu_now:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f"Face Detection - {args.model}", frame)

        if (now - start_time) >= args.window_sec:
            avg_fps = frame_count / (now - start_time) if (now - start_time) > 0 else 0.0
            avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0
            log_lines = []

            if avg_fps >= args.fps_threshold and avg_cpu <= args.cpu_threshold:
                log_lines.append("✅ Калибровка завершена успешно. Текущая конфигурация соответствует заданным условиям.")
                log_lines.append(f"🤖 Подходящая модель: {args.model}")
                log_lines.append(f"📊 Средний FPS: {avg_fps:.2f}")
                log_lines.append(f"💻 Средняя загрузка CPU: {avg_cpu:.1f}%")
                cap.release()
                cv2.destroyAllWindows()
                print("\n".join(log_lines))
                response = {"status": "Калибровка завершена", "output": "\n".join(log_lines)}
                exit(0)
            else:
                if current_index < last_index:
                    log_lines.append(f"⚠️ Параметры пока не достигнуты (FPS ≥ {args.fps_threshold}, CPU ≤ {args.cpu_threshold}%). Перейдём к более лёгкой модели.")
                    print("\n".join(log_lines))
                    current_index += 1
                    args.model, detector, net, start_time, frame_count, cpu_usages = switch_to(current_index)
                else:
                    log_lines.append(f"❌ По текущим ограничениям (FPS ≥ {args.fps_threshold}, CPU ≤ {args.cpu_threshold}%) подобрать модель не удалось.")
                    log_lines.append(f"📊 Средний FPS: {avg_fps:.2f}, 💻 Средняя загрузка CPU: {avg_cpu:.1f}%")
                    log_lines.append("📝 Рекомендуется ослабить требования или изменить параметры источника видео.")
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n".join(log_lines))
                    response = {"status": "Подходящих детекторов не найдено", "output": "\n".join(log_lines)}
                    exit(0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
