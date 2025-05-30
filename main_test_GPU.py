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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame, model_name, detector=None, net=None, confidence_threshold=0.5):
    h, w, _ = frame.shape

    if model_name == "yolov8":
        results = detector(frame, device=0)
        return [r.boxes.xyxy.cpu().numpy() for r in results][0]
    
    elif model_name == "dlib":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        boxes = []
        for face in faces:
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            boxes.append([x1, y1, x2, y2])
        return boxes

    elif model_name == "mtcnn":
        boxes, _ = detector.detect(frame)
        return boxes

    elif model_name == "insightface":
        faces = detector.get(frame)
        return [f.bbox for f in faces]

    elif model_name == "retinaface":
        faces = detector(frame)
        boxes = []
        for box, landmarks, score in faces:
            if score < confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            boxes.append([x1, y1, x2, y2])
        return boxes 

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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        return boxes
    
    elif model_name == "ssd":
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

    return []

def print_device_info(model_name):
    # Общая логика по выводу устройства
    print(f"\nДетектор: {model_name}")
    if model_name in ["yolov8", "mtcnn"]:
        print("Фреймворк: PyTorch")
        if torch.cuda.is_available():
            print("CUDA доступен. Детектор будет работать на GPU.")
        else:
            print("CUDA недоступен. Детектор будет работать на CPU.")
    elif model_name == "insightface":
        print("Фреймворк: InsightFace (ONNXRuntime)")
        try:
            providers = FaceAnalysis().get_providers()
            if "CUDAExecutionProvider" in providers:
                print("CUDA доступен. Детектор будет работать на GPU.")
            else:
                print("CUDA недоступен. Используется CPU.")
        except:
            print("Не удалось определить CUDA-поддержку. Используется CPU.")
    elif model_name == "retinaface":
        print("Фреймворк: RetinaFace (PyTorch/NumPy)")
        if torch.cuda.is_available():
            print("CUDA доступен. Возможна работа на GPU.")
        else:
            print("CUDA недоступен. Работает на CPU.")
    elif model_name == "mediapipe":
        print("Фреймворк: MediaPipe (TensorFlow)")
        if tf.config.list_physical_devices('GPU'):
            print("GPU доступен для TensorFlow.")
        else:
            print("GPU недоступен. MediaPipe будет использовать CPU.")
    elif model_name == "dlib":
        print("Фреймворк: Dlib")
        print("Работает на CPU. Поддержка GPU ограничена.")
    elif model_name == "haarcascade":
        print("Фреймворк: OpenCV (Haar Cascade)")
        print("Работает исключительно на CPU.")
    elif model_name == "ssd":
        print("Детектор: ssd")
        print("Фреймворк: OpenCV DNN (Caffe model)")

        try:
            net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model_weights)

            # Пробуем установить backend и target на CUDA
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            # Если ошибки нет — CUDA доступна
            print("CUDA backend доступен и используется для OpenCV DNN.")
        except Exception as e:
            print("CUDA backend недоступен. Используется CPU.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("=" * 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=["yolov8", "mtcnn", "insightface", "retinaface", "mediapipe", "dlib", "haarcascade", "ssd"])
    args = parser.parse_args()

    # Печать информации об устройстве и фреймворке
    print_device_info(args.model)

    cap = cv2.VideoCapture(0)

    if args.model == "yolov8":
        detector = YOLO(r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\yolov8n-face.pt")

    elif args.model == "mtcnn":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        global path_prototxt
        global path_model_weights
        path_prototxt = r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\deploy.prototxt.txt"
        path_model_weights = r"C:\Users\Alex\Desktop\diplom\Graduation-project\weights\res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model_weights)

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

        cv2.imshow(f"Face Detection - {args.model}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
