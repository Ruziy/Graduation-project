import cv2
import numpy as np

def calculate_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[...,2].mean()
    return brightness

def calculate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()  # стандартное отклонение
    return contrast

def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()  # дисперсия Лапласиана
    return sharpness

def calculate_noise(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = np.mean(np.abs(gray - blur))
    return noise

def calculate_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    brightness = calculate_brightness(frame)
    contrast = calculate_contrast(frame)
    sharpness = calculate_sharpness(frame)
    noise = calculate_noise(frame)
    blur = calculate_blur(frame)
    # Вывод информации на изображение
    text = f"Brightness: {brightness:.1f} | Contrast: {contrast:.1f} | | Noise: {noise:.1f} | Blur: {blur:.1f}"# Sharpness: {sharpness:.1f}
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Quality Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
