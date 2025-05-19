import os
import random
import cv2
import numpy as np
from pathlib import Path

# Функции для фильтров

def rotate_image(image):
    angle = random.randint(-180, 180)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated

def blur_image(image):
    return cv2.GaussianBlur(image, (21, 21), 0)

def add_noise(image):
    noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def adjust_brightness_contrast(image):
    alpha = 2.5  # Контраст (1.0 - без изменений)
    beta = 50   # Яркость (0 - без изменений)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def apply_filters(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        if not os.path.isfile(input_path):
            continue
        image = cv2.imread(input_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {input_path}")
            continue
        # Случайный выбор фильтра
        filter_func = random.choice([rotate_image, blur_image, add_noise, adjust_brightness_contrast])
        filter_func = adjust_brightness_contrast
        processed_image = filter_func(image)
        cv2.imwrite(output_path, processed_image)
        print(f"Обработано: {output_path}")

# Пример использования
apply_filters(r"C:\Users\Alex\Desktop\diplom\Graduation-project\speed_test\data\random_5000", r"C:\Users\Alex\Desktop\diplom\Graduation-project\speed_test\data\random_5000_adjust_brightness_contrast")
