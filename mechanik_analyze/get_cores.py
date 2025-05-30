import os
import platform
import psutil

try:
    import torch
    cuda_available = torch.cuda.is_available()
except ImportError:
    cuda_available = False

def get_cpu_info():
    print("="*50)
    print("🔍 Информация о системе:")
    print(f"🖥️ Процессор: {platform.processor()}")
    print(f"💡 Платформа: {platform.system()} {platform.release()}")
    
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    print(f"🧠 Физических ядер: {physical_cores}")
    print(f"🧵 Логических потоков: {logical_cores}")
    print("="*50)

    print("🚀 Поддержка CUDA:")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA доступна — {gpu_name}")
    else:
        print("❌ CUDA не обнаружена")

    print("="*50)
    print("📌 Рекомендации по выбору детектора лиц:")

    if cuda_available:
        print("🟢 Рекомендуется использовать: YOLOv8, RetinaFace или MTCNN (GPU)")
        print("   • Использование GPU значительно ускоряет обработку")
        print("   • Особенно полезно при больших объёмах данных")
    else:
        if logical_cores >= 12:
            print("✅ Рекомендуется использовать: YOLOv8 (CPU)")
            print("   • Высокая точность и использование всех потоков")
        elif logical_cores >= 6:
            print("✅ Рекомендуется: InsightFace или RetinaFace")
            print("   • Хорошее качество и умеренная загрузка CPU")
        elif logical_cores >= 4:
            print("✅ Рекомендуется: MediaPipe")
            print("   • Лёгкий и быстрый, подходит для большинства систем")
        else:
            print("⚠️ Рекомендуется: Haarcascade")
            print("   • Минимальные требования к ресурсам, но низкая точность")

    print("="*50)

get_cpu_info()
