# Unified API (FastAPI + Uvicorn) 
# Первая версия ПО без блока распознавания лиц
Единая точка входа, объединяющая два твоих проекта:

- `optimizer_part/` — калибровка/оптимизация/выбор модели (твоя логика сохранена)


## Запуск
##### Общие зависимости:
1.официальный установщик CMake. https://cmake.org/download/
Установите его, обязательно выбрав опцию "Add CMake to system PATH".
cmake --version

2.https://visualstudio.microsoft.com/visual-cpp-build-tools/
Скачайте "Build Tools for Visual Studio".
Во время установки обязательно выберите:
C++ build tools
А также проверьте, что выбраны:
MSVC v14.x
Windows 10 SDK
CMake tools for Windows (может потребоваться для некоторых зависимостей)
Перезагрузите компьютер после завершения установки.

```bash
Запустите run.bat, предварительно загрузив ENV.tar.gz и папку weights из google disk (https://drive.google.com/file/d/1leVnVAzmqxJCOMf5fjm5fQKKQdwqGv1M/view?usp=sharing).
```

Открой в браузере: http://localhost:8000/

- `/optimizer/` — ТОЛЬКО часть с оптимизатором

