# Вторая версия ПО с блоком распознавания лиц
## Общие зависимости (необходимо предварительно установить):
1.Официальный установщик CMake. 

https://cmake.org/download/

Установите его, обязательно выбрав опцию "Add CMake to system PATH".

После убедитесь что все корректно работает:
```bash
cmake --version
```
2.Скачайте "Build Tools for Visual Studio".

https://visualstudio.microsoft.com/visual-cpp-build-tools/

Во время установки обязательно выберите:

```bash
C++ build tools
```
А также проверьте, что выбраны:
```bash
MSVC v14.x
Windows 10 SDK
CMake tools for Windows (может потребоваться для некоторых зависимостей)
```

Перезагрузите компьютер после завершения установки.

3.Выгрузите необходимые ENV.tar.gz и папку weights из google disk (https://drive.google.com/file/d/1leVnVAzmqxJCOMf5fjm5fQKKQdwqGv1M/view?usp=sharing), и поместите согласно структуре(описана ниже).

 Либо скачайте полностью архив.
 ## Запуск
```bash
Запустите run.bat из CMD терминала Windows (предварительно загрузив ENV.tar.gz и папку weights)
```

Открой в браузере: http://localhost:8000/



## Структура проекта:
```bash
C:
|   DIPLOM_PO_win-64.tar.gz
|
\---unified_api
    |   gateway.py
    |   README.md
    |   run.bat
    |
    +---faceRec_part
    |   |   app.py
    |   |   log.txt
    |   |
    |   +---known_faces
    |   \---static
    |           app.js
    |           index.html
    |           style.css
    |
    +---optimizer_part
    |   |   algo_V4_fps.py
    |   |   base.py
    |   |   main_V3.py
    |   |   mecho_algo_V2.py
    |   |
    |   +---static
    |   |       styles.css
    |   |
    |   +---templates
    |   |       index.html
    |   |
    |   \---weights
    |           deploy.prototxt.txt
    |           haarcascade_frontalface_default.xml
    |           res10_300x300_ssd_iter_140000.caffemodel
    |           yolov8n-face.pt
    |
    \---templates
            hub.html
```
