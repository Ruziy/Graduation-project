# Unified API (FastAPI + Uvicorn)

Единая точка входа, объединяющая два твоих проекта:

- `optimizer_part/` — калибровка/оптимизация/выбор модели (твоя логика сохранена)
- `faceRec_part/` — распознавание лиц (твоя логика сохранена)

## Запуск

```bash
pip install fastapi uvicorn jinja2
# + зависимости твоих проектов (ultralytics, face_recognition, opencv-python и т.д.)

uvicorn gateway:app --host 0.0.0.0 --port 8000 --reload
```

Открой в браузере: http://localhost:8000/

- `/optimizer/` — первая часть (UI форм)  
- `/face/` — распознавание лиц

## Структура

```
unified_api/
  gateway.py
  templates/
    hub.html
  optimizer_part/
    main_V3.py
    base.py
    mecho_algo_V2.py
    algo_V4_fps.py
    templates/index.html
    static/styles.css
    ...
  faceRec_part/
    app.py
    settings.py
    static/{index.html, style.css, app.js}
    known_faces/...
    ...
```
