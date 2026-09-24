# Веса

| Файл | Модель | Источник | Лицензия | Назначение |
|---|---|---|---|---|
| yolo11s.pt | YOLO11-small, COCO | Ultralytics assets v8.3.0 | AGPL-3.0 | детектор для Part A (события) |
| yolo11n.pt | YOLO11-nano, COCO | Ultralytics assets v8.3.0 | AGPL-3.0 | детектор для Part B (риск, каждый кадр) и CPU-демо |

Ничего не дообучалось: используются оригинальные COCO-веса. `download.sh` восстанавливает файлы, если их нет.
