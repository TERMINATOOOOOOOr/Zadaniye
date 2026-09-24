#!/usr/bin/env bash
# Веса детектора: YOLO11 (Ultralytics, лицензия AGPL-3.0). Файлы уже лежат в репозитории;
# скрипт нужен только чтобы восстановить их при отсутствии (запускать один раз с интернетом).
set -e
cd "$(dirname "$0")"
BASE="https://github.com/ultralytics/assets/releases/download/v8.3.0"
for f in yolo11n.pt yolo11s.pt; do
  if [ ! -s "$f" ]; then
    echo "downloading $f"
    curl -L -o "$f" "$BASE/$f"
  fi
done
ls -la *.pt
