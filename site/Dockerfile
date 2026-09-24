# Сайт команды + пайплайн. Контекст сборки — КОРЕНЬ репозитория:
#   docker build -f site/Dockerfile -t trafficeye-site .
#   docker run --rm -p 8000:8000 -e PORT=8000 trafficeye-site
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # CPU-сборки torch с индекса PyTorch — в разы меньше, чем CUDA-колёса с PyPI
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    OMP_NUM_THREADS=2 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# libgl/libglib нужны OpenCV, ffmpeg — для чтения/записи mp4 (H.264 для браузера)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (кэшируемый слой), потом код
COPY site/requirements.txt /tmp/site-requirements.txt
RUN pip install -r /tmp/site-requirements.txt

COPY . .
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; else echo "root requirements.txt not found - installing site dependencies only"; fi

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --app-dir site --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
