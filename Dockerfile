# Образ для офлайн-прогона организаторов: те же две команды, что и без Docker.
#   docker build -t team .
#   docker run --gpus all -v /data/test:/data/test -v $PWD:/work team python run_submission.py --videos /data/test --out predictions.json
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 YOLO_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "predictions.json"]
