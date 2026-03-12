FROM python:3.12-slim

# FFmpeg 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# faster-whisper 모델 캐시 디렉토리
ENV HF_HOME=/app/.cache
ENV WHISPER_MODEL=base
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE=int8

CMD ["python", "main.py"]
