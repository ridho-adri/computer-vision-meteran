# ============================================================
# Dockerfile — computer-vision-meteran
# Fix: libxcb + OpenCV headless di Railway
# ============================================================

FROM python:3.11-slim

# Install system dependencies untuk OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libxext6 \
    libx11-6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Pastikan folder uploads ada
RUN mkdir -p static/uploads

EXPOSE 8000

# Jalankan dengan gunicorn
CMD gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
