# ⚡ Pembaca Meteran Listrik Otomatis
### Computer Vision — YOLOv8 + LeNet-5

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Railway-7B2CF2?style=for-the-badge)](https://web-production-311c.up.railway.app)
[![GitHub](https://img.shields.io/badge/GitHub-ridho--adri-181717?style=for-the-badge&logo=github)](https://github.com/ridho-adri/computer-vision-meteran)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)

---

## 📌 Deskripsi

Sistem pembaca meteran listrik analog secara otomatis menggunakan pendekatan **Computer Vision**. Pengguna cukup mengupload foto meteran, dan sistem akan mendeteksi serta membaca angka kWh secara otomatis tanpa input manual.

---

## 🧠 Arsitektur Pipeline

```
📷 Foto Meteran
      ↓
🔧 Preprocessing (CLAHE + Sharpening)
      ↓
🎯 YOLOv8  →  Deteksi & Localisasi Digit
      ↓
🔢 LeNet-5  →  Klasifikasi Tiap Digit (0–9)
      ↓
📊 Post-processing (Sort + Deduplikasi)
      ↓
✅ Angka Meteran (kWh)
```

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Web Framework | Flask 3.1.0 |
| Object Detection | YOLOv8 (Ultralytics) |
| Digit Classification | LeNet-5 (PyTorch) |
| Image Processing | OpenCV, Pillow |
| Deployment | Railway (Docker) |
| Frontend | HTML, CSS, Vanilla JS |

---

## 📁 Struktur Project

```
computer-vision-meteran/
├── app.py                  # Flask web application
├── pipeline.py             # CV pipeline (YOLOv8 + LeNet-5)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker config untuk Railway
├── Procfile                # Gunicorn entry point
├── models/
│   ├── yolov8_best.pt      # YOLOv8 trained model (~6MB)
│   └── lenet5_best.pt      # LeNet-5 trained model (~1.3MB)
├── dataset/
│   ├── data.yaml           # Dataset config
│   ├── train/              # Training data (images + labels)
│   └── valid/              # Validation data (images + labels)
├── templates/
│   └── index.html          # Frontend UI
└── static/
    └── css/style.css       # Styling
```

---

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/ridho-adri/computer-vision-meteran.git
cd computer-vision-meteran

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi
python app.py

# 4. Buka browser
# http://localhost:5000
```

---

## 🌐 Live Demo

Aplikasi sudah di-deploy dan dapat diakses di:

**👉 [https://web-production-311c.up.railway.app](https://web-production-311c.up.railway.app)**

---

## 📊 Dataset

Dataset berisi foto meteran listrik analog yang telah dibersihkan dan dianotasi untuk training YOLOv8:

| Split | Jumlah Gambar |
|-------|--------------|
| Train | Lihat folder `dataset/train/` |
| Valid | Lihat folder `dataset/valid/` |

Format anotasi: **YOLO format** (`.txt` per gambar)

---

## 👤 Author

**Ridho Adriansyah**  
Politeknik Caltex Riau  
Semester 8 — Computer Vision Project 2026
