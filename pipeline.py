# ============================================================
# computer-vision-meteran | pipeline.py
# appV1.0 Rev 2
# Fungsi: Pipeline CV — YOLOv8 + LeNet-5 + Preprocessing + Dedup
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# ----------------------------------------------------------------
# KONFIGURASI PATH
# ----------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
YOLO_PATH  = BASE_DIR / "models" / "yolov8_best.pt"
LENET_PATH = BASE_DIR / "models" / "lenet5_best.pt"

# ----------------------------------------------------------------
# DEFINISI MODEL LeNet-5
# ----------------------------------------------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Pipeline] Loading models on {device}...")
yolo_model  = YOLO(str(YOLO_PATH))
lenet_model = LeNet5(num_classes=10).to(device)
lenet_model.load_state_dict(
    torch.load(str(LENET_PATH), map_location=device)
)
lenet_model.eval()
print("[Pipeline] Models loaded successfully!")

# ----------------------------------------------------------------
# TRANSFORMASI LeNet-5
# ----------------------------------------------------------------
lenet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ----------------------------------------------------------------
# PREPROCESSING GAMBAR
# ----------------------------------------------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocessing gambar meteran sebelum deteksi.
    Meningkatkan kontras dan ketajaman digit.
    """
    h, w = image.shape[:2]
    if w > 1280:
        scale = 1280 / w
        image = cv2.resize(image, (1280, int(h * scale)))

    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge([l, a, b])
    image   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    image = cv2.filter2D(image, -1, kernel)

    return image


# ----------------------------------------------------------------
# KLASIFIKASI SATU DIGIT
# ----------------------------------------------------------------
def predict_single_digit(crop_bgr: np.ndarray) -> tuple[int, float]:
    """Klasifikasi satu crop digit menggunakan LeNet-5."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)
    tensor   = lenet_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = lenet_model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return int(pred.item()), float(conf.item())


# ----------------------------------------------------------------
# PIPELINE UTAMA
# ----------------------------------------------------------------
def read_meter(
    image_path: str,
    yolo_conf: float = 0.15,
    lenet_conf: float = 0.20,
    padding_ratio: float = 0.1,
) -> dict:
    """
    Pipeline utama: foto meteran → string angka.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {
            "meter_reading": "",
            "digits"       : [],
            "success"      : False,
            "message"      : "Gagal membaca gambar"
        }

    # Preprocessing
    image = preprocess_image(image)
    cv2.imwrite(image_path, image)

    h_img, w_img = image.shape[:2]
    annotated    = image.copy()

    # --- Stage 1: YOLOv8 ---
    yolo_results = yolo_model.predict(
        source  = image_path,
        conf    = yolo_conf,
        verbose = False,
    )

    boxes = yolo_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {
            "meter_reading": "",
            "digits"       : [],
            "success"      : False,
            "message"      : "Tidak ada digit terdeteksi"
        }

    # --- Stage 2: Crop + LeNet-5 ---
    digit_results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        yolo_conf_score = float(box.conf[0])

        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        x1p   = max(0,     x1 - pad_x)
        y1p   = max(0,     y1 - pad_y)
        x2p   = min(w_img, x2 + pad_x)
        y2p   = min(h_img, y2 + pad_y)

        crop = image[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        digit, lenet_conf_score = predict_single_digit(crop)

        if lenet_conf_score < lenet_conf:
            continue

        digit_results.append({
            "digit"     : digit,
            "x_center"  : (x1 + x2) / 2,
            "yolo_conf" : round(yolo_conf_score, 3),
            "lenet_conf": round(lenet_conf_score, 3),
            "bbox"      : (x1, y1, x2, y2),
        })

    if not digit_results:
        return {
            "meter_reading": "",
            "digits"       : [],
            "success"      : False,
            "message"      : "Semua digit gagal diklasifikasi"
        }

    # --- Stage 3: Post-processing ---
    digit_results.sort(key=lambda x: x["x_center"])

    # --- Deduplikasi: buang box yang overlap (digit fisik sama) ---
    deduped_results = []
    MIN_DISTANCE_RATIO = 0.5  # minimal jarak antar digit = 50% lebar box

    for current in digit_results:
        is_duplicate = False
        cur_width = current["bbox"][2] - current["bbox"][0]

        for kept in deduped_results:
            distance     = abs(current["x_center"] - kept["x_center"])
            min_distance = cur_width * MIN_DISTANCE_RATIO

            if distance < min_distance:
                # Box overlap dengan box yang sudah disimpan
                # → simpan yang confidence YOLO-nya lebih tinggi
                if current["yolo_conf"] > kept["yolo_conf"]:
                    deduped_results.remove(kept)
                    deduped_results.append(current)
                is_duplicate = True
                break

        if not is_duplicate:
            deduped_results.append(current)

    # Sort ulang setelah deduplikasi
    deduped_results.sort(key=lambda x: x["x_center"])
    digit_results = deduped_results

    meter_reading = "".join([str(d["digit"]) for d in digit_results])

    # --- Anotasi gambar ---
    for d in digit_results:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{d['digit']}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

    cv2.putText(
        annotated,
        f"Meteran: {meter_reading}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (0, 0, 255), 3
    )

    result_filename = "result_" + Path(image_path).name
    result_path     = str(Path(image_path).parent / result_filename)
    cv2.imwrite(result_path, annotated)

    return {
        "meter_reading": meter_reading,
        "digits"       : digit_results,
        "result_image" : result_filename,
        "success"      : True,
        "message"      : f"Berhasil membaca {len(digit_results)} digit"
    }