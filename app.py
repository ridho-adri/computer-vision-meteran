# ============================================================
# computer-vision-meteran | app.py
# appV1.0 Rev 1
# Fungsi: Flask Web App — Upload foto → baca angka meteran
# ============================================================

import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from pipeline import read_meter

# ----------------------------------------------------------------
# KONFIGURASI FLASK
# ----------------------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = Path("static/uploads")
ALLOWED_EXT   = {"jpg", "jpeg", "png", "bmp"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# UTILITAS
# ----------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )

def generate_filename(original: str) -> str:
    ext = original.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


# ----------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Tidak ada file dipilih"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Format tidak didukung"}), 400

    filename  = generate_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(str(file_path))

    # Pipeline dengan threshold yang sudah dioptimalkan
    result = read_meter(
        str(file_path),
        yolo_conf  = 0.15,
        lenet_conf = 0.20,
    )

    response = {
        "success"      : result["success"],
        "message"      : result["message"],
        "meter_reading": result["meter_reading"],
        "digits"       : result.get("digits", []),
        "original_img" : f"/static/uploads/{filename}",
        "result_img"   : f"/static/uploads/{result.get('result_image', filename)}"
                         if result["success"] else None,
    }

    return jsonify(response)


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  Meteran CV Web App — appV1.0 Rev 1")
    print("  Buka browser: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)