# ============================================================
# computer-vision-meteran | clean_dataset.py
# appV1.0 Rev 0
# Fungsi: Membersihkan dataset 17 kelas → 10 kelas (0-9)
#         + split train/valid otomatis
# ============================================================

import os
import shutil
import random
from pathlib import Path

# ----------------------------------------------------------------
# KONFIGURASI — Sesuaikan path ini dengan komputermu
# ----------------------------------------------------------------

# Path dataset asli 
SOURCE_IMAGES = Path("dataset_pertama/train/images")
SOURCE_LABELS = Path("dataset_pertama/train/labels")

# Path output dataset bersih
OUTPUT_DIR    = Path("dataset_bersih")

# Rasio split train/valid
TRAIN_RATIO   = 0.8
SEED          = 42

# ----------------------------------------------------------------
# MAPPING KELAS
# ----------------------------------------------------------------
VALID_CLASS_REMAP = {
    0:  0,   # '0'  → 0
    2:  1,   # '1'  → 1
    3:  2,   # '2'  → 2
    4:  3,   # '3'  → 3
    6:  4,   # '4'  → 4
    7:  5,   # '5'  → 5
    9:  6,   # '6'  → 6
    11: 7,   # '7'  → 7
    13: 8,   # '8'  → 8
    15: 9,   # '9'  → 9
}

# Kelas yang DIBUANG
INVALID_CLASSES = {1, 5, 8, 10, 12, 14, 16}


# ----------------------------------------------------------------
# FUNGSI UTILITAS
# ----------------------------------------------------------------
def clean_label_file(src_label: Path) -> list[str] | None:
    """
    Membaca file label, membuang baris dengan kelas invalid,
    dan meremap index kelas ke format baru (0-9).

    Returns:
        list of str : baris label yang sudah bersih
        None        : jika semua baris dibuang (file kosong)
    """
    if not src_label.exists():
        return None

    cleaned_lines = []

    with open(src_label, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        old_class = int(parts[0])

        # Buang kelas yang tidak diinginkan
        if old_class in INVALID_CLASSES:
            continue

        # Remap ke index baru
        if old_class not in VALID_CLASS_REMAP:
            continue

        new_class = VALID_CLASS_REMAP[old_class]
        new_line  = f"{new_class} {' '.join(parts[1:])}"
        cleaned_lines.append(new_line)

    # Kembalikan None jika tidak ada label tersisa
    return cleaned_lines if cleaned_lines else None


def save_label_file(dest_path: Path, lines: list[str]):
    """Simpan baris label ke file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def copy_image(src: Path, dest: Path):
    """Copy file gambar ke tujuan."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def save_data_yaml(output_dir: Path):
    """Simpan data.yaml baru yang sudah bersih."""
    yaml_content = """# ============================================================
# computer-vision-meteran | data.yaml
# appV1.0 Rev 0
# ============================================================

train: train/images
val:   valid/images

nc: 10
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ data.yaml disimpan → {yaml_path}")


# ----------------------------------------------------------------
# FUNGSI UTAMA
# ----------------------------------------------------------------
def clean_and_split_dataset():
    """
    Pipeline utama:
    1. Baca semua pasangan image+label
    2. Bersihkan label (buang kelas invalid, remap index)
    3. Buang gambar yang tidak punya label valid
    4. Split 80% train / 20% valid
    5. Simpan ke dataset_bersih/
    """
    print("=" * 55)
    print("  Clean Dataset — appV1.0 Rev 0")
    print("=" * 55)

    # Kumpulkan semua file gambar
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted([
        f for f in SOURCE_IMAGES.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    print(f"📂 Source images : {SOURCE_IMAGES}")
    print(f"📂 Source labels : {SOURCE_LABELS}")
    print(f"🖼️  Total gambar  : {len(all_images)}")
    print("=" * 55)

    # Proses setiap gambar
    valid_pairs   = []   # [(image_path, cleaned_lines)]
    skipped_count = 0

    for img_path in all_images:
        label_path = SOURCE_LABELS / (img_path.stem + ".txt")
        cleaned    = clean_label_file(label_path)

        if cleaned is None:
            skipped_count += 1
            continue

        valid_pairs.append((img_path, cleaned))

    print(f"\n✅ Gambar valid   : {len(valid_pairs)}")
    print(f"🗑️  Gambar dibuang : {skipped_count} (tidak ada label valid)")

    # Shuffle dan split train/valid
    random.seed(SEED)
    random.shuffle(valid_pairs)

    split_idx   = int(len(valid_pairs) * TRAIN_RATIO)
    train_pairs = valid_pairs[:split_idx]
    valid_pairs_split = valid_pairs[split_idx:]

    print(f"\n📊 Split dataset:")
    print(f"   Train : {len(train_pairs)} gambar")
    print(f"   Valid : {len(valid_pairs_split)} gambar")

    # Simpan train
    print("\n💾 Menyimpan train...")
    for img_path, label_lines in train_pairs:
        dest_img   = OUTPUT_DIR / "train" / "images" / img_path.name
        dest_label = OUTPUT_DIR / "train" / "labels" / (img_path.stem + ".txt")
        copy_image(img_path, dest_img)
        save_label_file(dest_label, label_lines)

    # Simpan valid
    print("💾 Menyimpan valid...")
    for img_path, label_lines in valid_pairs_split:
        dest_img   = OUTPUT_DIR / "valid" / "images" / img_path.name
        dest_label = OUTPUT_DIR / "valid" / "labels" / (img_path.stem + ".txt")
        copy_image(img_path, dest_img)
        save_label_file(dest_label, label_lines)

    # Simpan data.yaml baru
    save_data_yaml(OUTPUT_DIR)

    print("\n" + "=" * 55)
    print("✅ Dataset berhasil dibersihkan!")
    print(f"📁 Output : {OUTPUT_DIR.resolve()}")
    print("=" * 55)


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
if __name__ == "__main__":
    clean_and_split_dataset()