#!/usr/bin/env bash
set -euo pipefail

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE="/home/martinez/flower_phenotyping/data"
WORK="${BASE}/annotations/YOLO"

RAW_ORI="${BASE}/raw"
JSON_ORI="${BASE}/annotations/json"

RAW="${WORK}/raw_data"
COCO="${WORK}/coco_out"
SAHI="${WORK}/coco_sliced"
YOLO="${WORK}/yolo_dataset"

COCO_SCRIPT="/home/martinez/flower_phenotyping/src/20260316_per_image_jsons_to_coco.py"
SAHI_SCRIPT="/home/martinez/flower_phenotyping/src/20260402_YOLO_SAHI_Training.py"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ─── CLEAN ───────────────────────────────────────────────────────────────────
log "Cleaning workspace..."
rm -rf "${WORK}"
mkdir -p "${RAW}" "${COCO}" "${YOLO}"

# ─── COPY RAW DATA ──────────────────────────────────────────────────────────
log "Copying images and JSON..."
find "${RAW_ORI}" -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) -exec cp -t "${RAW}" {} +
find "${JSON_ORI}" -maxdepth 1 -name "*.json" -exec cp -t "${RAW}" {} +

# ─── COCO CONVERSION ─────────────────────────────────────────────────────────
log "Converting to COCO..."
python "${COCO_SCRIPT}" \
    --input "${RAW}" \
    --out "${COCO}" \
    --val 0.1 \
    --seed 42 \


# ─── CLEAN COCO ──────────────────────────────────────────────────────────────
log "Cleaning COCO..."
python /tmp/clean_coco.py \
    "${COCO}/instances_train.json" \
    "${COCO}/instances_train_clean.json"

python /tmp/clean_coco.py \
    "${COCO}/instances_val.json" \
    "${COCO}/instances_val_clean.json"

# ─── SAHI SLICING ────────────────────────────────────────────────────────────
log "Running SAHI..."
python "${SAHI_SCRIPT}" --base "${WORK}"

# ─── NORMALISE SAHI JSON ─────────────────────────────────────────────────────
log "Normalising SAHI output..."

for split in train val; do
    json_file=$(ls "${SAHI}/${split}"/*.json | head -n 1)
    mv "$json_file" "${SAHI}/${split}/instances.json"
done

# ─── 🔥 COCO → YOLO (NO RENAMING, CLEAN VERSION) ─────────────────────────────
log "Converting COCO to YOLO labels..."

SAHI="${SAHI}" YOLO="${YOLO}" python - << 'PY'
import json
from pathlib import Path
import os

BASE = os.environ["SAHI"]
YOLO = os.environ["YOLO"]

def normalize(poly, w, h):
    return [poly[i] / w if i % 2 == 0 else poly[i] / h for i in range(len(poly))]

def convert(coco_file, out_dir):
    coco = json.load(open(coco_file))

    images = {img["id"]: img for img in coco["images"]}
    labels = {}

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]

        name = Path(img["file_name"]).stem
        cls = ann["category_id"] - 1

        if not ann.get("segmentation"):
            continue

        poly = ann["segmentation"][0]
        poly = normalize(poly, w, h)

        if any(p < 0 or p > 1 for p in poly):
            continue

        labels.setdefault(name, []).append(
            str(cls) + " " + " ".join(map(str, poly))
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for k, v in labels.items():
        (Path(out_dir) / f"{k}.txt").write_text("\n".join(v))


convert(f"{BASE}/train/instances.json",
        f"{YOLO}/labels/train")

convert(f"{BASE}/val/instances.json",
        f"{YOLO}/labels/val")

PY
# ─── COPY IMAGES ────────────────────────────────────────────────────────────
log "Copying images..."

for split in train val; do
    mkdir -p "${YOLO}/images/${split}"
    find "${SAHI}/${split}" -type f \( -name "*.jpg" -o -name "*.png" \) \
        -exec cp {} "${YOLO}/images/${split}/" \;
done

# ─── FILTER BACKGROUND ───────────────────────────────────────────────────────
log "Filtering background images..."

python - << 'PY'
import random
from pathlib import Path

random.seed(42)

YOLO = Path("${YOLO}")

img_dir = YOLO / "images/train"
lbl_dir = YOLO / "labels/train"

# ⚙️ Ajustes
MIN_AREA = 0.002   # mantener flores pequeñas
EDGE_MARGIN = 0.02
BG_RATIO = 0.4     # 40% de background respecto a foreground

fg_imgs = []
bg_imgs = []

# 🔍 Paso 1: limpiar labels
for img_path in img_dir.glob("*.jpg"):
    label_path = lbl_dir / f"{img_path.stem}.txt"

    if not label_path.exists():
        bg_imgs.append(img_path)
        continue

    valid_lines = []

    with open(label_path, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())
            area = w * h

            # ❌ demasiado pequeña
            if area < MIN_AREA:
                continue

            # ❌ en bordes (mal slice)
            if x < EDGE_MARGIN or x > 1 - EDGE_MARGIN:
                continue
            if y < EDGE_MARGIN or y > 1 - EDGE_MARGIN:
                continue

            valid_lines.append(line)

    if len(valid_lines) == 0:
        # se convierte en background
        label_path.unlink(missing_ok=True)
        bg_imgs.append(img_path)
    else:
        # guardar limpio
        with open(label_path, "w") as f:
            f.writelines(valid_lines)
        fg_imgs.append(img_path)

print(f"Después de limpiar → FG: {len(fg_imgs)} | BG: {len(bg_imgs)}")

# 🎯 Paso 2: balancear background
target_bg = int(len(fg_imgs) * BG_RATIO)

if len(bg_imgs) > target_bg:
    remove_n = len(bg_imgs) - target_bg
    to_remove = random.sample(bg_imgs, remove_n)

    for img in to_remove:
        img.unlink()

print(f"Final → FG: {len(fg_imgs)} | BG: {target_bg}")
PY
# ─── DATA.YAML ───────────────────────────────────────────────────────────────
log "Writing data.yaml..."

python - << PY
import json
from pathlib import Path

cats = json.load(open("${COCO}/instances_train_clean.json"))["categories"]
cats = sorted(cats, key=lambda x: x["id"])

names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(cats))

Path("${WORK}/data.yaml").write_text(f"""path: ${YOLO}
train: images/train
val: images/val

nc: {len(cats)}
names:
{names}
""")
PY

# ─── SANITY CHECK ────────────────────────────────────────────────────────────
log "Sanity check..."

n_img=$(find "${YOLO}/images/train" -type f | wc -l)
n_lbl=$(find "${YOLO}/labels/train" -type f | wc -l)

log "Train images: ${n_img}"
log "Train labels: ${n_lbl}"

[[ ${n_img} -eq 0 ]] && die "No images found!"
[[ ${n_lbl} -eq 0 ]] && die "No labels found!"

# ─── TRAIN ───────────────────────────────────────────────────────────────────
log "Starting training..."

yolo task=segment mode=train \
    model=yolo11s-seg.pt \
    data="${WORK}/data.yaml" \
    imgsz=640 \
    epochs=100 \
    batch=24 \
    amp=True \
    workers=6 \
    cache=False \
    patience=30 \
    project="${WORK}/runs" \
    name="train_clean"

log "✅ DONE"