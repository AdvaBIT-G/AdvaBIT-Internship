#!/usr/bin/env bash
set -euo pipefail

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE="/home/martinez/flower_phenotyping/data"
WORK="${BASE}/annotations/YOLO"

RAW_ORI="${BASE}/raw"
JSON_ORI="${BASE}/annotations/new_json"

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
    --seed 42

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
python "${SAHI_SCRIPT}" \
   --base "${WORK}" \
   --overlap-h 0.2 \
   --overlap-w 0.2 \

# ─── NORMALISE SAHI JSON ─────────────────────────────────────────────────────
log "Normalising SAHI output..."
for split in train val; do
    json_file=$(ls "${SAHI}/${split}"/*.json | head -n 1)
    mv "$json_file" "${SAHI}/${split}/instances.json"
done

# ─── COCO → YOLO ─────────────────────────────────────────────────────────────
log "Converting COCO to YOLO labels..."

SAHI="${SAHI}" YOLO="${YOLO}" python - << 'PY'
import json
from pathlib import Path
import os

BASE = os.environ["SAHI"]
YOLO = os.environ["YOLO"]

PLANTA_NAME = "planta"   # ⚠️ corregido

def normalize(poly, w, h):
    return [
        poly[i] / w if i % 2 == 0 else poly[i] / h
        for i in range(len(poly))
    ]

def convert(coco_file, out_dir):
    coco = json.load(open(coco_file))

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    labels = {}

    for ann in coco["annotations"]:
        cat_name = cat_map[ann["category_id"]]

        # 🔥 eliminar planta completamente
        if cat_name == PLANTA_NAME:
            continue

        if not ann.get("segmentation"):
            continue

        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]

        name = Path(img["file_name"]).stem
        cls = 0  # solo flor

        poly = normalize(ann["segmentation"][0], w, h)

        if any(p < 0 or p > 1 for p in poly):
            continue

        labels.setdefault(name, []).append(
            f"{cls} " + " ".join(map(str, poly))
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

# ─── FILTER DATASET ─────────────────────────────────────────────────────────
log "Filtering dataset..."

YOLO_ENV="${YOLO}" python - << 'PY'
import random
from pathlib import Path
import os

random.seed(42)

YOLO = Path(os.environ["YOLO_ENV"])

img_dir = YOLO / "images/train"
lbl_dir = YOLO / "labels/train"

MIN_AREA = 0.0005
EDGE_MARGIN = 0.02
KEEP_EMPTY_PROB = 0.8

fg_imgs = []
bg_imgs = []

img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

for img_path in img_paths:
    label_path = lbl_dir / f"{img_path.stem}.txt"
    valid_lines = []

    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.split()))
                cls = int(parts[0])
                coords = parts[1:]

                xs = coords[0::2]
                ys = coords[1::2]

                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

                w = xmax - xmin
                h = ymax - ymin
                area = w * h

                if area < MIN_AREA:
                    continue

                # filtro de objetos cortados (robusto)
                touch = [
                    xmin <= EDGE_MARGIN,
                    xmax >= 1 - EDGE_MARGIN,
                    ymin <= EDGE_MARGIN,
                    ymax >= 1 - EDGE_MARGIN,
                ]

                if sum(touch) >= 2:
                    continue

                valid_lines.append(line)

    if len(valid_lines) == 0:
        if random.random() < KEEP_EMPTY_PROB:
            label_path.unlink(missing_ok=True)
            bg_imgs.append(img_path)
        else:
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
    else:
        with open(label_path, "w") as f:
            f.writelines(valid_lines)
        fg_imgs.append(img_path)

print(f"After cleaning → FG: {len(fg_imgs)} | BG: {len(bg_imgs)}")
PY

# ─── DATA.YAML ───────────────────────────────────────────────────────────────
log "Writing data.yaml..."

python - << PY
import json
from pathlib import Path

cats = json.load(open("${COCO}/instances_train_clean.json"))["categories"]

# ❌ quitar planta
cats = [c for c in cats if c["name"] != "planta"]

Path("${WORK}/data.yaml").write_text(f"""path: ${YOLO}
train: images/train
val: images/val

nc: 1
names:
  0: flor
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

log "Checking remaining classes..."

grep -r " 1 " "${YOLO}/labels/train" || true

# ─── TRAIN ───────────────────────────────────────────────────────────────────
log "Starting training..."

yolo task=segment mode=train \
    model=yolo11s-seg.pt \
    data="${WORK}/data.yaml" \
    imgsz=640 \
    epochs=200 \
    batch=24 \
    amp=True \
    workers=6 \
    cache=False \
    patience=150 \
    mosaic=0.5 \
    translate=0.3 \
    scale=0.6 \
    lr0=0.0005 \
    project="${WORK}/runs" \
    name="train_clean"

log "✅ DONE"