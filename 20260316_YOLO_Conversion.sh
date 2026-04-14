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
    batch=2 \
    patience=30 \
    project="${WORK}/runs" \
    name="train_clean"

log "✅ DONE"