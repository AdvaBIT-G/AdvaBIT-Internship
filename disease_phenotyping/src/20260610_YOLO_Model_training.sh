#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
BASE="/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/data"
WORK="${BASE}/YOLO"

RAW_ORI="${BASE}/raw"
JSON_ORI="${BASE}/annotations/json"

RAW="${WORK}/raw_data"
COCO="${WORK}/coco_out"
YOLO="${WORK}/yolo_dataset"

COCO_SCRIPT="/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/src/20260611_jsons_to_coco.py"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

rm -rf "$WORK"
mkdir -p "$RAW" "$COCO" "$YOLO"

# ─────────────────────────────────────────────
log "Copying data..."
find "$RAW_ORI" -name "*.jpg" -exec cp -t "$RAW" {} +
find "$JSON_ORI" -name "*.json" -exec cp -t "$RAW" {} +

# ─────────────────────────────────────────────
log "Converting to COCO..."
python "$COCO_SCRIPT" \
    --input "$RAW" \
    --out "$COCO" \
    --val 0.1 \
    --seed 42

# ─────────────────────────────────────────────
log "COCO → YOLO (FIXED segmentation)"

python - << 'PY'
import json
from pathlib import Path
import os

BASE = "/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/data/YOLO/coco_out"
YOLO = "/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/data/YOLO/yolo_dataset"

def norm(poly, w, h):
    return [poly[i]/w if i%2==0 else poly[i]/h for i in range(len(poly))]

def clamp(x):
    return max(0.0, min(1.0, x))

def convert(coco_file, out_dir):

    coco = json.load(open(coco_file))

    images = {img["id"]: img for img in coco["images"]}

    labels = {}

    for ann in coco["annotations"]:

        if not ann.get("segmentation"):
            continue

        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]

        poly = ann["segmentation"][0]

        # ── normalize + clamp (NO DROP)
        poly = norm(poly, w, h)
        poly = [clamp(p) for p in poly]

        if len(poly) < 6:
            continue

        name = Path(img["file_name"]).stem

        # SOLO UNA CLASE: flower = 0
        labels.setdefault(name, []).append(
            "0 " + " ".join(map(str, poly))
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for k, v in labels.items():
        (Path(out_dir) / f"{k}.txt").write_text("\n".join(v))


convert(f"{BASE}/instances_train.json",
        f"{YOLO}/labels/train")

convert(f"{BASE}/instances_val.json",
        f"{YOLO}/labels/val")
PY

# ─────────────────────────────────────────────
log "Copying train/val images..."
COCO="${COCO}" RAW="${RAW}" YOLO="${YOLO}" python - << 'PY'
import json
import shutil
import os
from pathlib import Path

coco_dir = Path(os.environ["COCO"])
raw_dir = Path(os.environ["RAW"])
yolo_dir = Path(os.environ["YOLO"])

mapping = {
    "train": "instances_train.json",
    "val": "instances_val.json"
}

for split, json_file in mapping.items():

    data = json.load(open(coco_dir / json_file))

    dst = yolo_dir / "images" / split
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0

    for img in data["images"]:

        src = raw_dir / img["file_name"]

        if src.exists():
            shutil.copy2(src, dst / src.name)
            copied += 1
        else:
            print("Missing:", src)

    print(f"{split}: copied {copied} images")
PY


# ─────────────────────────────────────────────
log "data.yaml"

cat > "$WORK/data.yaml" << EOF
path: $YOLO
train: images/train
val: images/val

nc: 1
names:
  0: leaf
EOF

# ─────────────────────────────────────────────
log "SANITY CHECK"

n_img=$(find "$YOLO/images/train" -type f | wc -l)
n_lbl=$(find "$YOLO/labels/train" -type f | wc -l)

echo "Images: $n_img"
echo "Labels: $n_lbl"

[[ $n_lbl -eq 0 ]] && echo "WARNING: NO LABELS FOUND"

# ─────────────────────────────────────────────
log "TRAINING"

yolo task=segment mode=train \
    model=yolo11s-seg.pt \
    data="$WORK/data.yaml" \
    imgsz=640 \
    epochs=200 \
    batch=24 \
    cache=False \
    patience=150 \
    mosaic=0.5 \
    lr0=0.01 \
    project="$WORK/runs" \
    name="train_no_sahi"

log "DONE"