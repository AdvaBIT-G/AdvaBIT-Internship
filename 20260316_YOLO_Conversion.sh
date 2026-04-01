#!/usr/bin/env bash
set -euo pipefail


###############################
# 1. PATHS
###############################

BASE_DIR="/home/martinez/flower_phenotyping/data"
ORI_IMG_DIR="${BASE_DIR}/raw"
JSON_ANN_DIR="${BASE_DIR}/annotations/json"

WORK_DIR="${BASE_DIR}/annotations/YOLO"
RAW_DIR="${WORK_DIR}/raw_data"
YOLO_DIR="${WORK_DIR}/yolo_dataset"

# Path to the script for ISAT-SAM → YOLO conversion
ISAT_TO_YOLO="/home/martinez/flower_phenotyping/src/20260316_per_image_jsons_to_coco.py"

###############################
# 2. CLEAN & PREP FOLDERS
###############################

echo ">>> Limpiando y preparando directorios..."
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
mkdir -p "${RAW_DIR}"
mkdir -p "${YOLO_DIR}"

###############################
# 3. COPY RAW IMAGES + JSONS
###############################

echo ">>> Copiando imágenes desde ${ORI_IMG_DIR} y JSONs desde ${JSON_ANN_DIR}..."

find "${ORI_IMG_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) \
    -exec cp -t "${RAW_DIR}" -- {} +

find "${JSON_ANN_DIR}" -maxdepth 1 -type f -iname '*.json' \
    -exec cp -t "${RAW_DIR}" -- {} +

# Borrar duplicados raros
find "${RAW_DIR}" -iname '*.out.JPG' -print0 | xargs -0 -r rm || true

###############################
# 4. ISAT-SAM JSON → YOLO 
# (without going through COCO or Ultralytics converter)
###############################

echo ">>> Converting JSONs from ISAT-SAM to YOLO segmentation format..."

python "${ISAT_TO_YOLO}" \
    --input   "${RAW_DIR}" \
    --out     "${YOLO_DIR}" \
    --images  "${RAW_DIR}" \
    --val     0.1

# After this you will have:
#   ${YOLO_DIR}/images/train/*.jpg
#   ${YOLO_DIR}/images/val/*.jpg
#   ${YOLO_DIR}/labels/train/*.txt
#   ${YOLO_DIR}/labels/val/*.txt

##########################
# 4.5 CLEAN NOISY IMAGES
##########################

echo ">>> removing images with >20 instances..."

MAX_INSTANCES=20
removed=0
for f in "${YOLO_DIR}/labels/train/"*.txt; do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt "$MAX_INSTANCES" ]; then
        stem=$(basename "$f" .txt)
        rm "$f"
        rm -f "${YOLO_DIR}/images/train/${stem}.jpg"
        rm -f "${YOLO_DIR}/images/train/${stem}.jpeg"
        rm -f "${YOLO_DIR}/images/train/${stem}.png"
        ((removed++)) || true
    fi
done

echo ">>> ${removed} noisy images removed"


###############################
# 5. WRITE data.yaml
###############################

echo ">>> Writing data.yaml..."

cat > "${WORK_DIR}/data.yaml" <<YAML
# YOLO11 segmentation dataset config

path: ${YOLO_DIR}

train: images/train
val:   images/val

# 0 y 1 must be the same as in CATEGORY_MAP in 20260316_per_image_jsons_to_coco.py
names:
  0: Flower
  1: Plant
YAML

###############################
# 6. TRAINING
###############################

echo ">>> Starting training YOLO11 segmentation..."

cd "${BASE_DIR}"

yolo task=segment mode=predict \
     model="/home/martinez/flower_phenotyping/data/runs/segment/train/weights/best.pt" \
     source="/home/martinez/flower_phenotyping/Series04" \
     save=True \
     save_txt=True\
     conf=0.6

echo ">>> Training finished. Results in: ${WORK_DIR}/runs"