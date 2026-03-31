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

# Ruta al script de conversión ISAT-SAM → YOLO directo
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
# 4. ISAT-SAM JSON → YOLO DIRECTO
# (sin pasar por COCO ni Ultralytics converter)
###############################

echo ">>> Convirtiendo JSONs de ISAT-SAM a formato YOLO segmentation..."

python "${ISAT_TO_YOLO}" \
    --input   "${RAW_DIR}" \
    --out     "${YOLO_DIR}" \
    --images  "${RAW_DIR}" \
    --val     0.1

# Después de esto tendrás:
#   ${YOLO_DIR}/images/train/*.jpg
#   ${YOLO_DIR}/images/val/*.jpg
#   ${YOLO_DIR}/labels/train/*.txt
#   ${YOLO_DIR}/labels/val/*.txt

###############################
# 5. WRITE data.yaml
###############################

echo ">>> Escribiendo data.yaml..."

cat > "${WORK_DIR}/data.yaml" <<YAML
# YOLO11 segmentation dataset config

path: ${YOLO_DIR}

train: images/train
val:   images/val

# 0 y 1 tienen que coincidir con CATEGORY_MAP en isat_to_yolo.py
names:
  0: Flower
  1: Plant
YAML

###############################
# 6. TRAINING
###############################

echo ">>> Iniciando entrenamiento YOLO11 segmentation..."

cd "${BASE_DIR}"

yolo task=segment mode=train \
     model=yolo11s-seg.pt \
     data="${WORK_DIR}/data.yaml" \
     imgsz=640 \
     epochs=50 \
     batch=4 \
     lr0=0.01 \
     patience=50 \
    hsv_h=0.015 \
     hsv_s=0.7 \
     hsv_v=0.6 \
     fliplr=0.5 \
     flipud=0.3 \
     degrees=15 \
     translate=0.2 \
     scale=0.3 \
     mosaic=0.5 \
     copy_paste=0.1 \
     project="${WORK_DIR}/runs" \
     name="train_plant_seg_v2"

echo ">>> Entrenamiento finalizado. Resultados en: ${WORK_DIR}/runs"