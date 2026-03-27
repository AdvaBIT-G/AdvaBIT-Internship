#!/usr/bin/env bash
set -euo pipefail

###############################
# 0. ENVIRONMENT (optional)
###############################
#conda init
#conda activate ultra_env

# If needed (only once):
# pip install --upgrade pip
# pip install ultralytics
# pip install opencv-python scikit-image numpy pandas matplotlib
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

###############################
# 1. PATHS
###############################

# Root of your project
BASE_DIR="/home/martinez/flower_phenotyping/data"

# Folder with your original images
ORI_IMG_DIR="${BASE_DIR}/raw"

#Folder with per-image JSON annotations
JSON_ANN_DIR="${BASE_DIR}/annotations/json"

# Working directory for YOLO11 dataset + runs
WORK_DIR="${BASE_DIR}/annotations/YOLO"
RAW_DIR="${WORK_DIR}/raw_data"
COCO_DIR="${WORK_DIR}/coco_out"
YOLO_DIR="${WORK_DIR}/yolo_dataset"   # final YOLO-format dataset

# Path to your script that converts per-image JSONs to COCO
PER_IMAGE_TO_COCO="/home/martinez/flower_phenotyping/src/20260316_per_image_jsons_to_coco.py"

###############################
# 2. CLEAN & PREP FOLDERS
###############################

#echo ">>> Cleaning and preparing working directories..."
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
mkdir -p "${RAW_DIR}"
mkdir -p "${COCO_DIR}"
mkdir -p "${YOLO_DIR}"

###############################
# 3. COPY RAW IMAGES + JSONS
###############################

echo ">>> Copying raw images ${ORI_IMG_DIR} and json files from ${JSON_ANN_DIR} to ${RAW_DIR}..."

find "${ORI_IMG_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) \
    -exec cp -t "${RAW_DIR}" -- {} +

find "${JSON_ANN_DIR}" -maxdepth 1 -type f -iname '*.json' \
    -exec cp -t "${RAW_DIR}" -- {} +

# Optional: remove any weird duplicate/out files
find "${RAW_DIR}" -iname '*.out.JPG' -print0 | xargs -0 -r rm || true

###############################
# 4. PER-IMAGE JSON -> COCO
###############################

echo ">>> Converting per-image JSONs to COCO format..."

python "${PER_IMAGE_TO_COCO}" \
   --input "${RAW_DIR}" \
   --out   "${COCO_DIR}" \
   --val   0.1 \
  

# After this you should have:
#   ${COCO_DIR}/annotations/instances_train.json
#   ${COCO_DIR}/annotations/instances_val.json
# and usually copied/linked images.

###############################
# 5. COCO -> YOLO (SEGMENTATION)
###############################

echo ">>> Converting COCO annotations to YOLO segmentation format..."
echo ">>> Removing previous YOLO dataset..."
rm -rf "${YOLO_DIR}"

python - <<PY
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="${COCO_DIR}/annotations",  # folder with instances_train.json / instances_val.json
    save_dir="${YOLO_DIR}",                # where YOLO-style images/ and labels/ will be created
    use_segments=True,                      # segmentation labels (polygons)
)
PY

# After this, you should have something like:
#   ${YOLO_DIR}/images/train/*.jpg
#   ${YOLO_DIR}/images/val/*.jpg
#   ${YOLO_DIR}/labels/train/*.txt
#   ${YOLO_DIR}/labels/val/*.txt

# Base dir where YOLO11dir lives
BASE_DIR="/home/martinez/flower_phenotyping/data/"
WORK_DIR="${BASE_DIR}/annotations/YOLO"

RAW_DIR="${WORK_DIR}/raw_data"
YOLO_DIR="${WORK_DIR}/yolo_dataset"

echo "BASE_DIR = ${BASE_DIR}"
echo "WORK_DIR = ${WORK_DIR}"
echo "RAW_DIR  = ${RAW_DIR}"
echo "YOLO_DIR = ${YOLO_DIR}"

echo ">>> Ensuring images/train and images/val exist under ${YOLO_DIR}..."

mkdir -p "${YOLO_DIR}/images/train"
mkdir -p "${YOLO_DIR}/images/val"

###############################
# 6. COPY MATCHING IMAGES
###############################
# For each label file in labels/train and labels/val:
#   - take the basename (without .txt)
#   - find corresponding image in RAW_DIR with .jpg/.jpeg/.png
#   - copy it into images/train or images/val

echo ">>> Matching label files to raw images and copying..."

python - <<PY
import os
import glob
import shutil

base_dir = r"${BASE_DIR}"
work_dir = r"${WORK_DIR}"
raw_dir  = r"${RAW_DIR}"
yolo_dir = r"${YOLO_DIR}"

labels_train_dir = os.path.join(yolo_dir, "labels", "train")
labels_val_dir   = os.path.join(yolo_dir, "labels", "val")
images_train_dir = os.path.join(yolo_dir, "images", "train")
images_val_dir   = os.path.join(yolo_dir, "images", "val")

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)

# Build an index of basename -> image path from RAW_DIR
print(f"Indexing images in {raw_dir}...")
img_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
image_map = {}

for pattern in img_exts:
    for img_path in glob.glob(os.path.join(raw_dir, pattern)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        # if duplicates exist, last one wins (usually not an issue)
        image_map[base] = img_path

print(f"Found {len(image_map)} images in raw_dir.")

def copy_for_split(labels_dir, images_dir):
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"Processing {len(label_files)} label files in {labels_dir}...")
    missing = 0

    for lbl in label_files:
        base = os.path.splitext(os.path.basename(lbl))[0]
        if base not in image_map:
            print(f"WARNING: no image found for label {base} in RAW_DIR.")
            missing += 1
            continue

        src_img = image_map[base]
        ext = os.path.splitext(src_img)[1]
        dst_img = os.path.join(images_dir, base + ext)

        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

    print(f"Done {labels_dir}. Missing images for {missing} labels.")

copy_for_split(labels_train_dir, images_train_dir)
copy_for_split(labels_val_dir, images_val_dir)
PY

###############################
# 7. CREATE 5-FOLD SPLITS
###############################
echo ">>> Creating 5-fold cross-validation splits..."
python - <<PY
import os, glob, random, shutil

yolo_dir = r"${YOLO_DIR}"
fold_base = os.path.join(yolo_dir, "folds")
os.makedirs(fold_base, exist_ok=True)

all_imgs = glob.glob(os.path.join(yolo_dir, "images", "train", "*.jpg"))
random.shuffle(all_imgs)
n_folds = 5
fold_size = len(all_imgs) // n_folds

for i in range(n_folds):
    fold_dir = os.path.join(fold_base, f"fold{i+1}")
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "labels"), exist_ok=True)

    val_imgs = all_imgs[i*fold_size:(i+1)*fold_size]
    train_imgs = [img for img in all_imgs if img not in val_imgs]

    # Copy train + val images and labels
    for img_path in train_imgs + val_imgs:
        shutil.copy2(img_path, os.path.join(fold_dir, "images", os.path.basename(img_path)))
        lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        lbl_path = os.path.join(yolo_dir, "labels", "train", lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(fold_dir, "labels", lbl_name))

    # Write data.yaml for this fold
    yaml_path = os.path.join(fold_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {fold_dir}
train: images
val: images
names:
  0: Flower
  1: Plant
""")
PY

###############################
# 8. AUTOMATIC GRID SEARCH PER FOLD
###############################
echo ">>> Starting grid search for 5-fold CV..."

# Define hyperparameter grid
IMG_SIZES=(416 512)
BATCH_SIZES=(2 4)
LEARNING_RATES=(0.001 0.002)
EPOCHS=(100 150)

for img in "${IMG_SIZES[@]}"; do
  for batch in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        run_name="img${img}_batch${batch}_lr${lr}_ep${epoch}"
        echo ">>> Training combination $run_name"

        for fold in {1..5}; do
          echo "Fold $fold..."
          yolo task=segment mode=train \
               model=yolo11n-seg.pt \
               data="${YOLO_DIR}/folds/fold${fold}/data.yaml" \
               imgsz="$img" \
               batch="$batch" \
               lr0="$lr" \
               epochs="$epoch" \
               project="${WORK_DIR}/runs" \
               name="${run_name}_fold${fold}"
        done
      done
    done
  done
done

echo ">>> Grid search + 5-fold CV finished. Check runs in ${WORK_DIR}/runs"

###############################
# 3. WRITE data.yaml
###############################

#echo ">>> Writing data.yaml..."

#cat > "${WORK_DIR}/data.yaml" <<YAML
# YOLO11 segmentation dataset config

# Root of the dataset (where images/ and labels/ live)
#path: /home/martinez/flower_phenotyping/data/annotations/YOLO/yolo_dataset

# These are relative to 'path'
#train: images/train
#val: images/val

# Class names — adjust if COCO had more classes.
# Background is implicit; do NOT include it.
#names:
  #0: Flower
  #1: Plant
#YAML

#
#echo ">>> Starting YOLO11 segmentation training..."

#cd "${BASE_DIR}"

#yolo task=segment mode=train \
     #model=yolo11n-seg.pt \
     #data="${WORK_DIR}/data.yaml" \
     #imgsz=416 \
     #epochs=100 \
     #batch=2\
     #lr0=0.001\
     #project="${WORK_DIR}/runs" \
     #name="train_plant_seg"

#echo ">>> Training finished. Check runs in: ${WORK_DIR}/runs"