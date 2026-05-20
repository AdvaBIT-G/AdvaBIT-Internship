import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from joblib import load
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

TEST_RAW_DIR = "/home/martinez/flower_phenotyping/data/full_model_testing/test"
TEST_MASK_DIR = "/home/martinez/flower_phenotyping/data/full_model_testing/masks"

EXTENSIONS = (".jpg", ".jpeg", ".png")

# =====================================
# FLOWER SEGMENTATION USING YOLO MODEL
# =====================================
yolo_model = YOLO("/home/martinez/flower_phenotyping/models/yolo/weights/best.pt")

results = yolo_model.predict(
    source=TEST_RAW_DIR,
    imgsz=1024,
    conf=0.3,
    save=True  
)

# folder for masks
mask_dir = TEST_MASK_DIR
os.makedirs(mask_dir, exist_ok=True)

for r in results:

    if r.masks is None:
        continue

    filename = os.path.basename(r.path)
    filename = os.path.splitext(filename)[0] + ".png"

    h, w = r.orig_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in r.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (w, h))
        m = (m > 0.5).astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, m)

    combined_mask = combined_mask.astype(np.uint8)

    cv2.imwrite(os.path.join(mask_dir, filename), combined_mask * 255)

print("✅ Masks saved")

# =========================
# HSV CLASSIFICATION
# =========================

def classify_hsv(h, s, v):

    # white
    if v > 200 and s < 50:
        return "white"

    # red
    if (0 <= h <= 10) or (170 <= h <= 179):
        return "red"

    # orange
    if 10 <= h <= 25:
        return "orange"

    # yellow
    if 25 <= h <= 35:
        return "yellow"

    # green
    if 35 <= h <= 85:
        return "green"

    # purple
    if 130 <= h <= 170:
        return "purple"

    return "unknown"

# =========================
# PROCESS MAKS
# =========================

def process_mask(mask_path):

    mask = cv2.imread(mask_path)

    if mask is None:
        return None

    # Remove background
    valid_mask = np.any(mask > 10, axis=-1)

    if np.sum(valid_mask) == 0:
        return None

    # convert flower pixels to HSV
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    pixels = hsv[valid_mask]

    labels = [
        classify_hsv(h, s, v)
        for h, s, v in pixels
    ]

    counts = Counter(labels)
    total = len(labels)

    percentages = {
        c: 100 * n / total
        for c, n in counts.items()
    }

    return percentages


# =========================
# PIPELINE
# =========================

results = []

for file in os.listdir(TEST_MASK_DIR):

    if not file.lower().endswith(EXTENSIONS):
        continue

    mask_path = os.path.join(TEST_MASK_DIR, file)

    percentages = process_mask(mask_path)

    if percentages is None:
        continue

    row = {"image": file}
    row.update(percentages)

    results.append(row)


# =============================
# SAVE RESULTS IN A DATAFRAME
# =============================

features = pd.DataFrame(results).fillna(0)

# =================
# LOAD COLOR MODEL
# =================
svm = load('/home/martinez/flower_phenotyping/models/color/flower_color_model_svm.joblib')

# =================
# PREDICTION
# =================

pred = svm.predict(features)
print('Predicted color:', pred[0])

