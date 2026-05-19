import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from joblib import load

# =========================
# CONFIG
# =========================

MASK_DIR = "/home/martinez/flower_phenotyping/data/annotations/color_annotations/test"

EXTENSIONS = (".jpg", ".jpeg", ".png")

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

for file in os.listdir(MASK_DIR):

    if not file.lower().endswith(EXTENSIONS):
        continue

    mask_path = os.path.join(MASK_DIR, file)

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

