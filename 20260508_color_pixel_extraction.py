import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter

# =========================
# CONFIG
# =========================

MASK_DIR = "/home/martinez/flower_phenotyping/data/annotations/color_annotations/train"
OUTPUT_CSV = "/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260508_mask_color_percentages.csv"

EXTENSIONS = (".jpg", ".jpeg", ".png")

# =========================
# HSV CLASSIFICATION
# =========================

def classify_hsv(h, s, v):

    # 1. blanco (prioridad máxima)
    if s < 40 and v > 200:
        return "white"

    # 2. rojo (caso circular en HSV)
    if (0 <= h <= 10) or (170 <= h <= 179):
        if s >= 50 and v >= 50:
            return "red"

    # 3. naranja
    if 10 <= h <= 25 and s >= 80 and v >= 80:
        return "orange"

    # 4. verde
    if 35 <= h <= 85 and s >= 40 and v >= 40:
        return "green"

    # 5. púrpura
    if 130 <= h <= 170 and s >= 40 and v >= 40:
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


# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(results).fillna(0)

df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print(df.head())