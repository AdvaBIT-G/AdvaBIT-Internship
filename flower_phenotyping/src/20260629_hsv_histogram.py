import os
import cv2
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

IMG_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/selected_raw/train"
CSV_LABELS = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/Color_training_dataset.csv"

OUTPUT_CSV = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/HSV_histogram_dataset.csv"

BINS = 16
EXTENSIONS = (".jpg", ".jpeg", ".png")

# =========================================================
# LOAD LABELS
# =========================================================

labels_df = pd.read_csv(CSV_LABELS, sep=";", decimal=",")

# clean names
labels_df["image"] = labels_df["image"].str.strip().str.lower()

# =========================================================
# HISTOGRAM FUNCTION
# =========================================================

def hsv_histogram(image, bins=16):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()

    # normalize histogram
    hist = hist / (np.sum(hist) + 1e-6)

    return hist

# =========================================================
# BUILD DATASET
# =========================================================

rows = []

for file in os.listdir(IMG_DIR):

    if not file.lower().endswith(EXTENSIONS):
        continue

    path = os.path.join(IMG_DIR, file)
    img = cv2.imread(path)

    if img is None:
        continue

    hist = hsv_histogram(img, bins=BINS)

    label_row = labels_df[labels_df["image"] == file.lower()]

    if len(label_row) == 0:
        continue

    label = label_row["Cluster"].values[0]

    row = {"image": file, "Cluster": label}

    # add histogram features
    for i, val in enumerate(hist):
        row[f"hist_{i}"] = val

    rows.append(row)

# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

print("Dataset created:", OUTPUT_CSV)
print(df.shape)
print(df.head())