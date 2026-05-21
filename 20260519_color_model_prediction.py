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

FEATURE_ORDER = [ 

    "green", 

    "yellow", 

    "orange", 

    "white", 

    "red", 

    "unknown", 

    "purple", 

    "median_h", 

    "median_s", 

    "median_v", 

    "std_h", 

    "std_s", 

    "std_v", 

    ] 

# =====================================
# FLOWER SEGMENTATION USING YOLO MODEL
# =====================================
yolo_model = YOLO("/home/martinez/flower_phenotyping/models/yolo/weights/best.pt")

results = yolo_model.predict(
    source=TEST_RAW_DIR,
    imgsz=1024,
    conf=0.3,
    device='cpu',
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

    # green
    if 35 <= h <= 85:
        return "green"
    
    # yellow
    if 25 <= h <= 35:
        return "yellow"
    
    # orange
    if 10 <= h <= 25:
        return "orange"

    # white
    if v > 200 and s < 50:
        return "white"

    # red
    if (0 <= h <= 10) or (170 <= h <= 179):
        return "red"

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

    # Keep only flower pixels
    pixels = hsv[valid_mask]

    # Color labels
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
    
    # hsv statistics

    h_values = pixels[:, 0]
    s_values = pixels[:, 1]
    v_values = pixels[:, 2]

    stats = {
        "median_h": np.median(h_values),
        "median_s": np.median(s_values),
        "median_v": np.median(v_values),

        "std_h": np.std(h_values),
        "std_s": np.std(s_values),
        "std_v": np.std(v_values),
    }


    # Ordered features
    all_features = {**percentages, **stats}

    features = {
        key: all_features.get(key, 0)
        for key in FEATURE_ORDER
    }

    return features



# =========================
# PIPELINE
# =========================

results = []

for file in os.listdir(TEST_MASK_DIR):

    if not file.lower().endswith(EXTENSIONS):
        continue

    mask_path = os.path.join(TEST_MASK_DIR, file)

    features = process_mask(mask_path)

    if features is None:
        continue

    row = {"image": file}
    row.update(features)

    results.append(row)


# =============================
# SAVE RESULTS IN A DATAFRAME
# =============================

features_df = pd.DataFrame(results).fillna(0)

# remove image column before prediction
X = features_df.drop(columns=["image"])

# =================
# LOAD COLOR MODEL
# =================
svm = load('/home/martinez/flower_phenotyping/models/color/flower_color_model_svm.joblib')

# =================
# PREDICTION
# =================

pred = svm.predict(X)

features_df["prediction"] = pred

print(features_df[["image", "prediction"]])

features_df.to_csv("20260521_color_predictions.xlsx", index=False)

print("✅ Results saved to 20260521_color_predictions.xlsx")
