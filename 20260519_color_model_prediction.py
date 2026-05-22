import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from joblib import load
from ultralytics import YOLO
import shutil


# =========================
# CONFIG
# =========================

TEST_RAW_DIR = "/home/martinez/flower_phenotyping/data/full_model_testing/test"
TEST_MASK_DIR = "/home/martinez/flower_phenotyping/data/full_model_testing/masks"


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

    original_name = os.path.basename(r.path)
    filename = os.path.splitext(original_name)[0] + ".png"

    h, w = r.orig_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in r.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, m)

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

def process_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        return None
    #Binary mask
    valid_mask = mask > 0
    if np.sum(valid_mask) == 0:
        return None

    # Remove background
    valid_mask = np.any(mask > 10, axis=-1)

    if np.sum(valid_mask) == 0:
        return None

    # convert flower pixels to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
        c: 100 * counts.get(c, 0) / total
        for c in FEATURE_ORDER[:7]
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


    # Merge
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

    if not file.lower().endswith(".png"):
        continue

    # file without extension
    base_name = os.path.splitext(file)[0]

    # mask
    mask_path = os.path.join(TEST_MASK_DIR, file)

    # raw image
    image_file = base_name + ".jpg"
    image_path = os.path.join(TEST_RAW_DIR, image_file)
    

    features = process_mask(image_path, mask_path)

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

features_df["cluster_prediction"] = pred

print(features_df[["image", "cluster_prediction"]])

features_df.to_csv("/home/martinez/flower_phenotyping/results/color/20260521_color_predictions.csv", index=False)

print("✅ Results saved to /home/martinez/flower_phenotyping/results/color/20260521_color_predictions.csv")

# ==========================
# IMAGE FOLDER PER CLUSTER
# ==========================

#Paths
csv_file = '/home/martinez/flower_phenotyping/results/color/20260521_color_predictions.csv'
source_folder = '/home/martinez/flower_phenotyping/data/raw'
output_folder = '/home/martinez/flower_phenotyping/results/color/'

#Read csv
df = pd.read_csv(csv_file)

#Clean image names
df['image'] = df['image'].astype(str).str.strip()
df['cluster_prediction'] = df['cluster_prediction'].astype(str).str.strip()

#Cluster folder creation 
clusters = df['cluster_prediction'].unique()

for cluster in clusters:
    cluster_path = os.path.join(output_folder, f"cluster_{cluster}")
    os.makedirs(cluster_path, exist_ok=True)

# List of files
available_files = os.listdir(source_folder)

#Dictionary for the case-insensitive search
file_map = {}

for f in available_files:
    base_name, ext = os.path.splitext(f) #separate name and extension
    if ext.lower() in ['.jpg']:
        file_map[base_name.lower()] = f

#Copy images
copied = 0
missing = 0

for _, row in df.iterrows():
    image_name = row['image']
    cluster = row['cluster_prediction']

    #Remove extension from csv image names
    base_name = os.path.splitext(image_name)[0].lower()

    #Search corresponding .jpg
    real_file = file_map.get(base_name)

    if real_file:

      source_path = os.path.join(source_folder, real_file)
      destination_path = os.path.join(
         output_folder,
         f"cluster_{cluster}",
         real_file    
        )

      shutil.copy2(source_path, destination_path)
      print(f"Copied: {real_file} -> cluster_{cluster}")
      copied += 1
    else:
      print(f"Not found: {image_name}")
      missing += 1

print("Process ended.")