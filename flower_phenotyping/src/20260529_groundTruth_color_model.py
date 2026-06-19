import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from collections import Counter
from joblib import load


# =========================
# CONFIG
# =========================

TEST_RAW_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/full_model_testing/groundTruth"
TEST_MASK_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/groundTruth"

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
svm = load('/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/models/color/flower_color_model_svm.joblib')

# =================
# PREDICTION
# =================

pred = svm.predict(X)
probs = svm.predict_proba(X)
confidence = probs.max(axis=1)

features_df["cluster_prediction"] = pred
features_df['prob_cluster1'] = probs[:,0]
features_df['prob_cluster2'] = probs[:,1]
features_df['prob_cluster3'] = probs[:,2]
features_df['prob_cluster4'] = probs[:,3]
features_df['confidence'] = confidence


print(features_df[["image", "cluster_prediction", "prob_cluster1", "prob_cluster2", "prob_cluster3", "prob_cluster4", "confidence"]])



# ==========================================
# LOAD GROUND TRUTH AND PREDICTION DATASETS
# ==========================================

gt_df = pd.read_csv('/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/20260529_groundTruth.csv')

pred_df = features_df

# =========================================
# TRANSLATE CLUSTER NUMBER INTO COLOR NAME
# =========================================
mapping = {
    1: "light green",
    2: "standard green",
    3: "dark purple",
    4: "light purple"
}

pred_df['cluster_prediction'] = (
    pred_df['cluster_prediction'].map(mapping)
)

# =======================
# MERGE BOTH DATAFRAMES
# =======================
merged = gt_df.merge(
    pred_df,
    on="image",
    how="inner"
)

# ====================
# EVALUATE THE MODEL
# ====================

# Accuracy
acc = accuracy_score(
    merged['Dominant_color'],
    merged['cluster_prediction']
)
print(acc)

#Confusion matrix

cm = confusion_matrix(
    merged['Dominant_color'],
    merged['cluster_prediction']
)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
plt.savefig("/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/results/figures/20260529_conf_matrix_groundTruth")

# Classification report

y_true = merged['Dominant_color']
y_pred = merged['cluster_prediction']
print(
    classification_report(
        y_true,
        y_pred
    )
)