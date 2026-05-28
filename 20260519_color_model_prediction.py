import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from joblib import load
from ultralytics import YOLO
import shutil
import math
import matplotlib.pyplot as plt
import seaborn as sns

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
probs = svm.predict_proba(X)
confidence = probs.max(axis=1)

features_df["cluster_prediction"] = pred
features_df['prob_cluster1'] = probs[:,0]
features_df['prob_cluster2'] = probs[:,1]
features_df['prob_cluster3'] = probs[:,2]
features_df['prob_cluster4'] = probs[:,3]
features_df['confidence'] = confidence


print(features_df[["image", "cluster_prediction", "prob_cluster1", "prob_cluster2", "prob_cluster3", "prob_cluster4", "confidence"]])

features_df.to_csv("/home/martinez/flower_phenotyping/results/color/20260526_color_predictions.csv", index=False)

print("✅ Results saved to /home/martinez/flower_phenotyping/results/color/20260526_color_predictions.csv")

# ==========================================
# ASSIGN SPECIFIC COLOR TONE TO EACH SAMPLE
# ==========================================

COLORS = {
    # GREENS
    "lime green": (75, 100, 95),
    "neon green": (90, 100, 98),
    "bright green": (110, 95, 100),
    "forest green": (120, 80, 57),
    "dark forest green": (120, 50, 35),
    "olive green": (60, 45, 70),
    "dark olive": (70, 65, 35),
    "sage green": (115, 25, 78),
    "mint green": (142, 55, 76),
    "sea green": (146, 60, 56),
    "emerald green": (158, 92, 67),
    "turquoise green": (160, 95, 45),
    "yellow green": (62, 40, 72),
    "pale green": (95, 35, 74),
    "pastel green": (142, 50, 68),
    "grasshopper green": (128, 95, 68),
    "moss green": (95, 55, 40),
    "khaki green": (80, 30, 55),
    "chlorophyll green": (115, 85, 85),

    # PURPLES
    "lavender": (285, 96, 93),
    "violet": (280, 35, 85),
    "deep purple": (300, 100, 58),
    "dark purple": (300, 70, 22),
    "plum purple": (295, 65, 45),
    "grape purple": (303, 97, 76),
    "magenta purple": (310, 90, 70),
    "purple haze": (282, 45, 78),
    "black purple": (290, 60, 15),
    "royal purple": (275, 80, 60),

    # RED / PINK / PISTILS
    "orange pistil": (30, 100, 100),
    "burnt orange": (25, 80, 70),
    "amber": (40, 90, 85),
    "copper": (20, 75, 60),
    "salmon": (6, 40, 68),
    "rose": (328, 42, 70),
    "pink pistil": (345, 30, 76),

    # TRICHOMES / RESIN
    "frosty white": (0, 0, 95),
    "milky white": (0, 5, 88),
    "amber trichome": (35, 65, 80),
    "golden resin": (45, 85, 90),
}

# Distance to HSV

def distance_hsv(hsv1, hsv2):

    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    # Hue circular
    dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180

    # Normalization
    ds = abs(s1 - s2) / 100
    dv = abs(v1 - v2) / 100

    return math.sqrt((3*dh**2) + (1*ds**2) + (0.5*dv**2))

# Specific color 
def specific_color(h, s, v):

    hsv_sample = (h, s, v)

    best_color = None
    best_distance = float("inf")

    for name, hsv_ref in COLORS.items():

        distance = distance_hsv(hsv_sample, hsv_ref)

        if distance < best_distance:
            best_distance = distance
            best_color = name

    return best_color

#Normalize hsv values 
def normalize_hsv_opencv(h, s, v):

    # OpenCV -> standard
    h = h * 2
    s = (s / 255) * 100
    v = (v / 255) * 100

    return h, s, v


features_df["specific_color"] = features_df.apply(
    lambda row: specific_color(
        *normalize_hsv_opencv(
        row["median_h"],
        row["median_s"],
        row["median_v"]
        )
    ),
    axis=1
)

features_df.to_csv("/home/martinez/flower_phenotyping/results/color/20260527_color_predictions.csv", index=False)

# =====================================
# PLOT CONFIDENCE VS COLOR VARIABILITY
# =====================================

sns.set_style("whitegrid")


# LONG FORMAT

df_long = features_df.melt(
    id_vars=["confidence", "cluster_prediction"],
    value_vars=["std_h", "std_s", "std_v"],
    var_name="channel",
    value_name="std"
)

# FACET GRID

g = sns.FacetGrid(
    df_long,
    col="channel",
    hue="cluster_prediction",
    palette="tab10",
    height=4,
    aspect=1.1,
    sharex=False,
    sharey=True
)

g.map_dataframe(
    sns.scatterplot,
    x="std",
    y="confidence",
    s=40,
    alpha=0.75,
    edgecolor="none"
)

# LABELS

g.set_axis_labels("Color variability (std)", "SVM confidence")
g.set_titles("{col_name}")

# Legend
g.add_legend(title="Cluster")

# Ajuste layout
plt.tight_layout()

plt.show()
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260528_confidence_vs_color_var.png")

# ======================
# PLOT HUE DISTRIBUTION
# ======================

plt.figure(figsize=(8,5))

sns.histplot(
    data=features_df,
    x="median_h",         
    bins=30,
    kde=True,
    color="purple"
)

plt.title("Hue Distribution")
plt.xlabel("Hue")
plt.ylabel("Frequency")

plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260528_hue_distribution.png")

# ======================
# S vs V SCATTERPLOT
# ======================
plt.figure(figsize=(7,6))

sns.scatterplot(
    data=features_df,
    x="median_s",
    y="median_v",
    hue="cluster_prediction", 
    palette="tab10",
    alpha=0.7
)

plt.title("Saturation vs Value")
plt.xlabel("S (Saturation)")
plt.ylabel("V (Value)")

plt.grid(True, alpha=0.3)
plt.legend(title="Cluster")

plt.show()
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260528_s_vs_v_scatter.png")


# =========================
# H vs S
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(
    data=features_df,
    x="h",
    y="s",
    hue="cluster_prediction",
    palette="tab10",
    alpha=0.7,
    ax=axes[0]
)

axes[0].set_title("Hue vs Saturation")
axes[0].set_xlabel("H")
axes[0].set_ylabel("S")

plt.tight_layout()
plt.show()
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260528_h_vs_s_scatter.png")

# =========================
# H vs V
# =========================
sns.scatterplot(
    data=features_df,
    x="h",
    y="v",
    hue="cluster_prediction",
    palette="tab10",
    alpha=0.7,
    ax=axes[1]
)

axes[1].set_title("Hue vs Value")
axes[1].set_xlabel("H")
axes[1].set_ylabel("V")

plt.tight_layout()
plt.show()
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260528_h_vs_v_scatter.png")


# ==========================
# IMAGE FOLDER PER CLUSTER
# ==========================

#Paths
csv_file = '/home/martinez/flower_phenotyping/results/color/20260526_color_predictions.csv'
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