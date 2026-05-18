import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

CSV = "/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260508_mask_color_percentages.csv"

FEATURES = [
    "green",
    "yellow",
    "orange",
    "white",
    "red",
    "purple",
    "unknown"
]

LABEL = "Cluster"

THRESHOLD_UNKNOWN = 40  # % to label the image as problematic


# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(CSV)

# =========================================================
# CLEAN LABEL COLUMN
# =========================================================

# convert to number
df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")

# convert 0 en NaN
df.loc[df[LABEL] == 0, LABEL] = np.nan

print("\nUnique labels:")
print(df[LABEL].unique())


# =========================================================
# NORMALIZE (SUM = 100)
# =========================================================

sum = df[FEATURES].sum(axis=1)

# avoid division by zero
sum = sum.replace(0, np.nan)

for col in FEATURES:
    df[col] = (df[col] / sum) * 100

df = df.fillna(0)

# =========================================================
# DETECT IMAGES WITH BIG PERCENTAGE OF UNKNOWN COLOR
# =========================================================

df["lots_unknown"] = df["unknown"] > THRESHOLD_UNKNOWN

print("\nImages with a big percentage of unknown color:")
print(df["lots_unknown"].sum())

# =========================================================
# USE ONLY IMAGES LABELLED FOR CENTROIDS
# =========================================================

df_train = df[df[LABEL].isin([1, 2, 3, 4, 5])].copy()

print("\nLabeled images:")
print(len(df_train))

# =========================================================
# CALCULATE CENTROIDS 
# =========================================================

centroids = df_train.groupby(LABEL)[FEATURES].median()

print("\nCENTROIDS:")
print(centroids)

# =========================================================
# EUCLIDEAN DISTANCE
# =========================================================

def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# =========================================================
# CLASSIFICATION
# =========================================================

predictions = []
confidences = []
min_distances = []

for idx, row in df.iterrows():

    vector = row[FEATURES].values.astype(float)

    distances = {}

    # compare with centroids
    for clase, centroid in centroids.iterrows():

        d = distance(vector, centroid.values)
        distances[clase] = d

    # sort
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    best_class = sorted_distances[0][0]
    best_distance = sorted_distances[0][1]

    if len(sorted_distances) > 1:
        second_distance = sorted_distances[1][1]
        confidence = 1 - (best_distance / second_distance)
    else:
        confidence = 1

    predictions.append(best_class)
    confidences.append(round(confidence, 3))
    min_distances.append(round(best_distance, 3))

# =========================================================
# RESULTS
# =========================================================

df["predicted_class"] = predictions
df["confidence"] = confidences
df["centroid_distance"] = min_distances

# =========================================================
# DOUBTFUL IMAGES 
# =========================================================

doubtful = df[
    (df["confidence"] < 0.15) |
    (df["lots_unknown"] == True)
].copy()

# =========================================================
# EXPORT
# =========================================================

df.to_csv("/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260518_classified_images.csv", index=False)
doubtful.to_csv("/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260518_doubtful_images.csv", index=False)
centroids.to_csv("/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260518_centroids.csv")

# =========================================================
# SUMMARY
# =========================================================

print("\n====================================")
print("CLASSIFICATION COMPLETED")
print("====================================")

print("\nClass distribution:")
print(df["predicted_class"].value_counts())

print("\nDoubtful_images:")
print(len(doubtful))

print("\nGenerated files:")
print("- 20260518_classified_images.csv")
print("- 20260518_doubtful_images.csv")
print("- 20260518_centroids.csv")
