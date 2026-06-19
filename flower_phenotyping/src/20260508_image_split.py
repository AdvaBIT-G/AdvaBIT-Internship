import os
import random
import shutil

# =========================
# CONFIG
# =========================

SOURCE_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/DINOv2/raw"
OUTPUT_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/DINOv2"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# =========================
# LIST FILES
# =========================

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

random.seed(SEED)
random.shuffle(files)

# =========================
# CALCULATE SPLITS
# =========================

n_total = len(files)

n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_files = files[:n_train]
val_files = files[n_train:n_train + n_val]
test_files = files[n_train + n_val:]

splits = {
    "train": train_files,
    "groundTruth": val_files,
    "test": test_files
}

# =========================
# CREATE FOLDERS
# =========================

for split_name in splits:

    os.makedirs(
        os.path.join(OUTPUT_DIR, split_name),
        exist_ok=True
    )

# =========================
# COPY FILES
# =========================

for split_name, split_files in splits.items():

    for file_name in split_files:

        src = os.path.join(SOURCE_DIR, file_name)

        dst = os.path.join(
            OUTPUT_DIR,
            split_name,
            file_name
        )

        shutil.copy2(src, dst)

# =========================
# SUMMARY
# =========================

print("Train:", len(train_files))
print("groundTruth:", len(val_files))
print("Test:", len(test_files))

print("Done.")