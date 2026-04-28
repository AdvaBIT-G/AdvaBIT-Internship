import os
import random
import shutil

#Remove images without json files associated.
raw_dir = "/home/martinez/flower_phenotyping/data/annotations/YOLO/raw_data"

files = os.listdir(raw_dir)

# separate by type
images = [f for f in files if f.endswith((".jpg", ".png"))]
jsons = set(f.replace(".json", "") for f in files if f.endswith(".json"))

deleted = 0

for img in images:
    name = os.path.splitext(img)[0]
    
    if name not in jsons:
        path = os.path.join(raw_dir, img)
        os.remove(path)
        deleted += 1
        print(f"Removed: {img}")

print(f"Total removed: {deleted}")

all_img_dir = "/home/martinez/flower_phenotyping/data/raw"
train_dir = "/home/martinez/flower_phenotyping/data/annotations/YOLO/raw_data"

out_dir = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/images"

os.makedirs(out_dir, exist_ok=True)

# -----------------------
# LIST OF FILES
# -----------------------
all_imgs = set(os.listdir(all_img_dir))
train_imgs = set(os.listdir(train_dir))
print("All:", len(all_imgs))
print("Train:", len(train_imgs))
# Exclude images from training
candidates = list(all_imgs - train_imgs)

print("Images available for ground truth:", len(candidates))

# -----------------------
# RANDOM SELECTION
# -----------------------
N = 150

if len(candidates) < N:
    raise ValueError(f"Not enough images ({len(candidates)})")

selected = random.sample(candidates, N)

print("Selected:", len(selected))

# -----------------------
# COPY
# -----------------------
for img in selected:
    src = os.path.join(all_img_dir, img)
    dst = os.path.join(out_dir, img)
    shutil.copy(src, dst)

print("Images copied to:", out_dir)