import os
import random
import shutil


all_img_dir = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/raw"
train_dir = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/YOLO/raw_data"

out_dir = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/results/YOLO/groundTruth/images"

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