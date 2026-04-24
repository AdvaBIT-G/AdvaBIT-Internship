import os
import random
import cv2
import numpy as np
import shutil

img_dir = "/home/martinez/flower_phenotyping/data/raw"
mask_dir = "/home/martinez/flower_phenotyping/src/runs/segment/predict"

out_img = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/images"
out_mask = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/masks"

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_mask, exist_ok=True)

files = sorted(os.listdir(mask_dir))

# -----------------------
# 1. RANDOM (80)
# -----------------------
random_sample = random.sample(files, 80)

# -----------------------
# 2. MASK SIZES
# -----------------------
areas = []
for f in files:
    mask = cv2.imread(os.path.join(mask_dir, f), 0)
    area = (mask > 0).sum()
    areas.append((f, area))

areas_sorted = sorted(areas, key=lambda x: x[1])

small = [f for f, _ in areas_sorted[:25]]
large = [f for f, _ in areas_sorted[-25:]]

# -----------------------
# 3. COMPLEXITY (components)
# -----------------------
def num_components(mask):
    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    return labels

complex_cases = []
for f in files:
    mask = cv2.imread(os.path.join(mask_dir, f), 0)
    mask_bin = mask > 0
    comps = num_components(mask_bin)
    complex_cases.append((f, comps))

complex_sorted = sorted(complex_cases, key=lambda x: x[1], reverse=True)
complex_top = [f for f, _ in complex_sorted[:20]]

# -----------------------
# 4. JOIN ALL
# -----------------------
selected = set()
selected.update(random_sample)
selected.update(small)
selected.update(large)
selected.update(complex_top)

selected = list(selected)[:150]

print("Seleccionadas:", len(selected))

# -----------------------
# 5. COPY FILES
# -----------------------
for f in selected:
    img_name = f.replace(".png", ".jpg")  # ajusta si es png/jpg

    shutil.copy(os.path.join(img_dir, img_name),
                os.path.join(out_img, img_name))

    shutil.copy(os.path.join(mask_dir, f),
                os.path.join(out_mask, f))

print("Dataset ready in /home/martinez/flower_phenotyping/results/metrics/groundTruth")