import os
import cv2
import numpy as np
import pandas as pd

seg_dir = "/home/martinez/flower_phenotyping/data/annotations/YOLO_annotations/masks/segmented_color"
output_csv = "/home/martinez/flower_phenotyping/data/annotations/YOLO_annotations/20260501_color_features.csv"

rows = []

for file in os.listdir(seg_dir):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(seg_dir, file)
    img = cv2.imread(path)

    if img is None:
        continue

    # ====================================
    # EXTRACT ONLY FLOWER (NOT BACKGROUND)
    # ====================================
    mask = np.any(img != [0, 0, 0], axis=2)
    pixels = img[mask]

    if len(pixels) == 0:
        continue

    # ====================================
    # SAMPLE ONLY 100 PIXELS
    # ====================================
    if len(pixels) > 100:
        idx = np.random.choice(len(pixels), 100, replace=False)
        pixels = pixels[idx]

    # =========================
    # AVERAGE COLOR (BGR)
    # =========================
    mean_bgr = pixels.mean(axis=0)

    # =========================
    # COLOR IN HSV
    # =========================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels_hsv = hsv[mask]

    # aplicar mismo muestreo a HSV
    if len(pixels_hsv) > 100:
        pixels_hsv = pixels_hsv[idx]

    mean_hsv = pixels_hsv.mean(axis=0)

    rows.append({
        "image": file,
        "mean_b": mean_bgr[0],
        "mean_g": mean_bgr[1],
        "mean_r": mean_bgr[2],
        "mean_h": mean_hsv[0],
        "mean_s": mean_hsv[1],
        "mean_v": mean_hsv[2],
        "num_pixels_used": len(pixels)
    })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print("✅ 100 pixels per flower extracted and saved")