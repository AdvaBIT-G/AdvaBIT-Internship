import os
import cv2
import numpy as np
import pandas as pd

seg_dir = "/home/martinez/flower_phenotyping/data/annotations/YOLO_annotations/masks/segmented_color"
output_csv = "/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260507_color_features.csv"

rows = []

for file in os.listdir(seg_dir):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(seg_dir, file)
    img = cv2.imread(path)

    if img is None:
        continue

    # =========================
    # MASK
    # =========================
    mask = np.any(img > 5, axis=2)

    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        continue

    # =========================
    # SAMPLE
    # =========================
    if len(coords) > 500:
        coords = coords[np.random.choice(len(coords), 500, replace=False)]

    # =========================
    # PIXELS
    # =========================
    bgr_pixels = img[coords[:, 0], coords[:, 1]]

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[coords[:, 0], coords[:, 1]]

    # =========================
    # REAL LAB (CIELAB)
    # =========================
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Convert OpenCV LAB -> real CIELAB
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)  # L*: 0-100
    lab[..., 1] = lab[..., 1] - 128.0            # a*: -128 to 127
    lab[..., 2] = lab[..., 2] - 128.0            # b*: -128 to 127

    lab_pixels = lab[coords[:, 0], coords[:, 1]]

    # =========================
    # FEATURES
    # =========================

    # BGR
    median_bgr = np.median(bgr_pixels, axis=0)
    std_bgr = np.std(bgr_pixels, axis=0)

    # HSV
    median_hsv = np.median(hsv_pixels, axis=0)
    std_hsv = np.std(hsv_pixels, axis=0)

    # LAB
    median_lab = np.median(lab_pixels, axis=0)
    std_lab = np.std(lab_pixels, axis=0)

    rows.append({
        "image": file,

        # =====================
        # BGR
        # =====================
        "median_b": median_bgr[0],
        "median_g": median_bgr[1],
        "median_r": median_bgr[2],

        "std_b": std_bgr[0],
        "std_g": std_bgr[1],
        "std_r": std_bgr[2],

        # =====================
        # HSV
        # =====================
        "median_h": median_hsv[0],
        "median_s": median_hsv[1],
        "median_v": median_hsv[2],

        "std_h": std_hsv[0],
        "std_s": std_hsv[1],
        "std_v": std_hsv[2],

        # =====================
        # CIELAB REAL
        # =====================
        "median_l": median_lab[0],
        "median_a": median_lab[1],
        "median_b_lab": median_lab[2],

        "std_l": std_lab[0],
        "std_a": std_lab[1],
        "std_b_lab": std_lab[2],

        "num_pixels_used": len(coords)
    })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print("✅ Features (BGR + HSV + real CIELAB) extracted")