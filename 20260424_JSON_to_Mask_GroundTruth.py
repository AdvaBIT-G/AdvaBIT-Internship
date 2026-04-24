import json
import numpy as np
import cv2
import os

# =========================
# PATHS
# =========================
input_dir = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/masks"
output_dir = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/gt_masks"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOOP FILES
# =========================
for file in os.listdir(input_dir):

    # 🔴 SOLO JSON (IMPORTANTE)
    if not file.endswith(".json"):
        continue

    path = os.path.join(input_dir, file)

    # =========================
    # LOAD JSON
    # =========================
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print("❌ JSON error:", file, e)
        continue

    # =========================
    # IMAGE SIZE
    # =========================
    info = data.get("info", {})
    h = int(info.get("height", 0))
    w = int(info.get("width", 0))

    if h == 0 or w == 0:
        print("❌ Invalid size:", file)
        continue

    mask = np.zeros((h, w), dtype=np.uint8)

    objects = data.get("objects", [])

    print(f"Processing {file} -> objects: {len(objects)}")

    drawn = 0

    # =========================
    # OBJECT LOOP
    # =========================
    for obj in objects:

        seg = obj.get("segmentation", None)

        if seg is None:
            continue

        try:
            pts = np.array(seg, dtype=np.float32)

            # 🔥 FIX ISAT FORMATS (nested or flat)
            pts = pts.reshape(-1, 2)

            # remove invalid values
            pts = pts[~np.isnan(pts).any(axis=1)]
            pts = pts[~np.isinf(pts).any(axis=1)]

            if len(pts) < 3:
                continue

            # 🔥 scale if normalized
            if pts.max() <= 1.5:
                pts[:, 0] *= w
                pts[:, 1] *= h

            # 🔥 clip to image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            pts = pts.astype(np.int32).reshape((-1, 1, 2))

            # =========================
            # DRAW (CRITICAL FIX)
            # =========================
            cv2.fillPoly(mask, [pts], 255)

            drawn += 1

        except Exception as e:
            print("⚠️ skip object:", e)

    # =========================
    # SAVE MASK
    # =========================
    out_name = file.replace(".json", ".png")
    out_path = os.path.join(output_dir, out_name)

    cv2.imwrite(out_path, mask)

    # =========================
    # DEBUG CHECK
    # =========================
    if drawn == 0:
        print("⚠️ EMPTY MASK:", file)

    if np.sum(mask) == 0:
        print("⚠️ WARNING: completely empty mask:", file)

print("✅ DONE - masks generated successfully")