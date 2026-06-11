from ultralytics import YOLO
import os
import cv2
import numpy as np

# model
model = YOLO("/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/models/yolo/weights/best.pt")

# paths
input_dir = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/raw"
output_dir = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/YOLO_annotations/masks"

mask_dir = os.path.join(output_dir, "masks_binary")
seg_dir = os.path.join(output_dir, "segmented_color")

os.makedirs(mask_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

# prediction
print(model.task)
results = model.predict(
    source=input_dir,
    imgsz=1024,
    conf=0.3,
    save=False
)

for r in results:

    filename = os.path.basename(r.path)
    name = os.path.splitext(filename)[0]

    h, w = r.orig_shape

    # =========================
    # BINARY MASK
    # =========================
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    if r.masks is None:
      print(f"⚠️ No mask for {filename}")
      continue

    if r.masks is not None:
        for mask in r.masks.data:
            m = mask.cpu().numpy()
            m = cv2.resize(m, (w, h))
            m = (m > 0.3).astype(np.uint8)
            combined_mask = np.logical_or(combined_mask, m)

    combined_mask = combined_mask.astype(np.uint8) * 255

    # save binary mask
    cv2.imwrite(os.path.join(mask_dir, name + ".png"), combined_mask)

    # ==============================
    # SEGMENTED IMAGEN (REAL COLOR)
    # ==============================
    img = r.orig_img.copy()

    segmented = cv2.bitwise_and(img, img, mask=combined_mask)

    cv2.imwrite(os.path.join(seg_dir, name + ".png"), segmented)

print("✅ Binary masks and segmented images saved")