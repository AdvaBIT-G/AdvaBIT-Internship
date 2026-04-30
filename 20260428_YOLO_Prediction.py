from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("/home/martinez/flower_phenotyping/models/yolo/weights/best.pt")

results = model.predict(
    source="/home/martinez/flower_phenotyping/data/YOLO/Images_to_predict",
    imgsz=1024,
    conf=0.3,
    save=True  
)

# folder for masks
mask_dir = "/home/martinez/flower_phenotyping/results/YOLO/pred_masks"
os.makedirs(mask_dir, exist_ok=True)

for r in results:

    if r.masks is None:
        continue

    filename = os.path.basename(r.path)
    filename = os.path.splitext(filename)[0] + ".png"

    h, w = r.orig_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in r.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (w, h))
        m = (m > 0.5).astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, m)

    combined_mask = combined_mask.astype(np.uint8)

    cv2.imwrite(os.path.join(mask_dir, filename), combined_mask * 255)

print("✅ Images + masks saved")