import os
import cv2
import numpy as np

# =========================
# PATHS
# =========================
gt_dir = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/gt_masks"
pred_dir = "/home/martinez/flower_phenotyping/src/runs/segment/predict"

# =========================
# METRICS
# =========================
ious = []
dices = []
precisions = []
recalls = []

# =========================
# LOOP GT
# =========================
for file in os.listdir(gt_dir):

    if not file.endswith(".png"):
        continue

    gt_path = os.path.join(gt_dir, file)

    # pred is JPG (same base name)
    pred_file = file.replace(".png", ".jpg")
    pred_path = os.path.join(pred_dir, pred_file)

    if not os.path.exists(pred_path):
        print(f"⚠️ Missing prediction: {pred_file}")
        continue

    # =========================
    # LOAD IMAGES
    # =========================
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"❌ Load error: {file}")
        continue

    # =========================
    # BINARIZE
    # =========================
    gt = (gt > 0).astype(np.uint8)

    # 🔥 IMPORTANT: predictions are grayscale image
    # threshold needed
    pred = (pred > 127).astype(np.uint8)

    # =========================
    # METRICS
    # =========================
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou = intersection / (union + 1e-7)
    dice = (2 * intersection) / (gt.sum() + pred.sum() + 1e-7)

    precision = intersection / (pred.sum() + 1e-7)
    recall = intersection / (gt.sum() + 1e-7)

    # =========================
    # STORE
    # =========================
    ious.append(iou)
    dices.append(dice)
    precisions.append(precision)
    recalls.append(recall)

    print(f"{file}")
    print(f"  IoU: {iou:.4f} | Dice: {dice:.4f} | P: {precision:.4f} | R: {recall:.4f}")

# =========================
# FINAL RESULTS
# =========================
print("\n===== FINAL RESULTS =====")
print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f}")