import os
import cv2
import numpy as np

# =========================
# PATHS
# =========================
gt_dir = "/home/martinez/flower_phenotyping/results/YOLO/groundTruth/png_masks"
pred_dir = "/home/martinez/flower_phenotyping/results/YOLO/pred_masks" 

# =========================
# METRICS STORAGE
# =========================
ious = []
dices = []
precisions = []
recalls = []

# =========================
# LOOP
# =========================
for file in os.listdir(gt_dir):

    if not file.endswith(".png"):
        continue

    gt_path = os.path.join(gt_dir, file)
    pred_path = os.path.join(pred_dir, file)

    if not os.path.exists(pred_path):
        print(f"⚠️ Missing prediction: {file}")
        continue

    # =========================
    # LOAD
    # =========================
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"❌ Load error: {file}")
        continue

    # =========================
    # RESIZE (por seguridad)
    # =========================
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # =========================
    # BINARIZE
    # =========================
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    # =========================
    # EDGE CASE (ambos vacíos)
    # =========================
    if gt.sum() == 0 and pred.sum() == 0:
        iou = 1.0
        dice = 1.0
        precision = 1.0
        recall = 1.0
    else:
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