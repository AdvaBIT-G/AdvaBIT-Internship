import os

# Folders
png_folder = "/home/martinez/flower_phenotyping/results/metrics/groundTruth/gt_masks"
jpg_folder = "/home/martinez/flower_phenotyping/data/annotations/YOLO/Predict_images"

# Obtain names of the .png files
png_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(png_folder)
    if f.lower().endswith(".png")
}

print(f"PNG found: {len(png_names)}")

# Find JPGs and remove of the same exists in png format
deleted = 0

for f in os.listdir(jpg_folder):
    if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"):
        base_name = os.path.splitext(f)[0]

        if base_name in png_names:
            path = os.path.join(jpg_folder, f)
            os.remove(path)
            deleted += 1
            print(f"Removed: {f}")

print(f"\nTotal removed: {deleted}")
print("DONE.")