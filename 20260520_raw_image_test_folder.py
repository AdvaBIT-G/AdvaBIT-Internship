import os
import shutil

# PNG folder (test masks)
png_folder = r"/home/martinez/flower_phenotyping/data/annotations/color_annotations/test"

# JPEG folder to search the raw images
jpg_folder = r"/home/martinez/flower_phenotyping/data/raw"

# Final folder
final_folder = r"/home/martinez/flower_phenotyping/data/full_model_testing/test"

# Create final folder if it does not exist
os.makedirs(final_folder, exist_ok=True)

# Loop over PNG files
for file in os.listdir(png_folder):
    if file.lower().endswith(".png"):

        # Name without extension
        base_name = os.path.splitext(file)[0]

        # Search same name in .jpg
        file_jpg = base_name + ".jpg"
        path_jpg = os.path.join(jpg_folder, file_jpg)

        # If exists, copy
        if os.path.exists(path_jpg):
            destination = os.path.join(final_folder, file_jpg)
            shutil.copy2(path_jpg, destination)
            print(f"Copied: {file_jpg}")
        else:
            print(f"Not found: {file_jpg}")

print("Process ended.")