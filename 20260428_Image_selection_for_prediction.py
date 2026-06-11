import os
import shutil

# Folders
origin_folder = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/raw"
json_folder = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/YOLO/raw_data"
new_folder = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/YOLO/Images_to_predict"

# Create new folder if it does not exist
os.makedirs(new_folder, exist_ok=True)

# Image extensions
img_ext = (".jpg", ".jpeg", ".png", ".bmp")

# Obtain images from A
images_A = {
    f for f in os.listdir(origin_folder)
    if f.lower().endswith(img_ext)
}

# Obtain names of JSON in B
json_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(json_folder)
    if f.lower().endswith(".json")
}

# Filter images without JSON file associated
filtered_images = [
    img for img in images_A
    if os.path.splitext(img)[0] not in json_names
]

print(f"Total images in A: {len(images_A)}")
print(f"Total JSON in B: {len(json_names)}")
print(f"Images without JSON: {len(filtered_images)}")

# Copy images
for image in filtered_images:
    origin = os.path.join(origin_folder, image)
    destiny = os.path.join(new_folder, image)

    if os.path.isfile(origin):
        shutil.copy2(origin, destiny)

print("DONE.")