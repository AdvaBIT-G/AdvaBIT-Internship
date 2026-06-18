import os
import shutil

# Rutas
json_folder = r"/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/data/annotations/json"
raw_folder = r"/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/data/raw"
new_folder = r"/home/martinez/internship_howest/AdvaBIT-Internship/disease_phenotyping/data/YOLO/raw_data"

# Create folder if it does not exist
os.makedirs(new_folder, exist_ok=True)

# Obtain base names
names = {
    os.path.splitext(file)[0]
    for file in os.listdir(json_folder)
    if os.path.isfile(os.path.join(json_folder, file))
}

# Search for coincidences and copy them into raw_data folder
for file in os.listdir(raw_folder):
    filepath = os.path.join(raw_folder, file)

    if os.path.isfile(filepath):
        base_name = os.path.splitext(file)[0]

        if base_name in json_folder:
            shutil.copy2(filepath, new_folder)
            print(f"Copiado: {file}")

print("Process completed.")