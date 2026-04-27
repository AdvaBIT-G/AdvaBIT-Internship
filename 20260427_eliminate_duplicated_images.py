import os

origin_folder = "/home/martinez/flower_phenotyping/Series04"
new_folder = "/home/martinez/flower_phenotyping/data/raw"

# Obtain file names 
origin_files = set(
    f for f in os.listdir(origin_folder)
    if os.path.isfile(os.path.join(origin_folder, f))
)

new_files = [
    f for f in os.listdir(new_folder)
    if os.path.isfile(os.path.join(new_folder, f))
]

# Search coincidences and remove duplicates in the origin folder 
for file in origin_files:
    if file in new_files:
        full_path = os.path.join(origin_folder, file)
        os.remove(full_path)
        print(f"Removed: {full_path}")

print("Done.")