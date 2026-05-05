import os
import shutil
import pandas as pd

# Config
csv_path = "C:/Users/GEMA/OneDrive - Pure Production AG/Documents/Internship Howest/flower_phenotyping/results/20260505_5_cluster_results.csv"
image_folder = "C:/Users/GEMA/OneDrive - Pure Production AG/Documents/Internship Howest/flower_phenotyping/data/annotations/YOLO_annotations/masks"
output_folder = "C:/Users/GEMA/OneDrive - Pure Production AG/Documents/Internship Howest/flower_phenotyping/results"
samples_per_cluster = 10

# Read CSV
df = pd.read_csv(csv_path)

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Cluster iteration
for cluster_value, group in df.groupby("cluster"):
    
    # Select up to 10 random rows
    sampled = group.sample(n=min(samples_per_cluster, len(group)), random_state=42)
    
    # Create cluster folder
    cluster_folder = os.path.join(output_folder, f"new_cluster_{cluster_value}")
    os.makedirs(cluster_folder, exist_ok=True)
    
    for _, row in sampled.iterrows():
        image_name = str(row["image"])
        
        # Assure .png extension
        if not image_name.endswith(".png"):
            image_name += ".png"
        
        src_path = os.path.join(image_folder, image_name)
        dst_path = os.path.join(cluster_folder, image_name)
        
        # Copy if exists
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Image not found: {src_path}")