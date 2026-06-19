import pandas as pd

df_colors = pd.read_csv("/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/20260508_mask_color_percentages.csv")
df_hsv = pd.read_csv("/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/results/20260518_4_cluster_results.csv")

# cleaning names
for df in [df_colors, df_hsv]:
    df["image"] = (
        df["image"]
        .str.strip()
        .str.lower()
    )


# left join
merged = pd.merge(
    df_colors,
    df_hsv,
    on="image",
    how="left"
)

print("Rows:", len(merged))

# rows with missing data
missing = merged[merged.isna().any(axis=1)]

print("Rows with NaN:")
print(missing)

merged = merged.drop(['median_b', 'median_g', 'median_r', 'std_b', 'std_g', 'std_r', 'Cluster', 'num_pixels_used'], axis=1)

merged.to_csv("/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/20260518_color_training_dataset.csv", index=False)