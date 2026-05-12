import pandas as pd

df_colors = pd.read_csv("/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260508_mask_color_percentages.csv")
df_hsv = pd.read_csv("/home/martinez/flower_phenotyping/results/20260507_5_cluster_results.csv")

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

merged.to_csv("/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260512_color_training_dataset.csv", index=False)