import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("/home/martinez/flower_phenotyping/data/annotations/YOLO_annotations/20260505_color_features.csv")

# =========================
# 2. SELECT FEATURES
# =========================
features = [
    'median_h', 'median_s', 'median_v',
    'std_h', 'std_s', 'std_v'
]

X = df[features].values
# =========================
# 3. HANDLE HUE CORRECTLY
# =========================
# OpenCV: H ∈ [0,179] → convertir a grados reales (0–360)
H_deg = X[:, 0] * 2

H_rad = np.deg2rad(H_deg)
H_sin = np.sin(H_rad)
H_cos = np.cos(H_rad)

# reconstruir feature matrix sin H original
X_transformed = np.column_stack((
    H_sin,
    H_cos,
    X[:, 1:]   # resto de features
))

# =========================
# 4. SCALE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

# =========================
# 5. CLUSTERING
# =========================
Z = linkage(X_scaled, method='ward')

# =========================
# 6. DENDROGRAM
# =========================
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260505_color_dendrogram2.png")
plt.close()

# =========================
# 7. ELBOW METHOD
# =========================
wss = []
K_range = range(2, 12)

for k in K_range:
    labels = fcluster(Z, k, criterion='maxclust')

    centroids = np.array([
        X_scaled[labels == i].mean(axis=0)
        for i in range(1, k+1)
    ])

    dist = 0
    for i in range(1, k+1):
        cluster_points = X_scaled[labels == i]
        dist += np.sum((cluster_points - centroids[i-1])**2)

    wss.append(dist)

plt.figure()
plt.plot(K_range, wss, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WSS")
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260505_elbow_plot2.png")
plt.close()

# =========================
# 8. CHOOSE K
# =========================
k_opt = 5
clusters = fcluster(Z, k_opt, criterion='maxclust')

# =============================
# 9. VISUALIZATION (OPTIMIZED)
# =============================
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df['median_h'] * 2,   # convert to real frades
    df['median_s'],
    c=clusters,
    cmap='tab10'
)

plt.xlabel("Hue (degrees)")
plt.ylabel("Saturation")
plt.title("Clusters in HSV space")
plt.colorbar(scatter)

plt.savefig("/home/martinez/flower_phenotyping/results/figures/hsv_clusters2.png", dpi=300)
plt.close()

# =========================
# 10. SAVE RESULTS
# =========================
df['cluster'] = clusters
df.to_csv("/home/martinez/flower_phenotyping/results/20260505_5_cluster_results.csv", index=False)

print(df.head())