import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist

# 1. Load data
df = pd.read_csv("/home/martinez/flower_phenotyping/data/annotations/YOLO_annotations/20260501_color_features.csv")

# 2. Select HSV columns
X = df[['mean_h', 'mean_s', 'mean_v']].values

# 3. (Opcional pero recomendado) Manejar hue circular
# Convertir H a seno/coseno si está en grados (0–360)
H_rad = np.deg2rad(X[:, 0])
H_sin = np.sin(H_rad)
H_cos = np.cos(H_rad)

X_transformed = np.column_stack((H_sin, H_cos, X[:, 1], X[:, 2]))

# 4. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

# 5. Hierarchical clustering
Z = linkage(X_scaled, method='ward')

# 6. Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260504_color_dendrogram.png")

# 7. Elbow method (aprox)
wss = []
K_range = range(1, 10)

for k in K_range:
    labels = fcluster(Z, k, criterion='maxclust')
    
    # centroid calculation
    centroids = np.array([
        X_scaled[labels == i].mean(axis=0)
        for i in range(1, k+1)
    ])
    
    # sum of quadratic distances intra-cluster
    dist = 0
    for i in range(1, k+1):
        cluster_points = X_scaled[labels == i]
        dist += np.sum((cluster_points - centroids[i-1])**2)
    
    wss.append(dist)

# 8. Plot elbow
plt.figure()
plt.plot(K_range, wss, marker='o')
plt.title("Elbow Method (Hierarchical Clustering)")
plt.xlabel("Cluster number (k)")
plt.ylabel("WSS (Intra-cluster variance)")
plt.savefig("/home/martinez/flower_phenotyping/results/figures/20260504_elbow_plot.png")

# 9. Choose k
k_opt = 7
clusters = fcluster(Z, k_opt, criterion='maxclust')

df['cluster'] = clusters
df.to_csv("/home/martinez/flower_phenotyping/results/20260504_color_cluster_results.csv", index=False)

print(df.head())