import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from autoencoder_arch import FlowerDataset, Autoencoder

# =========================
# CONFIG
# =========================

# Input data and trained model
MASK_DIR = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/data/autoencoder/train_masks"

CHECKPOINT_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/autoencoder/autoencoder_final.pth"

# Output paths
FIGURES_DIR = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/figures"
FEATURES_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/latent_features.npy"
FILENAMES_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/latent_filenames.npy"

# Labelled dataset
LABELS_CSV_PATH = None

RUN_TAG = "20260807_1"  # to save the different figures with a unique name
BATCH_SIZE = 16
TSNE_PERPLEXITY = 10

# Silhouette score: range of k values to test
SIL_K_VALUES = [2, 3, 4, 5, 6]

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)


def load_model(device):
    """Load the trained autoencoder from a checkpoint."""
    model = Autoencoder().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH} (epoch {checkpoint.get('epoch', '?')})")
    return model


def visualize_reconstructions(model, dataloader, device, n=10):
    """Compare original and reconstructed images."""
    batch, _filenames = next(iter(dataloader))
    batch = batch.to(device)
    recon, _ = model(batch)

    batch = batch.cpu().numpy()
    recon = recon.detach().cpu().numpy()

    n = min(n, len(batch))
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))

    for i in range(n):
        axes[i, 0].imshow(np.transpose(batch[i], (1, 2, 0)))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.transpose(recon[i], (1, 2, 0)))
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_reconstructed_images.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def extract_features(model, dataloader, device):
    """Extract latent features from the encoder, keeping track of the filename of each sample."""
    features = []
    filenames = []

    with torch.no_grad():
        for batch, batch_filenames in dataloader:
            batch = batch.to(device)
            _, encoded = model(batch)

            # Reduce each feature map to a single value using global average pooling
            pooled = F.adaptive_avg_pool2d(encoded, output_size=1)
            pooled = pooled.squeeze(-1).squeeze(-1)

            features.append(pooled.cpu().numpy())
            filenames.extend(batch_filenames)  # batch_filenames is a list/tuple of strings

    features = np.concatenate(features, axis=0)
    filenames = np.array(filenames)

    np.save(FEATURES_PATH, features)
    np.save(FILENAMES_PATH, filenames)
    print(f"Features saved in: {FEATURES_PATH} (shape={features.shape})")
    print(f"Filenames saved in: {FILENAMES_PATH} (n={len(filenames)})")

    return features, filenames


def load_external_labels(filenames, csv_path=LABELS_CSV_PATH):
    """
    Load a CSV with columns 'filename,label' and align it to the order of `filenames`.

    Returns a numpy array of labels in the same order as `filenames`, or None if
    csv_path is not set. Filenames present in `filenames` but missing from the CSV
    are assigned the label "unknown".
    """
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError("LABELS_CSV_PATH must have 'filename' and 'label' columns")

    lookup = dict(zip(df["filename"], df["label"]))

    missing = [f for f in filenames if f not in lookup]
    if missing:
        print(f"Warning: {len(missing)} filenames not found in {csv_path}, labeled as 'unknown'. "
              f"Example missing: {missing[:5]}")

    labels = np.array([lookup.get(f, "unknown") for f in filenames])
    return labels


def plot_pca(features, labels=None):
    """Reduce feature dimensionality using PCA."""
    n_components = min(50, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        codes, categories = pd.factorize(labels)
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=codes, cmap="tab10", s=25)
        handles, _ = scatter.legend_elements()
        plt.legend(handles, categories, title="Label")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=25)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_latent_space_pca.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

    return reduced, pca


def plot_pca_variance(pca):
    """Plot explained variance (individual and cumulative) per principal component."""
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_components = len(explained)
    components = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Individual explained variance as bars
    ax1.bar(components, explained, alpha=0.6, color="steelblue", label="Individual")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")

    # Cumulative explained variance as a line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="firebrick", marker="o", markersize=3, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance ratio")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.95, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("PCA explained variance")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_pca_explained_variance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def plot_tsne(features, labels=None, perplexity=TSNE_PERPLEXITY):
    """Visualize the latent space using t-SNE."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        codes, categories = pd.factorize(labels)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=codes, cmap="tab10", s=15)
        handles, _ = scatter.legend_elements()
        plt.legend(handles, categories, title="Label")
    else:
        plt.scatter(features_2d[:, 0], features_2d[:, 1], s=15)
    plt.title("Latent space t-SNE")
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_latent_space_tsne.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

def compute_silhouette_scores(features, k_values=SIL_K_VALUES):
    """Fit KMeans for each k, compute the silhouette score, and keep the labels per k."""
    scores = {}
    labels_by_k = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        scores[k] = score
        labels_by_k[k] = labels
        print(f"k={k}: silhouette score = {score:.4f}")

    # Plot silhouette scores vs k
    plt.figure(figsize=(7, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs. k")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_silhouette_scores.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

    best_k = max(scores, key=scores.get)
    print(f"Best k according to silhouette score: {best_k} (score={scores[best_k]:.4f})")

    return scores, labels_by_k, best_k


def main():
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and trained model
    dataset = FlowerDataset(MASK_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = load_model(device)

    # Evaluate reconstruction quality
    visualize_reconstructions(model, dataloader, device)

    # Extract latent features
    features = extract_features(model, dataloader, device)

    # Clustering quality: silhouette score for k=2..6, computed on the raw latent space
    scores, labels_by_k, best_k = compute_silhouette_scores(features)
    best_labels = labels_by_k[best_k]  # cluster assignment for the best k, used to color the plots below

    # PCA: scatter plot (colored by cluster) + explained variance plot
    reduced, pca = plot_pca(features, labels=best_labels)
    plot_pca_variance(pca)

    # t-SNE (on PCA-reduced features, colored by cluster)
    plot_tsne(reduced, labels=best_labels)


if __name__ == "__main__":
    main()