"""
Feature Extraction and Analysis.

This script loads a trained autoencoder and performs:
    1. Reconstruction visualization
    2. Latent feature extraction
    3. PCA dimensionality reduction
    4. t-SNE visualization

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from autoencoder_arch import FlowerDataset, Autoencoder

# =========================
# CONFIG
# =========================

# Input data and trained model
MASK_DIR = "/home/gmartinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/autoencoder/train_masks"

CHECKPOINT_PATH = "/home/gmartinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/models/autoencoder/autoencoder_final.pth"

# Output paths
FIGURES_DIR = "/home/gmartinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/results/figures"
FEATURES_PATH = "/home/gmartinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/results/latent_features.npy"

RUN_TAG = "20260804"  # to save the different figures with a unique name
BATCH_SIZE = 16
TSNE_PERPLEXITY = 30

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
    batch = next(iter(dataloader)).to(device)
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
    """Extract latent features from the encoder."""
    features = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, encoded = model(batch)

            # Reduce each feature map to a single value using global average pooling
            pooled = F.adaptive_avg_pool2d(encoded, output_size=1)
            pooled = pooled.squeeze(-1).squeeze(-1)

            features.append(pooled.cpu().numpy())

    features = np.concatenate(features, axis=0)
    np.save(FEATURES_PATH, features)
    print(f"Features saved in: {FEATURES_PATH} (shape={features.shape})")
    return features


def plot_pca(features):
    """Reduce feature dimensionality using PCA."""
    reduced = PCA(n_components=50).fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_latent_space_pca.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

    return reduced


def plot_tsne(features, perplexity=TSNE_PERPLEXITY):
    """Visualize the latent space using t-SNE."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=5)
    plt.title("Latent space t-SNE")
    out_path = os.path.join(FIGURES_DIR, f"{RUN_TAG}_latent_space_tsne.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def main():
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and trained model
    dataset = FlowerDataset(MASK_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = load_model(device)

    # Evaluate reconstruction quality
    visualize_reconstructions(model, dataloader, device)

    # Extract latent features and visualize them
    features = extract_features(model, dataloader, device)
    reduced = plot_pca(features)
    plot_tsne(reduced)


if __name__ == "__main__":
    main()
