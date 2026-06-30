
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================

RAW_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/selected_raw/train"
MASK_DIR = "/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/autoencoder/train_masks"

os.makedirs(MASK_DIR, exist_ok=True)

# =========================
# 1. YOLO SEGMENTATION
# =========================

yolo_model = YOLO("/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/models/yolo/best.pt")

results = yolo_model.predict(
    source=RAW_DIR,
    imgsz=1024,
    conf=0.3,
    device=0,
    save=True,
    stream=True
)

for r in results:
    if r.orig_img is None or r.masks is None:
        continue

    stem = os.path.splitext(os.path.basename(r.path))[0]

    h, w = r.orig_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in r.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (w, h))
        m = (m > 0.5).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, m)

    img = r.orig_img.copy()
    segmented = cv2.bitwise_and(img, img, mask=combined_mask)

    cv2.imwrite(os.path.join(MASK_DIR, f"{stem}.png"), segmented)

print("✅ YOLO segmentation done")

# =========================
# 2. DATASET
# =========================

class FlowerDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)

        mask = np.sum(img, axis=2) > 20
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return self.__getitem__((idx + 1) % len(self.files))

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        crop = img[ymin:ymax+1, xmin:xmax+1]
        crop = cv2.resize(crop, (224, 224))
        crop = crop.astype(np.float32) / 255.0

        crop = np.transpose(crop, (2, 0, 1))  # CHW

        return torch.tensor(crop, dtype=torch.float32)

dataset = FlowerDataset(MASK_DIR)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# =========================
# 3. AUTOENCODER MODEL
# =========================

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 256, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 256, 3, padding=1)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 256, 2, stride=2)

        self.out = nn.Conv2d(256, 3, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)

        x = attn_out.permute(0, 2, 1).view(b, c, h, w)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        encoded = x  # latent space

        # Bottleneck
        x = F.relu(self.bottleneck(x))

        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))

        x = torch.sigmoid(self.out(x))

        return x, encoded

# =========================
# 4. TRAINING SETUP
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.L1Loss()

# =========================
# 5. TRAINING LOOP
# =========================

epochs = 200

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        recon, _ = model(batch)

        loss = loss_fn(recon, batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, f"/home/martinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/models/checkpoint_epoch_{epoch+1}.pth")

# ================================
# 6. RECONSTRUCTION VISUALIZATION
# ================================

model.eval()

batch = next(iter(dataloader)).to(device)
recon, _ = model(batch)

batch = batch.cpu().numpy()
recon = recon.detach().cpu().numpy()

for i in range(min(10, len(batch))):
    plt.figure(figsize=(4, 2))

    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(batch[i], (1, 2, 0)))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(recon[i], (1, 2, 0)))
    plt.title("Reconstructed")

    plt.show()

# =========================
# 7. FEATURE EXTRACTION
# =========================

features = []

model.eval()

with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)

        _, encoded = model(batch)

        features.append(encoded.cpu().numpy())

features = np.concatenate(features, axis=0)
features = features.reshape(features.shape[0], -1)

# =========================
# 8. PCA + t-SNE
# =========================

features = PCA(n_components=50).fit_transform(features)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(features_2d[:, 0], features_2d[:, 1], s=5)
plt.title("Latent space t-SNE")
plt.show()

