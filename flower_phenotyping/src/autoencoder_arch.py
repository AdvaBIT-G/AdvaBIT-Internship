"""
Dataset and architecture of the Autoencoder.
This script will be used in 20260803_Autoencoder_Training.py and 20260803_Feature_Extraction_and_Analysis.py

"""

import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# =========================
# DATASET
# =========================

class FlowerDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing flower images.
    
    Images are cropped to remove background, resized to 224x224,
    normalized, and converted into PyTorch tensors.
    """
    def __init__(self, folder):
        # Store image paths for later loading
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # Convert image from OpenCV BGR format to RGB
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate a simple mask to separate flower from background
        mask = np.sum(img, axis=2) > 20
        ys, xs = np.where(mask)

        # Skip images where no flower pixels are detected
        if len(xs) == 0:
            return self.__getitem__((idx + 1) % len(self.files))

        # Crop flower region
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        crop = img[ymin:ymax + 1, xmin:xmax + 1]

        # Resize and normalize image
        crop = cv2.resize(crop, (224, 224))
        crop = crop.astype(np.float32) / 255.0

        # Convert from HWC (OpenCV) to CHW (PyTorch)
        crop = np.transpose(crop, (2, 0, 1))

        return torch.tensor(crop, dtype=torch.float32)


# =========================
# MODEL
# =========================

class Autoencoder(nn.Module):
    """
    Convolutional autoencoder with an attention module.

    The encoder extracts features, the bottleneck stores the latent
    representation, and the decoder reconstructs the input image.

    The latent representation is returned for downstream analysis.
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 256, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Attention layer to capture spatial dependencies
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck (latent representation)
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

        # Apply attention on flattened spatial features
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)

        # Restore image dimensions
        x = attn_out.permute(0, 2, 1).view(b, c, h, w)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Store latent representation before decoding
        encoded = x 

        # Bottleneck
        x = F.relu(self.bottleneck(x))

        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        x = torch.sigmoid(self.out(x))

        return x, encoded
