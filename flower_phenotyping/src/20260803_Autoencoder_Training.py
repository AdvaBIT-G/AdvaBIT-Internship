"""
Autoencoder Training.

This script trains the autoencoder using the segmented flower images and
saves both intermediate checkpoints and the final trained model for feature extraction.

"""

import os
import torch
from torch.utils.data import DataLoader

from autoencoder_arch import FlowerDataset, Autoencoder

# =========================
# CONFIG
# =========================

# Input data and output paths
MASK_DIR = "/home/gmartinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/data/autoencoder/train_masks"
CHECKPOINT_DIR = "/home/gmartinez/internship_howest/AdvaBIT-Internship/flower_phenotyping/models/autoencoder"
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "autoencoder_final.pth")

# Training parameters
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-4
SAVE_EVERY = 20  # Save a checkpoint every N epochs, in addition to the final one

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    """Train the autoencoder and save the trained model."""

    # Select GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and create batches
    dataset = FlowerDataset(MASK_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer and reconstruction loss
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass and reconstruction loss
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)

            # Backpropagation and parameter update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f} - Avg: {avg_loss:.4f}")

        # Saves a checkpoint every SAVE_EVERY epochs (in addition to the final one)
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save the final trained model
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, FINAL_MODEL_PATH)

    print(f"✅ Training completed. Final model saved in: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
