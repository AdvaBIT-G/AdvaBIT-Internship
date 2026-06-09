import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, MultiHeadAttention, Add
from ultralytics import YOLO
from keras.callbacks import ModelCheckpoint

# =========================
# CONFIG
# =========================

RAW_DIR = "/home/martinez/flower_phenotyping/data/DINOv2/train"
MASK_DIR = "/home/martinez/flower_phenotyping/data/DINOv2/masks"

# =====================================
# FLOWER SEGMENTATION USING YOLO MODEL
# =====================================
 
yolo_model = YOLO("/home/martinez/flower_phenotyping/models/yolo/weights/best.pt")

results = yolo_model.predict(
    source= RAW_DIR,
    imgsz=1024,
    conf=0.3,
    device='cpu',
    save=True,
    stream=True  
)

# folder for masks
mask_dir = MASK_DIR
os.makedirs(mask_dir, exist_ok=True)

for r in results:
    if r.orig_img is None:
        print(f"Empty image: {r.path}")
        continue
    if r.masks is None:
        print("No masks")
        continue

    original_name = os.path.basename(r.path)
    stem = os.path.splitext(original_name)[0]

    h, w = r.orig_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in r.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, m)

   #Save segmented image (real color)
    img = r.orig_img.copy()
    segmented = cv2.bitwise_and(img, img, mask=combined_mask)
    seg_path = os.path.join(MASK_DIR, f"{stem}.png")
    cv2.imwrite(seg_path, segmented)

print("✅ Segmented images saved")


# ===========
# DATASET
# ===========
images = []

for file in os.listdir(MASK_DIR):
    path = os.path.join(MASK_DIR, file)
    img = cv2.imread(path)
    mask = np.sum(img, axis=2) > 20
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        continue

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

   
    crop = img[ymin:ymax+1, xmin:xmax+1]


    crop = cv2.resize(crop, (224, 224))

    crop = crop.astype(np.float32) / 255.0

    images.append(crop)

images = np.array(images)

x_train = images
y_train = images

inputs = tf.keras.Input(shape=(224, 224, 3))

# =====================
# ENCODER
# =====================
x1 = Conv2D(256, 3, activation='relu', padding='same')(inputs)
p1 = MaxPooling2D(2)(x1)

x2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
p2 = MaxPooling2D(2)(x2)

attn = MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(p2, p2)

x2_attn = Add()([p2, attn])

x3 = Conv2D(64, 3, activation='relu', padding='same')(x2_attn)
encoded = MaxPooling2D(2)(x3)

# ===================
# BOTTLENECK
# ===================

b = Conv2D(256, 3, activation='relu', padding='same')(encoded)

# =====================
# DECODER
# =====================
x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(b)
x = Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(x)

outputs = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(inputs, outputs)

# ================
# COMPILE
# ================
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mae'
)

# =====================
# Save every 10 epochs
# =====================


checkpoint = ModelCheckpoint(
    filepath='/home/martinez/flower_phenotyping/models/autoencoder/checkpoints/model_epoch_{epoch:03d}.keras',
    save_freq='epoch',
    save_weights_only=False
)

class SaveEvery10Epochs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.model.save(
                f'/home/martinez/flower_phenotyping/models/autoencoder/checkpoints/model_epoch_{epoch+1:03d}.keras'
            )


# =============
# TRAIN
# =============


autoencoder.fit(
    x_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    callbacks=[SaveEvery10Epochs()]
)


pred = autoencoder.predict(x_train[:50])
pred = np.clip(pred, 0.0, 1.0)

for i in range(50):
    plt.figure(figsize=(4,2))

    # original
    plt.subplot(1,2,1)
    plt.imshow(x_train[i])
    plt.title("Original")

    # reconstructed
    plt.subplot(1,2,2)
    plt.imshow(pred[i])
    plt.title("Reconstructed")
    plt.savefig(f'/home/martinez/flower_phenotyping/results/figures/20260603_autoencoder_reconstruction_{i}.png', bbox_inches='tight')
    plt.show()
    plt.close()


autoencoder.summary()
