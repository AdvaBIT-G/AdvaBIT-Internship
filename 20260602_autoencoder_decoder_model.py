import numpy as np
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D 

TRAIN_DIR = "/home/martinez/flower_phenotyping/data/DINOv2/train"
TEST_DIR = "/home/martinez/flower_phenotyping/data/DINOv2/test"
VAL_DIR = "/home/martinez/flower_phenotyping/data/DINOv2/groundTruth"

# ===========
# DATASET
# ===========
images = []

for file in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, file)
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    images.append(img)

images = np.array(images)

x_train = images
y_train = images

inputs = tf.keras.Input(shape=(224, 224, 3))

# =====================
# ENCODER
# =====================
x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = MaxPooling2D(2)(x)

x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(2)(x)

x = Conv2D(128, 3, activation='relu', padding='same')(x)
encoded = MaxPooling2D(2)(x)

# =====================
# DECODER
# =====================
x = Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(encoded)
x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)

outputs = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.models.Model(inputs, outputs)

# ================
# COMPILE
# ================
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# =============
# TRAIN
# =============

autoencoder.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)


pred = autoencoder.predict(x_train[:5])

for i in range(5):
    plt.figure(figsize=(4,2))

    # original
    plt.subplot(1,2,1)
    plt.imshow(x_train[i])
    plt.title("Original")

    # reconstructed
    plt.subplot(1,2,2)
    plt.imshow(pred[i])
    plt.title("Reconstructed")

    plt.show()

autoencoder.summary()