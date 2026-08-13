import os
import sys
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import joblib

from autoencoder_arch import Autoencoder

# paths of the already trained models
AUTOENCODER_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/autoencoder/autoencoder_final.pth"
CLASSIFIER_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/color_classifier/color_classifier.joblib"
OUTPUT_CSV_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/autoencoder_color_predictions.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    """
    Load the autoencoder and the color classifier already trained
   
    """
    # load the trained autoencoder
    autoencoder = Autoencoder().to(device)
    checkpoint = torch.load(AUTOENCODER_PATH, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    # load the trained color classifier
    saved = joblib.load(CLASSIFIER_PATH)
    classifier = saved["model"]

    return autoencoder, classifier


def preprocess_image(path):
    """
    Preprocess the images the same way as done during the training. Images are converted to RGB,
    cropped to only use the flower and not the background, and then resized and normalized.
   
    """
    # read image and convert from BGR to RGB
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # crop the flower (same criteria used during training)
    mask = np.sum(img, axis=2) > 20
    ys, xs = np.where(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    img = img[ymin:ymax + 1, xmin:xmax + 1]

    # resize and normalize
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW

    return torch.tensor(img, dtype=torch.float32)


def predict_color(path, autoencoder, classifier):
    """
    Function used to predict the color class of the flower. First the images are preprocessed,
    then they pass through the autoencoder, and at the end the color classifier predicts the color class.
   
    """
    image = preprocess_image(path)
    image = image.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        _, encoded = autoencoder(image)
        feature = F.adaptive_avg_pool2d(encoded, output_size=1)
        feature = feature.squeeze(-1).squeeze(-1)
        feature = feature.cpu().numpy()

    color = classifier.predict(feature)[0]
    return color


def main():
    image_paths = sys.argv[1:]

    autoencoder, classifier = load_models()

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    #To save the color class results in a csv file with the filename and the resulting color.
    with open(OUTPUT_CSV_PATH, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "predicted_color"])

        for path in image_paths:
            color = predict_color(path, autoencoder, classifier)
            print(path, "-> color:", color)
            writer.writerow([os.path.basename(path), color])

    print("\nPredictions saved to:", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()