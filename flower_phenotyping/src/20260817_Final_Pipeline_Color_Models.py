import os
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import joblib
from ultralytics import YOLO

###############################
# CONFIG
###############################

RAW_DIR = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/data/selected_raw/predict'

YOLO_WEIGHTS = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/yolo/best.pt'
AUTOENCODER_PATH = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/autoencoder/autoencoder_final.pth'
CLASSIFIER_PATH = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/color_classifier/color_classifier.joblib'
SVM_PATH = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/color/flower_color_model_svm.joblib'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_ORDER = [
    'green', 'yellow', 'orange', 'white', 'red', 'unknown', 'purple', 'median_h',
    'median_s', 'median_v', 'std_h', 'std_s', 'std_v'
]

#############################
# LOAD MODELS
#############################

def load_yolo():
    return YOLO(YOLO_WEIGHTS)

def load_autoencoder_and_logistic_regression():
    autoencoder = Autoencoder().to(device)
    checkpoint = torch.load(AUTOENCODER_PATH, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()

    saved = joblib.load(CLASSIFIER_PATH)
    logistic_regression = saved['model']
    return autoencoder, logistic_regression

def load_svm():
    return joblib.load(SVM_PATH)

#########################################
# YOLO SEGMENTATION FUNCTION
#########################################
def segment_image(yolo_model, image_path):
    """
    Returns:
    -combined_mask: binary mask, all the instances detected, merged
    -segmented_img: original image with the background removed
    
    """
    results = yolo_model.predict(
        source=image_path,
        imgsz=1024,
        conf=0.3,
        device='cpu',
        save=False,
    )

    r = results[0]

    if r.masks is None:
        return None, None

    h, w = r.orig_shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in r.masks.data:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8)*255
        combined_mask = np.maximum(combined_mask, m)
    
    img = r.orig_img.copy()
    segmented_img = cv2.bitwise_and(img, img, mask=combined_mask)

    return combined_mask, segmented_img

################################################
# PIXEL DATA EXTRACTION AND SVM MODEL FUNCTIONS
################################################
def classify_hsv(h, s, v):
    if 35 <= h <= 85:
        return "green"
    if 25 <= h <= 35:
        return "yellow"
    if 10 <= h <= 25:
        return "orange"
    if v > 200 and s < 50:
        return "white"
    if (0 <= h <= 10) or (170 <= h <= 179):
        return "red"
    if 130 <= h <= 170:
        return "purple"
    return "unknown"

def extract_hsv_features(image, combined_mask):
    valid_mask = combined_mask > 0
    if np.sum(valid_mask) == 0:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[valid_mask]

    labels = [classify_hsv(h, s, v) for h, s, v in pixels]
    counts = Counter(labels)
    total = len(labels)

    percentages = {c: 100 * counts.get(c, 0) / total for c in FEATURE_ORDER[:7]}

    h_values = pixels[:, 0]
    s_values = pixels[:, 1]
    v_values = pixels[:, 2]

    stats = {
        "median_h": np.median(h_values), "median_s": np.median(s_values), "median_v": np.median(v_values),
        "std_h": np.std(h_values), "std_s": np.std(s_values), "std_v": np.std(v_values),
    }

    all_features = {**percentages, **stats}
    return [all_features.get(k, 0) for k in FEATURE_ORDER]

def predict_svm(svm_model, features):
    X = np.array(features).reshape(1, -1)
    pred = svm_model.predict(X)[0]
    proba = svm_model.predict_proba(X)[0]
    probability = float(proba.max())
    return pred, probability

####################################
# AUTOENCODER FUNCTIONS
####################################
def preprocess_for_autoencoder(segmented_img):
    img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)

    mask = np.sum(img, axis=2) > 20
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    img = img[ymin:ymax + 1, xmin:xmax + 1]

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float32)

def predict_logistic_regression(autoencoder, logistic_regression, segmented_img):
    tensor = preprocess_for_autoencoder(segmented_img)
    if tensor is None:
        return None, None

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        _, encoded = autoencoder(tensor)
        feature = F.adaptive_avg_pool2d(encoded, output_size = 1)
        feature = feature.squeeze(-1).squeeze(-1)
        feature = feature.cpu().numpy()

    pred = logistic_regression.predict(feature)[0]
    proba = logistic_regression.predict_proba(feature)[0]
    probability = float(proba.max())

    return pred, probability

#########################
# PIPELINE
#########################

def process_image(image_path, yolo_model, autoencoder, logistic_regression, svm_model):
    combined_mask, segmented_img = segmented_img(yolo_model, image_path)
    if combined_mask is None:
        print(f'[WARNING] No flower detected on {image_path}')

    #SVM model

    svm_class, svm_prob = None, None
    hsv_features = extract_hsv_features(original_img, combined_mask)
    if hsv_features is not None:
        svm_class, svm_prob = predict_svm(svm_model, hsv_features)

    #Autoencoder + logistic regression model
    lr_class, lr_prob = predict_logistic_regression(autoencoder, logistic_regression, segmented_img)

    print(f'[{image_path}] SVM -> {svm_class} ({svm_prob}) | LR -> {lr_class} ({lr_prob})')

def main():
    yolo_model = load_yolo()
    autoencoder, logistic_regression = load_autoencoder_and_logistic_regression()
    svm_model = load_svm()

    image_files = [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.lower().endwith(('.jpg', '.jpeg', '.png'))
    ]
    for image_path in image_files:
        process_image(image_path, yolo_model, autoencoder, logistic_regression, svm_model)

if __name__ == '__main__':
    main()  