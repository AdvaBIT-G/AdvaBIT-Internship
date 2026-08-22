"""
The objective of this pipeline is to process one image at a time instead of all the images of a folder.
The pipeline is similar to Final_Pipeline_Color_Models.py, but adapted to just process one image and save the results
in a dictionary.

"""

import sys
import json
from collections import Counter
 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import joblib
from ultralytics import YOLO
 
from autoencoder_arch import Autoencoder
 
# ###########################
# CONFIG
# ###########################

YOLO_WEIGHTS      = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/yolo/best.pt'
AUTOENCODER_PATH  = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/autoencoder/autoencoder_final.pth'
CLASSIFIER_PATH   = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/color_classifier/color_classifier.joblib'
SVM_PATH          = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/color/flower_color_model_svm.joblib'

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
FEATURE_ORDER = ['green', 'yellow', 'orange', 'white', 'red', 'unknown', 'purple']
 
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
        m = (m > 0.5).astype(np.uint8) * 255
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
 
    percentages = {c: 100 * counts.get(c, 0) / total for c in FEATURE_ORDER}
    return [percentages.get(k, 0) for k in FEATURE_ORDER]
 
 
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
        feature = F.adaptive_avg_pool2d(encoded, output_size=1)
        feature = feature.squeeze(-1).squeeze(-1)
        feature = feature.cpu().numpy()
 
    pred = logistic_regression.predict(feature)[0]
    proba = logistic_regression.predict_proba(feature)[0]
    probability = float(proba.max())
 
    return pred, probability
 
 
#########################
# PIPELINE
#########################
 
def main():
    result = {
        "image_path": None,
        "svm_class": None,
        "svm_probability": None,
        "lr_class": None,
        "lr_probability": None,
        "error": None,
    }
 
    if len(sys.argv) < 2:
        result["error"] = "No image path."
        print(json.dumps(result))
        sys.exit(1)
 
    image_path = sys.argv[1]
    result["image_path"] = image_path
 
    
    yolo_model = load_yolo()
    autoencoder, logistic_regression = load_autoencoder_and_logistic_regression()
    svm_model = load_svm()
 
    combined_mask, segmented_img = segment_image(yolo_model, image_path)
    if combined_mask is None:
        result["error"] = "No flower detected in the image."
        print(json.dumps(result))
        return
 
    original_img = cv2.imread(image_path)
 
    # SVM model
    hsv_features = extract_hsv_features(original_img, combined_mask)
    if hsv_features is not None:
        svm_class, svm_prob = predict_svm(svm_model, hsv_features)
        result["svm_class"] = str(svm_class)
        result["svm_probability"] = float(svm_prob)
 
    # Autoencoder + Logistic Regression
    lr_class, lr_prob = predict_logistic_regression(
        autoencoder, logistic_regression, segmented_img
    )
    if lr_class is not None:
        result["lr_class"] = str(lr_class)
        result["lr_probability"] = float(lr_prob)
 

    print(json.dumps(result))
 
 
if __name__ == '__main__':
    main()