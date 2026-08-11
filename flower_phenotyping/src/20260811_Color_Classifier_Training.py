import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import joblib

# =========================
# CONFIG
# =========================
FEATURES_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/latent_features.npy"
FILENAMES_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/latent_filenames.npy"
LABELS_CSV_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/Labels.csv"

FIGURES_DIR = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/figures"
MODEL_OUT_PATH = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/models/color_classifier/color_classifier.joblib"

N_SPLITS = 3  
RANDOM_STATE = 42

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)


def load_dataset():
    """Load latent features, filenames and corresponding color labels."""
    features = np.load(FEATURES_PATH)
    filenames = np.load(FILENAMES_PATH)

    df_labels = pd.read_csv(LABELS_CSV_PATH, sep=";")
    label_dict = dict(zip(df_labels["filename"], df_labels["label"]))

    labels = np.array([label_dict[f] for f in filenames])

    print("Amount of samples:", len(labels))
    print("Class distribution:")
    print(pd.Series(labels).value_counts())

    return features, labels


def get_candidate_models():
    """Define the pipelines (scaler + classifier) to compare."""
    candidates = {
        "logreg": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "svm": SVC(kernel="rbf", class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    }

    pipelines = {
        name: Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        for name, clf in candidates.items()
    }
    return pipelines


def compare_models(pipelines, features, labels, cv):
    """Evaluate each pipeline with cross validation and return the average score of each one."""
    mean_scores = {}
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="balanced_accuracy")
        print(f"{name}: average accuracy = {scores.mean():.3f}")
        mean_scores[name] = scores.mean()
    return mean_scores


def plot_confusion_matrix(labels, predictions):
    disp = ConfusionMatrixDisplay.from_predictions(labels, predictions, cmap="Blues")
    plt.title("Confusion matrix")
    out_path = os.path.join(FIGURES_DIR, "color_classifier_confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved in: {out_path}")


def main():
    features, labels = load_dataset()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    pipelines = get_candidate_models()
    mean_scores = compare_models(pipelines, features, labels, cv)

    best_name = max(mean_scores, key=mean_scores.get)
    print("\nBest model:", best_name)
    best_pipeline = pipelines[best_name]

    # Cross Validation Predictions
    predictions = cross_val_predict(best_pipeline, features, labels, cv=cv)

    print("\nReporte de clasificacion:")
    print(classification_report(labels, predictions))

    plot_confusion_matrix(labels, predictions)

    # Train final model
    best_pipeline.fit(features, labels)

    joblib.dump({"model": best_pipeline}, MODEL_OUT_PATH)
    print("\nModel saved in:", MODEL_OUT_PATH)


if __name__ == "__main__":
    main()