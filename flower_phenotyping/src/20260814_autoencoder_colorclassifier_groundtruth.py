import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

CSV_PREDICTIONS = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/results/autoencoder_color_predictions.csv'
CSV_REAL_LABELS = '/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/data/annotations/color_annotations/Pred_masks_real_labels.csv'

pred = pd.read_csv(CSV_PREDICTIONS, sep=',')
gt = pd.read_csv(CSV_REAL_LABELS, sep=';')

# Merge using the filename column
df = pd.merge(pred, gt, on='filename', how='inner')

print(f'Predictions: {len(pred)}, Ground Truth: {len(gt)}, Coincidences: {len(df)}')

####################
# General Accuracy
####################

acc = accuracy_score(df['real_color'], df['predicted_color'])
print(f"Accuracy: {acc:.2%}")

##########################
# Classification report
##########################
print(classification_report(df['real_color'], df['predicted_color']))

#######################
# Confusion matrix
#######################
labels = sorted(df['real_color'].unique())
cm = confusion_matrix(df['real_color'], df['predicted_color'], labels=labels)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig("20260814_confusion_matrix_autoencoder_color.png", dpi=150)
plt.show()