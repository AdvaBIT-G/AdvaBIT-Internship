import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from joblib import dump

CSV = '/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260518_color_training_dataset.csv'

df = pd.read_csv(CSV)

########################
# DATA EXPLORATION
########################
df.info()

#Basic statistics
df.describe()

#Number of null values and duplicated values
df.isnull().sum()
df.duplicated().sum()

#Drop image column
df = df.drop(columns=['image'])

#Visualize the distribution of the variables
for column in df.columns:
    if column in ['cluster']:
        continue
    sns.displot(df, x = column)
    plt.title(column)

sns.catplot(data=df, x='cluster', kind="count")
plt.title('Cluster')

#############################
# MODEL TRAINING PRE-STEPS
#############################

#Define features and target
X = df.drop(columns=['cluster'])
y = df['cluster']

#Split into train and test (stratified for disbalanced classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


######################
# MODEL TRAINING
######################

# One pipeline for all models
models = {
    'SVM': SVC(kernel='rbf', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
}

#Encoding for XGBoost, as it needs the classes starting from 0, not 1
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train) 
y_test_encoded = le.transform(y_test)

#Iterate over the different models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')
    print(f"{name}: {scores.mean():.3f}")

# XGBoost with label encoding
xgb = XGBClassifier()
xgb.fit(X_train_scaled, y_train_encoded)
scores = cross_val_score(xgb, X_train_scaled, y_train_encoded, cv=5, scoring='f1_macro')
print(f"XGBoost: {scores.mean():.3f}")

###########################
# MODEL EVALUATION
###########################

# SVM
svm = models['SVM']

y_pred_svm = svm.predict(X_test_scaled)

print('\n===== SVM =====')
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Predictions train/test
y_train_pred_svm = svm.predict(X_train_scaled)
y_test_pred_svm = svm.predict(X_test_scaled)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred_svm)
test_acc = accuracy_score(y_test, y_test_pred_svm)

train_f1 = f1_score(y_train, y_train_pred_svm, average='macro')
test_f1 = f1_score(y_test, y_test_pred_svm, average='macro')

print('\nSVM OVERFITTING CHECK')
print(f'Train Accuracy: {train_acc:.3f}')
print(f'Test Accuracy : {test_acc:.3f}')

print(f'Train F1 Macro: {train_f1:.3f}')
print(f'Test F1 Macro : {test_f1:.3f}')

# Random Forest
rf = models['Random Forest']

y_pred_rf = rf.predict(X_test_scaled)

print('\n===== RANDOM FOREST =====')
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Predictions train/test
y_train_pred_rf = rf.predict(X_train_scaled)
y_test_pred_rf = rf.predict(X_test_scaled)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred_rf)
test_acc = accuracy_score(y_test, y_test_pred_rf)

train_f1 = f1_score(y_train, y_train_pred_rf, average='macro')
test_f1 = f1_score(y_test, y_test_pred_rf, average='macro')

print('\nRANDOM FOREST OVERFITTING CHECK')
print(f'Train Accuracy: {train_acc:.3f}')
print(f'Test Accuracy : {test_acc:.3f}')

print(f'Train F1 Macro: {train_f1:.3f}')
print(f'Test F1 Macro : {test_f1:.3f}')


# XGBoost
y_pred_xgb = xgb.predict(X_test_scaled)

print('\n===== XGBOOST =====')

print(
    classification_report(
        y_test_encoded,
        y_pred_xgb,
        target_names=le.classes_.astype(str)
    )
)

cm_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)

sns.heatmap(
    cm_xgb,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Predictions train/test
y_train_pred_xgb = xgb.predict(X_train_scaled)
y_test_pred_xgb = xgb.predict(X_test_scaled)

# Metrics
train_acc = accuracy_score(y_train_encoded, y_train_pred_xgb)
test_acc = accuracy_score(y_test_encoded, y_test_pred_xgb)

train_f1 = f1_score(
    y_train_encoded,
    y_train_pred_xgb,
    average='macro'
)

test_f1 = f1_score(
    y_test_encoded,
    y_test_pred_xgb,
    average='macro'
)

print('\nXGBOOST OVERFITTING CHECK')
print(f'Train Accuracy: {train_acc:.3f}')
print(f'Test Accuracy : {test_acc:.3f}')

print(f'Train F1 Macro: {train_f1:.3f}')
print(f'Test F1 Macro : {test_f1:.3f}')

############################
# HYPERPARAMETER TUNING
############################

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)

#######################################
# MODEL TRAINING WITH BEST PARAMETERS
#######################################

# SVM
svm = SVC(kernel='rbf', class_weight='balanced', C=10, gamma='scale')
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

#####################################
# MODEL METRICS WITH BEST PARAMETERS
#####################################

print('\n===== SVM =====')
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Predictions train/test
y_train_pred_svm = svm.predict(X_train_scaled)
y_test_pred_svm = svm.predict(X_test_scaled)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred_svm)
test_acc = accuracy_score(y_test, y_test_pred_svm)

train_f1 = f1_score(y_train, y_train_pred_svm, average='macro')
test_f1 = f1_score(y_test, y_test_pred_svm, average='macro')

print('\nSVM OVERFITTING CHECK')
print(f'Train Accuracy: {train_acc:.3f}')
print(f'Test Accuracy : {test_acc:.3f}')

print(f'Train F1 Macro: {train_f1:.3f}')
print(f'Test F1 Macro : {test_f1:.3f}')

#########################################################
# MODEL TRAINING WITH DEFAULT PARAMETERS (FINAL VERSION)
#########################################################

# SVM
svm = SVC(kernel='rbf', class_weight='balanced')
svm.fit(X_train_scaled, y_train)

################################
# SAVE THE MODEL AND THE SCALER
################################

dump(svm, 'flower_color_model_svm.joblib')
dump(scaler, 'scaler.joblib')