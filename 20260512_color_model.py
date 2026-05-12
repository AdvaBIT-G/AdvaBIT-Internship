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

CSV = '/home/martinez/flower_phenotyping/data/annotations/color_annotations/20260512_color_training_dataset.csv'

df = pd.read_csv(CSV)

#Data exploration
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

#Define features and target
X = df.drop(columns=['cluster'])
y = df['cluster']

#Split into train and test (stratified for disbalanced classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model training

# One pipeline for all models
models = {
    'SVM': SVC(kernel='rbf', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'XGBoost': XGBClassifier()
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

# XGBoost por separado con y encodeado
xgb = XGBClassifier()
xgb.fit(X_train_scaled, y_train_encoded)
scores = cross_val_score(xgb, X_train_scaled, y_train_encoded, cv=5, scoring='f1_macro')
print(f"XGBoost: {scores.mean():.3f}")