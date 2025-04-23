'''
Michelle Keohane, Ishan Chotalia
DS 5220
Spring 2025
Project
'''

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("data/cleaned_news.csv")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
	)

# create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# generate ROC curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))

# show which features are most important
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
print("Top 10 feature importances:")
for i in indices:
	print(f"{vectorizer.get_feature_names_out()[i]}: {feature_importances[i]:.4f}")

