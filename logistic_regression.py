'''
Michelle Keohane, Ishan Chotalia
DS 5220
Spring 2025
Project
'''

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# change working directory
os.chdir(r"C:\Users\mkeo9\OneDrive\NEU\DS 5220 Supervised Machine Learning\Project")

df = pd.read_csv("cleaned_news.csv")

# define features and labels
X = df["text"]
y = df["label"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X_vectorized, y, test_size=0.2, random_state=42
	)

# create and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))