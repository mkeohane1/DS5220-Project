import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)
import matplotlib.pyplot as plt

# 1. Load data
df_full = pd.read_csv("cleaned_news.csv")    

# 2. Define full and 20% sample
datasets = {
    'Full dataset': df_full,
    '20% Sample': df_full.sample(frac=0.2, random_state=42)
}

# 3. Start ROC plot
plt.figure(figsize=(7, 5))

for label, df in datasets.items():
    # 4. Vectorize
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vect.fit_transform(df["text"])
    y = df["label"]
    
    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 6. Fit Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 7. Predict & evaluate
    y_pred = model.predict(X_test)
    print(f"\n=== Results for {label} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # 8. Feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print("\nTop 10 feature importances:")
    for i in indices:
        print(f"{vect.get_feature_names_out()[i]}: {importances[i]:.4f}")
    
    # 9. ROC curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc_score:.3f})")

# 10. Random-chance baseline
plt.plot([0, 1], [0, 1], '--', color='gray', label="Random Chance (AUC=0.5)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Full vs. 20% Sample — Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
