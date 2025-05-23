import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load pre‑cleaned & labeled data
df = pd.read_csv('cleaned_news.csv')  


# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.20,
    stratify=df['label'],
    random_state=42
)

# 3. sklearn Pipeline: TF‑IDF → XGB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        stop_words='english'
    )),
    ('xgb', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# 4. Hyperparameter grid
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth':    [3, 5],
    'xgb__learning_rate': [0.1, 0.05],
    'xgb__subsample':    [0.8, 1.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# 5. Fit & select best
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# 6. Evaluate
y_pred = grid.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['fake','real']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# 8. ROC Curve & AUC
y_prob = grid.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()