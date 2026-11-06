from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

ROOT=Path(__file__).resolve().parents[1]
DATA_PATH=ROOT / "data" / "creditcard.csv"
MODEL_PATH=ROOT / "models" / "fraud_model.joblib"

df=pd.read_csv(DATA_PATH)

X=df.drop(columns=["Class"])
y=df["Class"]

X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model=Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
])

print("Training model...")
model.fit(X_train, y_train)
probabilities=model.predict_proba(X_test)[:, 1]
predictions=(probabilities >= 0.5).astype(int)
print("\n--- MODEL RESULTS ---")
print("Average Precision (PR-AUC):",
      round(average_precision_score(y_test, probabilities), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions, zero_division=0))
print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, predictions))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("\nModel saved to:", MODEL_PATH)
