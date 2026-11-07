from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
MODEL_DIR = ROOT / "models" / "ieee_cis"

df = pd.read_csv(DATA)

with open(MODEL_DIR / "feature_columns.json") as f:
    columns = json.load(f)

numeric_cols = columns["numeric"]
categorical_cols = columns["categorical"]
feature_cols = columns["all"]

X = df[feature_cols].copy()
y = df["isFraud"].astype(int)

for col in categorical_cols:
    X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)

for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

medians = joblib.load(MODEL_DIR / "numeric_medians.joblib")
X_valid.loc[:, numeric_cols] = X_valid[numeric_cols].fillna(medians)

encoder = joblib.load(MODEL_DIR / "ordinal_encoder.joblib")
encoded = encoder.transform(X_valid[categorical_cols])

X_cat = pd.DataFrame(encoded, columns=categorical_cols, index=X_valid.index)
X_valid = pd.concat(
    [X_valid[numeric_cols].astype("float64"), X_cat], axis=1
)[feature_cols].astype("float64")

model = XGBClassifier()
model.load_model(str(MODEL_DIR / "ieee_xgboost.json"))

probabilities = model.predict_proba(X_valid)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_valid, probabilities)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_index])

results = {
    "best_threshold": best_threshold,
    "precision": float(precision[best_index]),
    "recall": float(recall[best_index]),
    "f1_score": float(f1_scores[best_index]),
    "baseline_threshold": 0.5,
    "baseline_f1": float(f1_score(y_valid, (probabilities >= 0.5).astype(int)))
}

with open(MODEL_DIR / "threshold.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nTHRESHOLD OPTIMIZATION COMPLETE")
print(json.dumps(results, indent=4))
