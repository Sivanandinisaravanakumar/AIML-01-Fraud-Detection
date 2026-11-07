from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
MODEL_DIR = ROOT / "models" / "ieee_cis"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading IEEE-CIS features...")
df = pd.read_csv(DATA_PATH)

target = "isFraud"

categorical_cols = [
    "card1", "card2", "card3", "card5", "card6",
    "addr1", "addr2", "DeviceType", "DeviceInfo"
]

numeric_cols = [
    "TransactionDT", "TransactionAmt",
    "transaction_hour", "transaction_day",
    "card_txn_count_1h", "card_txn_count_24h",
    "card_avg_amount_30d", "amount_vs_card_average",
    "device_changed", "address_changed", "device_missing"
]

feature_cols = numeric_cols + categorical_cols

X = df[feature_cols].copy()
y = df[target].astype(int)

for col in categorical_cols:
    X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)

for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit preprocessing only on training data
medians = X_train[numeric_cols].median()

X_train.loc[:, numeric_cols] = X_train[numeric_cols].fillna(medians)
X_valid.loc[:, numeric_cols] = X_valid[numeric_cols].fillna(medians)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train_cat = pd.DataFrame(
    encoder.fit_transform(X_train[categorical_cols]),
    columns=categorical_cols,
    index=X_train.index
)
X_valid_cat = pd.DataFrame(
    encoder.transform(X_valid[categorical_cols]),
    columns=categorical_cols,
    index=X_valid.index
)

X_train = pd.concat(
    [X_train[numeric_cols].astype("float64"), X_train_cat],
    axis=1
)[feature_cols].astype("float64")

X_valid = pd.concat(
    [X_valid[numeric_cols].astype("float64"), X_valid_cat],
    axis=1
)[feature_cols].astype("float64")

scale_pos_weight = float(
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Fraud ratio:", round(float(y.mean()), 5))
print("Scale pos weight:", round(scale_pos_weight, 2))

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=2,
    random_state=42,
    early_stopping_rounds=30
)

print("Training XGBoost model...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=50
)

probabilities = model.predict_proba(X_valid)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "PR_AUC": float(average_precision_score(y_valid, probabilities)),
    "ROC_AUC": float(roc_auc_score(y_valid, probabilities)),
    "Precision": float(precision_score(y_valid, predictions, zero_division=0)),
    "Recall": float(recall_score(y_valid, predictions, zero_division=0)),
    "F1": float(f1_score(y_valid, predictions, zero_division=0)),
    "Confusion_Matrix": confusion_matrix(y_valid, predictions).tolist(),
    "Best_Iteration": int(model.best_iteration),
    "Threshold": 0.5
}

model.save_model(str(MODEL_DIR / "ieee_xgboost.json"))
joblib.dump(encoder, MODEL_DIR / "ordinal_encoder.joblib")
joblib.dump(medians, MODEL_DIR / "numeric_medians.joblib")

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump({
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "all": feature_cols
    }, f, indent=4)

print("\nIEEE-CIS MODEL TRAINING COMPLETE")
print(json.dumps(metrics, indent=4))
print("Saved model to:", MODEL_DIR)
