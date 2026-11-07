
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/ieee_cis/processed/train_features.csv"
MODEL_DIR = ROOT / "models/ieee_cis"

torch.manual_seed(42)
np.random.seed(42)

print("Loading IEEE-CIS data...")
df = pd.read_csv(DATA_PATH)

# Reproduce the exact transaction split used by XGBoost.
indices = np.arange(len(df))
y = df["isFraud"].astype(int).to_numpy()

train_idx, valid_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

train_df = df.iloc[train_idx].copy()
valid_df = df.iloc[valid_idx].copy()
y_valid = y[valid_idx]

print("Training transactions:", len(train_df))
print("Validation transactions:", len(valid_df))

# Load existing XGBoost artifacts.
with open(MODEL_DIR / "feature_columns.json") as f:
    columns = json.load(f)

numeric_cols = columns["numeric"]
categorical_cols = columns["categorical"]
feature_cols = columns["all"]

encoder = joblib.load(MODEL_DIR / "ordinal_encoder.joblib")
medians = joblib.load(MODEL_DIR / "numeric_medians.joblib")

xgb_model = XGBClassifier()
xgb_model.load_model(str(MODEL_DIR / "ieee_xgboost.json"))

def prepare_xgb(frame):
    X = frame[feature_cols].copy()

    for col in categorical_cols:
        X[col] = (
            X[col].astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric = X[numeric_cols].fillna(medians)

    categorical = pd.DataFrame(
        encoder.transform(X[categorical_cols]),
        columns=categorical_cols,
        index=X.index,
    )

    return pd.concat([numeric, categorical], axis=1)[feature_cols].astype("float64")

print("Scoring XGBoost...")
xgb_valid = prepare_xgb(valid_df)
xgb_scores = xgb_model.predict_proba(xgb_valid)[:, 1]

# Build the graph from training transactions ONLY.
print("Building training-only graph...")

card_cols = ["card1", "card2", "card3", "card5", "card6"]

def make_card_key(frame):
    keys = frame[card_cols].fillna("unknown").astype(str).agg("_".join, axis=1)
    missing = frame[card_cols].isna().all(axis=1)
    keys.loc[missing] = (
        "unknown_card_" + frame.loc[missing, "TransactionID"].astype(str)
    )
    return keys

train_df["card_key"] = make_card_key(train_df)
valid_df["card_key"] = make_card_key(valid_df)

train_df["card_node"] = "card_" + train_df["card_key"]
train_df["device_node"] = (
    "device_" + train_df["DeviceInfo"].fillna("unknown").astype(str)
)

valid_df["card_node"] = "card_" + valid_df["card_key"]
valid_df["device_node"] = (
    "device_" + valid_df["DeviceInfo"].fillna("unknown").astype(str)
)

all_nodes = pd.Index(
    pd.concat([
        train_df["card_node"],
        train_df["device_node"],
    ]).unique()
)

node_to_id = {node: i for i, node in enumerate(all_nodes)}

card_ids = train_df["card_node"].map(node_to_id).to_numpy()
device_ids = train_df["device_node"].map(node_to_id).to_numpy()

edge_index = torch.tensor(
    np.vstack([
        np.concatenate([card_ids, device_ids]),
        np.concatenate([device_ids, card_ids]),
    ]),
    dtype=torch.long,
)

numeric_features = [
    "TransactionAmt",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_avg_amount_30d",
    "device_changed",
    "address_changed",
    "device_missing",
]

card_stats = train_df.groupby("card_node")[numeric_features].mean()
card_stats["transaction_count"] = train_df.groupby("card_node").size()

card_stats = card_stats.replace([np.inf, -np.inf], np.nan)
card_stats = card_stats.fillna(card_stats.median()).fillna(0)

means = card_stats.mean()
stds = card_stats.std().replace(0, 1)
card_stats = (card_stats - means) / stds

node_features = torch.zeros((len(all_nodes), 9), dtype=torch.float32)

node_features[:, 0] = torch.tensor([
    1.0 if str(node).startswith("card_") else 0.0
    for node in all_nodes
])

for card, row in card_stats.iterrows():
    node_features[node_to_id[card], 1:] = torch.tensor(
        row.to_numpy(dtype=np.float32)
    )

labels = torch.full((len(all_nodes),), -1, dtype=torch.long)

card_labels = train_df.groupby("card_node")["isFraud"].max()

for card, label in card_labels.items():
    labels[node_to_id[card]] = int(label)

graph = Data(
    x=node_features,
    edge_index=edge_index,
    y=labels,
)

# GNN model architecture.
class FraudGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(9, 16)
        self.conv2 = GCNConv(16, 16)
        self.output = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.output(x)

model = FraudGCN()

train_mask = graph.y >= 0
train_labels = graph.y[train_mask]

counts = torch.bincount(train_labels, minlength=2).float()
weights = counts.sum() / (2 * counts)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

print("Training leakage-aware GNN...")

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    logits = model(graph.x, graph.edge_index)

    loss = F.cross_entropy(
        logits[train_mask],
        graph.y[train_mask],
        weight=weights,
    )

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# Generate GNN risk scores using training-only graph.
model.eval()

with torch.no_grad():
    node_scores = torch.softmax(
        model(graph.x, graph.edge_index), dim=1
    )[:, 1].numpy()

training_prior = float(train_df["isFraud"].mean())

def get_gnn_score(row):
    card = row["card_node"]
    device = row["device_node"]

    if card in node_to_id:
        return float(node_scores[node_to_id[card]])

    if device in node_to_id:
        return float(node_scores[node_to_id[device]])

    return training_prior

gnn_scores = valid_df.apply(get_gnn_score, axis=1).to_numpy()

# Fixed 80:20 starting blend.
hybrid_scores = 0.8 * xgb_scores + 0.2 * gnn_scores

def calculate_metrics(y_true, scores, threshold=0.5):
    predictions = (scores >= threshold).astype(int)

    return {
        "PR_AUC": float(average_precision_score(y_true, scores)),
        "ROC_AUC": float(roc_auc_score(y_true, scores)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
    }

xgb_threshold = json.load(
    open(MODEL_DIR / "threshold.json")
)["best_threshold"]

# Find an exploratory validation threshold for the hybrid.
precision, recall, thresholds = precision_recall_curve(
    y_valid, hybrid_scores
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))
hybrid_threshold = float(thresholds[best_index])

results = {
    "validation_transactions": int(len(y_valid)),
    "training_transactions": int(len(train_idx)),
    "xgboost_threshold": float(xgb_threshold),
    "hybrid_weight_xgboost": 0.8,
    "hybrid_weight_gnn": 0.2,
    "xgboost_at_0.5": calculate_metrics(y_valid, xgb_scores, 0.5),
    "xgboost_at_saved_threshold": calculate_metrics(
        y_valid, xgb_scores, xgb_threshold
    ),
    "hybrid_at_0.5": calculate_metrics(y_valid, hybrid_scores, 0.5),
    "hybrid_best_validation_threshold": hybrid_threshold,
    "hybrid_at_best_validation_threshold": calculate_metrics(
        y_valid, hybrid_scores, hybrid_threshold
    ),
    "note": (
        "Exploratory validation results. Hybrid threshold is selected "
        "on this validation set; use an untouched test set for final claims."
    ),
}

output_path = MODEL_DIR / "hybrid_metrics.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nHYBRID EVALUATION COMPLETE")
print(json.dumps(results, indent=4))
print("Metrics saved to:", output_path)
