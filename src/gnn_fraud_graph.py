
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
OUTPUT_DIR = ROOT / "data" / "ieee_cis" / "graph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading IEEE-CIS training data...")
df = pd.read_csv(DATA_PATH)

card_columns = ["card1", "card2", "card3", "card5", "card6"]
df["card_key"] = df[card_columns].fillna("unknown").astype(str).agg("_".join, axis=1)

missing_card = df[card_columns].isna().all(axis=1)
df.loc[missing_card, "card_key"] = (
    "unknown_card_" + df.loc[missing_card, "TransactionID"].astype(str)
)

df["card_node"] = "card_" + df["card_key"]
df["device_node"] = "device_" + df["DeviceInfo"].fillna("unknown").astype(str)

all_nodes = pd.Index(
    pd.concat([df["card_node"], df["device_node"]]).unique()
)
node_to_id = {node: idx for idx, node in enumerate(all_nodes)}

card_ids = df["card_node"].map(node_to_id).to_numpy()
device_ids = df["device_node"].map(node_to_id).to_numpy()

edge_index = torch.tensor(
    np.vstack([
        np.concatenate([card_ids, device_ids]),
        np.concatenate([device_ids, card_ids])
    ]),
    dtype=torch.long
)

# Aggregate transaction features at card level.
numeric_features = [
    "TransactionAmt",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_avg_amount_30d",
    "device_changed",
    "address_changed",
    "device_missing"
]

card_stats = df.groupby("card_node")[numeric_features].mean()
card_stats["transaction_count"] = df.groupby("card_node").size()

# Fill missing values using training-data medians.
card_stats = card_stats.replace([np.inf, -np.inf], np.nan)
card_stats = card_stats.fillna(card_stats.median()).fillna(0)

# Standardize card features.
means = card_stats.mean()
stds = card_stats.std().replace(0, 1)
card_stats = (card_stats - means) / stds

# Seven original features plus transaction count.
num_features = len(numeric_features) + 1
node_features = torch.zeros(
    (len(all_nodes), num_features + 1),
    dtype=torch.float32
)

# First feature: card node = 1, device node = 0.
node_features[:, 0] = torch.tensor([
    1.0 if str(node).startswith("card_") else 0.0
    for node in all_nodes
])

for card, row in card_stats.iterrows():
    node_id = node_to_id[card]
    node_features[node_id, 1:] = torch.tensor(
        row.to_numpy(dtype=np.float32)
    )

# Assign labels to card nodes only.
labels = torch.full((len(all_nodes),), -1, dtype=torch.long)
card_labels = df.groupby("card_node")["isFraud"].max()

for card, fraud_label in card_labels.items():
    labels[node_to_id[card]] = int(fraud_label)

data = Data(
    x=node_features,
    edge_index=edge_index,
    y=labels
)

output_path = OUTPUT_DIR / "fraud_graph_v2.pt"
torch.save(data, output_path)

print("\nImproved graph created!")
print("Transactions:", len(df))
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Feature dimensions:", data.num_features)
print("Card nodes:", int((labels >= 0).sum()))
print("Fraud-labelled card nodes:", int((labels == 1).sum()))
print("Saved:", output_path)
