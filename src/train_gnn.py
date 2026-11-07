import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "ieee_cis" / "graph" / "fraud_graph.pt"
MODEL_DIR = ROOT / "models" / "ieee_cis"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
device = torch.device("cpu")

print("Loading fraud graph...")
data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)

# Only card nodes have labels. Device nodes are excluded.
labeled_idx = torch.where(data.y >= 0)[0].numpy()
labels = data.y[labeled_idx].numpy()

train_idx, val_idx = train_test_split(
    labeled_idx,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True

data = data.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)

class FraudGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.output(x, edge_index=None) if False else self.output(x)

model = FraudGCN(data.num_features).to(device)

train_labels = data.y[train_mask]
class_counts = torch.bincount(train_labels, minlength=2).float()
class_weights = class_counts.sum() / (2 * class_counts)
class_weights = class_weights.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training GCN...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        logits[train_mask],
        data.y[train_mask],
        weight=class_weights
    )
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(
                val_logits[val_mask],
                data.y[val_mask]
            ).item()
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Validation Loss: {val_loss:.4f}")

model.eval()
with torch.no_grad():
    probabilities = torch.softmax(
        model(data.x, data.edge_index)[val_mask], dim=1
    )[:, 1].cpu().numpy()

y_true = data.y[val_mask].cpu().numpy()
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "average_precision": float(average_precision_score(y_true, probabilities)),
    "roc_auc": float(roc_auc_score(y_true, probabilities)),
    "precision": float(precision_score(y_true, predictions, zero_division=0)),
    "recall": float(recall_score(y_true, predictions, zero_division=0)),
    "f1": float(f1_score(y_true, predictions, zero_division=0)),
    "validation_card_nodes": int(len(y_true)),
    "training_card_nodes": int(len(train_idx))
}

torch.save(model.state_dict(), MODEL_DIR / "fraud_gcn.pt")
with open(MODEL_DIR / "gnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nGCN training complete!")
print(json.dumps(metrics, indent=2))
print("Model saved:", MODEL_DIR / "fraud_gcn.pt")
