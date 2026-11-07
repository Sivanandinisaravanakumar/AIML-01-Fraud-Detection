
from pathlib import Path
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
TEST_FILE = ROOT / "data" / "ieee_cis" / "processed" / "test_features.csv"

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading train and test data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# Compare only common model features.
# Exclude IDs and target labels.
exclude = {"TransactionID", "isFraud", "card_key"}

common_columns = [
    col for col in train.columns
    if col in test.columns and col not in exclude
]

if not common_columns:
    raise ValueError("No common feature columns found.")

reference_data = train[common_columns].copy()
current_data = test[common_columns].copy()

# Keep the report manageable.
reference_data = reference_data.sample(
    n=min(100000, len(reference_data)), random_state=42
)
current_data = current_data.sample(
    n=min(100000, len(current_data)), random_state=42
)

print("Reference rows:", len(reference_data))
print("Current rows:", len(current_data))
print("Features compared:", len(common_columns))

print("Generating Evidently drift report...")

report = Report(metrics=[DataDriftPreset()])

snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data,
)

report_file = OUTPUT_DIR / "ieee_data_drift_report.html"
snapshot.save_html(str(report_file))

snapshot.save_json(
    str(OUTPUT_DIR / "ieee_data_drift_snapshot.json")
)

print("\nDrift metric keys:")
print(snapshot.dict().keys())

print("\nDrift metric details:")
print(json.dumps(snapshot.dict(), indent=2, default=str)[:5000])

# Extract dataset drift summary
metrics_data = snapshot.dict()

drift_metric = next(
    (
        metric for metric in metrics_data["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    ),
    None,
)

if drift_metric is None:
    raise RuntimeError("Dataset drift summary metric not found.")

drifted_count = int(drift_metric["value"]["count"])
drift_share = float(drift_metric["value"]["share"])

# Retraining trigger configuration
DRIFT_THRESHOLD = 0.50
retraining_required = drift_share >= DRIFT_THRESHOLD

trigger_result = {
    "dataset": "IEEE-CIS",
    "total_features": len(common_columns),
    "drifted_features": drifted_count,
    "drift_share": drift_share,
    "drift_threshold": DRIFT_THRESHOLD,
    "retraining_required": retraining_required,
    "action": (
        "RETRAINING_REQUESTED"
        if retraining_required
        else "NO_ACTION"
    ),
}

trigger_file = OUTPUT_DIR / "retraining_trigger.json"

with open(trigger_file, "w", encoding="utf-8") as file:
    json.dump(trigger_result, file, indent=4)

print("\nRETRAINING TRIGGER RESULT")
print(json.dumps(trigger_result, indent=4))
print("Trigger file:", trigger_file)
