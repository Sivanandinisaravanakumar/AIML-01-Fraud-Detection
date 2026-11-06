import json
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_xgboost.json"

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

consumer = KafkaConsumer(
    "fraud-transactions",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)

print("Kafka Fraud Detection Consumer Started...")
print("Waiting for API transactions...\n")

try:
    for message in consumer:
        transaction = message.value

        # Support both old flat messages and new API messages
        features = transaction.get("features", transaction)

        missing = [f for f in FEATURES if f not in features]

        if missing:
            print("Invalid transaction. Missing:", missing)
            continue

        input_data = pd.DataFrame(
            [[features[f] for f in FEATURES]],
            columns=FEATURES
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        prediction = (
            "FRAUD" if probability >= 0.5 else "NORMAL"
        )

        print("--------------------------------")
        print("Transaction ID:", transaction.get("transaction_id"))
        print("Fraud Probability:", round(probability, 4))
        print("Prediction:", prediction)

except KeyboardInterrupt:
    print("\nConsumer stopped.")

finally:
    consumer.close()