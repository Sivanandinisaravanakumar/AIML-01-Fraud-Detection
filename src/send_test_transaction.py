import json
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

transaction = {
    "transaction_id": 1002,
    "Time": 1000,
    "Amount": 2500.50
}

# Add all 28 PCA features
for i in range(1, 29):
    transaction[f"V{i}"] = 0.0

producer.send("fraud-transactions", value=transaction)
producer.flush()
producer.close()

print("Complete test transaction sent successfully!")