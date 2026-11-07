import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "fraud-transactions",
    bootstrap_servers=["localhost:9092"],
    group_id="ieee-cis-fraud-consumer-test",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda value: json.loads(value.decode("utf-8"))
)

print("IEEE-CIS Kafka Consumer Started...")
print("Waiting for IEEE-CIS transactions...\n")

try:
    for message in consumer:
        transaction = message.value

        print("--------------------------------")
        print("IEEE-CIS TRANSACTION RECEIVED")
        print("Transaction ID:", transaction.get("transaction_id"))
        print("Fraud Probability:", transaction.get("fraud_probability"))
        print("Threshold:", transaction.get("threshold"))
        print("Prediction:", transaction.get("prediction"))
        print("Label:", transaction.get("label"))
        print("Features:", json.dumps(
            transaction.get("features", {}),
            indent=2
        ))
        print("--------------------------------\n")

except KeyboardInterrupt:
    print("\nIEEE-CIS Consumer stopped.")

finally:
    consumer.close()

