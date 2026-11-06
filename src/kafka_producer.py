import json
from kafka import KafkaProducer

# Connect to Kafka running in Docker
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

def send_transaction(transaction):
    producer.send("fraud-transactions", value=transaction)
    producer.flush()
    print("Transaction sent:", transaction)


if __name__ == "__main__":
    sample_transaction = {
        "transaction_id": 1001,
        "amount": 2500.50,
        "merchant": "Online Store",
        "is_fraud": 0
    }

    send_transaction(sample_transaction)
    producer.close()