from pathlib import Path
import json
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier
from kafka import KafkaProducer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_xgboost.json"

app = FastAPI(
    title="Real-Time Financial Fraud Detection API",
    version="1.0.0",
    description="Educational fraud detection prototype"
)

# Load trained model
model = XGBClassifier()
model.load_model(str(MODEL_PATH))

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers=["host.docker.internal:9093"],
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


class Transaction(BaseModel):
    Time: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs"
    }


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        transaction_id = str(uuid.uuid4())

        features = pd.DataFrame([transaction.model_dump()])

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        prediction = (
            "FRAUD_ALERT" if probability >= 0.5
            else "NORMAL"
        )

        # Send transaction and prediction to Kafka
        kafka_message = {
            "transaction_id": transaction_id,
            "features": transaction.model_dump(),
            "fraud_probability": round(probability, 4),
            "prediction": prediction
        }

        producer.send(
            "fraud-transactions",
            value=kafka_message
        )
        producer.flush()

        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(probability, 4),
            "prediction": prediction,
            "threshold": 0.5,
            "kafka_status": "MESSAGE_SENT"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )