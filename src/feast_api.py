from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
FEAST_REPO = ROOT / "feature_repo" / "feature_repo"

store = FeatureStore(repo_path=str(FEAST_REPO))

app = FastAPI(
    title="Fraud Detection Feast Feature API",
    version="1.0.0",
    description="IEEE-CIS online feature retrieval"
)

class CardRequest(BaseModel):
    card_key: str

@app.get("/")
def home():
    return {"message": "Feast Feature API is running"}

@app.post("/features")
def get_features(request: CardRequest):
    try:
        result = store.get_online_features(
            features=[
                "card_fraud_features:card_txn_count_1h",
                "card_fraud_features:card_txn_count_24h",
                "card_fraud_features:card_avg_amount_30d",
                "card_fraud_features:amount_vs_card_average",
                "card_fraud_features:device_changed",
                "card_fraud_features:address_changed",
                "card_fraud_features:device_missing",
            ],
            entity_rows=[{"card_key": request.card_key}],
        )

        return result.to_dict()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
