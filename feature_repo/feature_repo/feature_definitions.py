from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    ValueType,
)
from feast.types import Float64, Int64, String

project = Project(
    name="feature_repo",
    description="IEEE-CIS Financial Fraud Detection Feature Store",
)

card = Entity(
    name="card",
    join_keys=["card_key"],
    value_type=ValueType.STRING,
    description="Composite card identifier used as an entity proxy",
)

fraud_source = FileSource(
    name="fraud_transactions_source",
    path="data/fraud_features.parquet",
    timestamp_field="event_timestamp",
)

card_fraud_features = FeatureView(
    name="card_fraud_features",
    entities=[card],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="card_txn_count_1h", dtype=Float64),
        Field(name="card_txn_count_24h", dtype=Float64),
        Field(name="card_avg_amount_30d", dtype=Float64),
        Field(name="amount_vs_card_average", dtype=Float64),
        Field(name="device_changed", dtype=Int64),
        Field(name="address_changed", dtype=Int64),
        Field(name="device_missing", dtype=Int64),
    ],
    online=True,
    source=fraud_source,
    tags={"project": "fraud-detection", "dataset": "IEEE-CIS"},
)

fraud_detection_v1 = FeatureService(
    name="fraud_detection_v1",
    features=[card_fraud_features],
)
