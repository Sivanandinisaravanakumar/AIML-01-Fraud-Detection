# AI-Based Credit Card Fraud Detection System

## Project Overview
An AI-based credit card fraud detection system that predicts whether a transaction is fraudulent or normal. The project uses XGBoost for prediction and Apache Kafka for real-time transaction streaming.

**Educational prototype only — not intended for real financial decisions.**

## Tech Stack
- Python
- Pandas, Scikit-learn
- XGBoost
- FastAPI
- Apache Kafka
- Docker
- MLflow

## Model Performance
| Metric | Score |
|---|---:|
| ROC-AUC | 0.9805 |
| PR-AUC | 0.8382 |
| Precision | 0.4398 |
| Recall | 0.8571 |

## Architecture
1. Transaction is submitted through the FastAPI endpoint.
2. XGBoost predicts the fraud probability.
3. The API publishes transaction details and prediction to Kafka.
4. The Kafka consumer receives the message and displays the prediction.

## Run the API with Docker

Build the image:

```bash
docker build -t fraud-detection-api .
```

Start Kafka using Docker Compose:

```bash
docker compose up -d kafka
```

Start the API:

```bash
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api
```

Open Swagger documentation:

```text
http://localhost:8000/docs
```

## Run the Kafka Consumer

From the project root, activate the Python environment and run:

```bash
python -m src.kafka_consumer
```

The consumer listens to the `fraud-transactions` topic using `localhost:9092`.

## API Endpoint

`POST /predict`

Accepts transaction features (`Time`, `V1`–`V28`, `Amount`) and returns:
- Transaction ID
- Fraud probability
- Prediction
- Threshold
- Kafka message status

## Dataset
Credit Card Fraud Detection dataset from the ULB Machine Learning Group, containing anonymized transaction features.

## Disclaimer
This project is developed for educational and demonstration purposes. It is not a certified fraud detection solution.
