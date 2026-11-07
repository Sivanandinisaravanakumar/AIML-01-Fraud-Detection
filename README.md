# AI-Based Credit Card Fraud Detection System

An end-to-end machine learning capstone project for credit card fraud detection using traditional ML, real-time streaming, feature stores, model monitoring, and containerized deployment.

> **Educational prototype only.** This project is not intended for real financial decisions or production banking use.

## 1. Project Overview

This project explores two fraud detection datasets and two complementary machine learning approaches:

- **ULB Credit Card Fraud Dataset:** An anonymized dataset with `Time`, `Amount`, and PCA-transformed transaction features (`V1`–`V28`).
- **IEEE-CIS Fraud Detection Dataset:** A larger transaction and identity dataset used for feature engineering, XGBoost training, and additional graph-based experiments.

The system supports transaction scoring through FastAPI and publishes prediction events to Apache Kafka.

## 2. Architecture

```text
                 Transaction Request
                         |
                         v
                    FastAPI API
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Feature Store            ML Model
          (Feast)               (XGBoost)
             |                       |
             +-----------+-----------+
                         |
                         v
                 Fraud Prediction
                         |
                         v
                  Kafka Producer
                         |
                         v
               fraud-transactions
                         |
                         v
                  Kafka Consumer
                         |
                         v
                Transaction Alerts
```

Additional project components:

- Streamlit dashboard for analyst case review and monitoring.
- Evidently AI for data drift analysis.
- Graph Neural Network experiments for transaction relationship analysis.
- Docker and Kubernetes deployment for the IEEE-CIS API.

## 3. Tech Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Graph learning | PyTorch, PyTorch Geometric |
| API | FastAPI, Uvicorn |
| Feature store | Feast |
| Streaming | Apache Kafka |
| Dashboard | Streamlit |
| Monitoring | Evidently AI |
| Experiment tracking | MLflow |
| Containerization | Docker |
| Orchestration | Kubernetes, Minikube |

## 4. Model Performance

### Original ULB model

The following metrics are reported by the original ULB project README:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9805 |
| PR-AUC | 0.8382 |
| Precision | 0.4398 |
| Recall | 0.8571 |

### IEEE-CIS XGBoost model

Evaluation on the project's validation split:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8810 |
| PR-AUC | 0.3507 |
| Precision | 0.3161 |
| Recall | 0.4549 |
| F1-score | 0.3730 |

The selected validation threshold is approximately **0.7657**. Threshold selection affects the precision-recall trade-off.

### Graph Neural Network experiment

The second GNN experiment achieved:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8503 |
| Average Precision | 0.5749 |
| Precision | 0.3514 |
| Recall | 0.6575 |
| F1-score | 0.4580 |

### Hybrid experiment

The XGBoost and GNN scores were combined experimentally. The hybrid did not outperform the XGBoost validation result in this experiment. The GNN is therefore treated as an additional research signal, not as a proven improvement.

> Metrics are dataset- and split-specific. They should not be compared as though they came from the same evaluation dataset or protocol.

## 5. Project Structure

```text
AIML-01-Fraud-Detection/
│
├── src/
│   ├── api.py
│   ├── ieee_api.py
│   ├── ieee_feature_engineering.py
│   ├── train_ieee_model.py
│   ├── ieee_kafka_consumer.py
│   ├── feast_api.py
│   ├── evaluate_hybrid.py
│   ├── drift_monitor.py
│   └── dashboard.py
│
├── models/
│   └── ieee_cis/
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── feature_definitions.py
│       └── data/
│
├── data/
│   └── ieee_cis/
│
├── processed/
├── reports/
├── k8s/
│   └── ieee-deployment.yaml
│
├── Dockerfile
├── Dockerfile.ieee
├── Dockerfile.ieee.dockerignore
├── requirements-ieee.txt
└── README.md
```

Large source datasets and local runtime files are excluded from the repository where appropriate. Feast runtime files are included in the deployment build because the API uses the local feature store.

## 6. Dataset

### ULB Credit Card Fraud Dataset

Contains anonymized credit card transactions with:
- `Time`
- `Amount`
- `V1` to `V28`
- Fraud label

### IEEE-CIS Fraud Detection Dataset

The IEEE-CIS dataset contains transaction and identity information. Feature engineering includes:
- Transaction hour and day
- Transaction velocity features
- Previous transaction counts over time windows
- Average transaction amount features
- Device and address change indicators
- Device missing-value indicators

Dataset source: [IEEE-CIS Fraud Detection – Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

Download the dataset through Kaggle and place the extracted files in the expected local data directory before running the feature engineering pipeline.

## 7. API Usage

### Original ULB API

The original API uses:

```text
POST /predict
```

Run the original API using Docker:

```powershell
docker build -t fraud-detection-api .
docker compose up -d kafka
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### IEEE-CIS API

The IEEE-CIS API uses FastAPI, the XGBoost model, and Feast online features.

The API returns fields including:
- Transaction ID
- Fraud probability
- Decision threshold
- Prediction and label
- Model name
- Feature source
- Kafka message status

The API is exposed on port `8000` inside the container.

## 8. Kafka Streaming

The application publishes transaction predictions to the Kafka topic:

```text
fraud-transactions
```

Kafka listener configuration:

| Client | Bootstrap server |
|---|---|
| Windows host consumer | `localhost:9092` |
| Docker/Kubernetes API | `host.docker.internal:9093` |

Start Kafka:

```powershell
docker compose up -d kafka
```

Run the original ULB consumer:

```powershell
python -m src.kafka_consumer
```

Run the IEEE-CIS consumer:

```powershell
python -m src.ieee_kafka_consumer
```

The consumer receives transaction messages and displays prediction details.

## 9. Feast Feature Store

Feast is used to retrieve online transaction features for the IEEE-CIS prediction API.

The local feature store configuration is located at:

```text
feature_repo/feature_repo/feature_store.yaml
```

The feature repository contains feature definitions and local registry and online-store data.

The online feature view excludes the target label to avoid label leakage during inference.

## 10. Streamlit Dashboard

The dashboard provides:
- Model and prediction overview
- Historical analyst case queue
- Case status and feedback updates
- Data drift monitoring
- Kafka live transaction alerts

Run the dashboard from the project environment:

```powershell
streamlit run src/dashboard.py
```

## 11. Data Drift Monitoring

Evidently AI is used to compare training and test feature distributions.

The monitoring report and associated JSON outputs are stored under:

```text
reports/
```

The current drift analysis identified drift in a subset of monitored features. The configured aggregate drift threshold did not trigger an automatic retraining action.

The retraining trigger is an indicator for review, not a fully automated retraining pipeline.

## 12. Docker Deployment

Build the IEEE-CIS API image:

```powershell
docker build -f Dockerfile.ieee -t fraud-detection-ieee-api:latest .
```

The image contains:
- IEEE-CIS API code
- XGBoost model artifacts
- Feast feature definitions and local store data
- Python runtime dependencies

## 13. Kubernetes Deployment with Minikube

Load the image into Minikube if it is not already available:

```powershell
minikube image load fraud-detection-ieee-api:latest
```

Apply the Kubernetes manifest:

```powershell
kubectl apply -f k8s/ieee-deployment.yaml
```

Check deployment status:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
kubectl rollout status deployment/ieee-fraud-api
```

Open the service URL:

```powershell
minikube service ieee-fraud-api --url
```

Keep the terminal running while using the returned service URL. Swagger documentation is available at:

```text
<service-url>/docs
```

The Kubernetes service uses NodePort `30082` and targets container port `8000`.

## 14. Important Limitations

- This is an educational prototype, not a production banking system.
- Dataset anonymization limits the interpretation of some features.
- Validation metrics do not guarantee performance on future transactions.
- Fraud thresholds must be selected according to operational requirements.
- The GNN and hybrid results are experimental.
- Drift detection does not automatically retrain or redeploy the model.
- The local Feast SQLite configuration is intended for this project setup, not as a production multi-instance feature-store architecture.
- Kafka events are currently published to a shared topic by the two project APIs; consumers should identify the dataset/model fields in each message.

## 15. Future Improvements

- Evaluate models on a strictly independent test set.
- Improve temporal validation and leakage prevention.
- Explore model calibration and cost-sensitive threshold selection.
- Improve graph construction and GNN evaluation.
- Add automated model retraining with approval controls.
- Add authentication, monitoring, and production-grade feature-store infrastructure.
- Separate Kafka topics or consumer groups by dataset and application.

## 16. Disclaimer

This project was developed for educational and capstone demonstration purposes. Predictions must not be used as the sole basis for approving, declining, or investigating real financial transactions.