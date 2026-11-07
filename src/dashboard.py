from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from kafka import KafkaConsumer

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "ieee_cis" / "processed"
MODEL_DIR = ROOT / "models" / "ieee_cis"
REPORTS_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_features.csv"
CASE_FILE = REPORTS_DIR / "analyst_cases.csv"
DRIFT_FILE = REPORTS_DIR / "retraining_trigger.json"
METRICS_FILE = MODEL_DIR / "metrics.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Financial Fraud Detection")
st.caption("IEEE-CIS | Fraud Analyst Monitoring Dashboard")

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def create_case_file():
    if not TRAIN_FILE.exists():
        st.error(f"Training feature file not found: {TRAIN_FILE}")
        st.stop()

    required_columns = [
        "TransactionID",
        "TransactionAmt",
        "isFraud",
    ]

    df = pd.read_csv(
        TRAIN_FILE,
        usecols=required_columns,
    )

    fraud_cases = df[df["isFraud"] == 1].head(100).copy()

    fraud_cases["case_status"] = "Open"
    fraud_cases["analyst_feedback"] = "Pending"

    fraud_cases.to_csv(CASE_FILE, index=False)

    return fraud_cases


def load_cases():
    if not CASE_FILE.exists():
        return create_case_file()

    cases = pd.read_csv(CASE_FILE)

    if "case_status" not in cases.columns:
        cases["case_status"] = "Open"

    if "analyst_feedback" not in cases.columns:
        cases["analyst_feedback"] = "Pending"

    cases["case_status"] = cases["case_status"].fillna("Open")
    cases["analyst_feedback"] = cases["analyst_feedback"].fillna("Pending")

    return cases


def save_cases(edited_cases):
    temp_file = CASE_FILE.with_suffix(".tmp")
    edited_cases.to_csv(temp_file, index=False)
    temp_file.replace(CASE_FILE)


# --------------------------------------------------
# Kafka Consumer
# --------------------------------------------------

@st.cache_resource
def get_dashboard_consumer():
    return KafkaConsumer(
        "fraud-transactions",
        bootstrap_servers=["localhost:9092"],
        group_id="streamlit-live-alerts",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
        request_timeout_ms=10000,
    )


def read_live_alerts():
    consumer = get_dashboard_consumer()

    if "live_alerts" not in st.session_state:
        st.session_state.live_alerts = []

    messages = consumer.poll(
        timeout_ms=500,
        max_records=50,
    )

    for records in messages.values():
        for message in records:
            transaction = message.value

            st.session_state.live_alerts.append({
                "Received At": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Transaction ID": transaction.get("transaction_id"),
                "Fraud Probability": transaction.get(
                    "fraud_probability"
                ),
                "Threshold": transaction.get("threshold"),
                "Prediction": transaction.get("prediction"),
                "Label": transaction.get("label"),
            })

    st.session_state.live_alerts = (
        st.session_state.live_alerts[-500:]
    )


# --------------------------------------------------
# Load monitoring information
# --------------------------------------------------

drift = load_json(DRIFT_FILE)
model_metrics = load_json(METRICS_FILE)
cases = load_cases()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Dashboard Menu")
st.sidebar.write("Project: AIML-01")
st.sidebar.write("Dataset: IEEE-CIS")

page = st.sidebar.radio(
    "Select View",
    [
        "Overview",
        "Analyst Case Queue",
        "Data Drift",
        "Live Kafka Alerts",
    ],
)

# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":

    st.header("Monitoring Overview")

    drift_count = drift.get("drifted_features", 0)
    total_features = drift.get("total_features", 0)
    drift_share = drift.get("drift_share", 0)

    retraining_required = drift.get(
        "retraining_required",
        False,
    )

    open_count = int(
        (cases["case_status"] == "Open").sum()
    )

    closed_count = int(
        (cases["case_status"] == "Closed").sum()
    )

    pr_auc = model_metrics.get("PR_AUC")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Drifted Features",
        f"{drift_count}/{total_features}",
    )

    col2.metric(
        "Drift Share",
        f"{drift_share * 100:.1f}%",
    )

    col3.metric(
        "Open Cases",
        open_count,
    )

    col4.metric(
        "XGBoost PR-AUC",
        f"{pr_auc:.4f}" if isinstance(pr_auc, (int, float)) else "N/A",
    )

    st.divider()

    st.subheader("System Status")

    if retraining_required:
        st.error(
            "Drift threshold reached. "
            "Model retraining review is required."
        )
    else:
        st.success(
            "Drift is below the configured threshold. "
            "No retraining request."
        )

    st.info(
        f"Closed cases: {closed_count} | "
        f"Total analyst cases: {len(cases)}"
    )

    st.warning(
        "The current case queue contains historical labelled "
        "IEEE-CIS fraud examples. It is not a live Kafka alert feed."
    )


# --------------------------------------------------
# Analyst Case Queue
# --------------------------------------------------

elif page == "Analyst Case Queue":

    st.header("Analyst Case Queue")

    st.write(
        "Review historical fraud examples and update their case status."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cases", len(cases))

    col2.metric(
        "Open Cases",
        int((cases["case_status"] == "Open").sum()),
    )

    col3.metric(
        "Closed Cases",
        int((cases["case_status"] == "Closed").sum()),
    )

    st.divider()

    with st.form("analyst_case_form"):

        edited_cases = st.data_editor(
            cases,
            hide_index=True,
            use_container_width=True,
            key="analyst_cases_editor",
            disabled=[
                "TransactionID",
                "TransactionAmt",
                "isFraud",
            ],
            column_config={
                "TransactionID": st.column_config.NumberColumn(
                    "Transaction ID",
                ),
                "TransactionAmt": st.column_config.NumberColumn(
                    "Transaction Amount",
                    format="%.2f",
                ),
                "isFraud": st.column_config.NumberColumn(
                    "Actual Fraud Label",
                ),
                "case_status": st.column_config.SelectboxColumn(
                    "Case Status",
                    options=[
                        "Open",
                        "Investigating",
                        "Closed",
                    ],
                    required=True,
                ),
                "analyst_feedback": st.column_config.SelectboxColumn(
                    "Analyst Feedback",
                    options=[
                        "Pending",
                        "Confirmed Fraud",
                        "False Positive",
                        "Needs Review",
                    ],
                    required=True,
                ),
            },
        )

        save_clicked = st.form_submit_button(
            "Save Analyst Updates",
            type="primary",
            use_container_width=True,
        )

    if save_clicked:
        try:
            save_cases(edited_cases)

            st.success(
                "Analyst updates saved successfully!"
            )

            st.rerun()

        except (OSError, PermissionError, ValueError) as error:
            st.error(
                f"Could not save analyst updates: {error}"
            )

    st.caption(
        "Updates are saved locally in reports/analyst_cases.csv."
    )


# --------------------------------------------------
# Data Drift
# --------------------------------------------------

elif page == "Data Drift":

    st.header("Evidently Data Drift Monitoring")

    if not drift:
        st.warning(
            "Drift report data not found. "
            "Run src/drift_monitor.py first."
        )
    else:
        drift_count = drift.get("drifted_features", 0)
        total_features = drift.get("total_features", 0)
        drift_share = drift.get("drift_share", 0)
        threshold = drift.get("drift_threshold", 0.5)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Features", total_features)
        col2.metric("Drifted Features", drift_count)

        col3.metric(
            "Drift Share",
            f"{drift_share * 100:.1f}%",
        )

        st.progress(
            min(max(drift_share, 0.0), 1.0),
            text="Share of drifted features",
        )

        st.write(
            f"Configured retraining threshold: "
            f"{threshold * 100:.1f}%"
        )

        if drift.get("retraining_required", False):
            st.error(
                "Retraining review required: drift threshold reached."
            )
        else:
            st.success(
                "No retraining request: drift is below threshold."
            )

        st.json(drift)

    report_file = REPORTS_DIR / "ieee_data_drift_report.html"

    if report_file.exists():
        st.caption(
            "Detailed Evidently HTML report: "
            "reports/ieee_data_drift_report.html"
        )


# --------------------------------------------------
# Live Kafka Alerts
# --------------------------------------------------

elif page == "Live Kafka Alerts":

    st.header("Live Kafka Fraud Alerts")

    st.caption(
        "Real-time transactions from Kafka topic: "
        "fraud-transactions"
    )

    @st.fragment(run_every="3s")
    def display_live_alerts():

        try:
            read_live_alerts()

            alerts = st.session_state.live_alerts

            fraud_count = sum(
                1 for alert in alerts
                if alert["Prediction"] == 1
            )

            normal_count = len(alerts) - fraud_count

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Alerts", len(alerts))
            col2.metric("Fraud Alerts", fraud_count)
            col3.metric("Normal Transactions", normal_count)

            st.divider()

            if alerts:
                alert_df = pd.DataFrame(alerts)

                st.dataframe(
                    alert_df.iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Waiting for new Kafka transactions..."
                )

        except Exception as error:
            st.error(
                f"Kafka connection or reading error: {error}"
            )

    display_live_alerts()


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AIML-01 Financial Fraud Detection | "
    "Streamlit Analyst Dashboard"
)

