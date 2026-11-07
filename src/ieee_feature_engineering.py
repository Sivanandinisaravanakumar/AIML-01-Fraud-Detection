
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ieee_cis"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print("Loading selected IEEE-CIS columns...")

    transaction_columns = [
        "TransactionID", "isFraud", "TransactionDT",
        "TransactionAmt", "card1", "card2", "card3",
        "card5", "card6", "addr1", "addr2"
    ]

    identity_columns = [
        "TransactionID", "DeviceInfo", "DeviceType"
    ]

    train = pd.read_csv(
        DATA_DIR / "train_transaction.csv",
        usecols=transaction_columns
    )

    test = pd.read_csv(
        DATA_DIR / "test_transaction.csv",
        usecols=lambda col: col in transaction_columns
        and col != "isFraud"
    )

    train["dataset_type"] = "train"
    test["dataset_type"] = "test"

    identity_train = pd.read_csv(
        DATA_DIR / "train_identity.csv",
        usecols=identity_columns
    )

    identity_test = pd.read_csv(
        DATA_DIR / "test_identity.csv",
        usecols=identity_columns
    )

    identity = pd.concat(
        [identity_train, identity_test],
        ignore_index=True
    )

    data = pd.concat(
        [train, test],
        ignore_index=True
    )

    data = data.merge(
        identity,
        on="TransactionID",
        how="left"
    )

    print("Total records:", len(data))

    return data


def create_features(data):
    print("Creating transaction features...")

    data = data.copy()

    # Card-based identifier (proxy, not a real customer ID)
    card_columns = [
        "card1", "card2", "card3", "card5", "card6"
    ]

    data["card_key"] = (
        data[card_columns]
        .astype("string")
        .fillna("missing")
        .agg("_".join, axis=1)
    )

    # Prevent unrelated transactions with missing card1
    # from being grouped together.
    missing_card = data["card1"].isna()

    data.loc[missing_card, "card_key"] = (
        "missing_"
        + data.loc[missing_card, "TransactionID"].astype(str)
    )

    # Time features
    data["transaction_hour"] = (
        data["TransactionDT"] // 3600
    ) % 24

    data["transaction_day"] = (
        data["TransactionDT"] // 86400
    )

    data["transaction_time"] = pd.to_datetime(
        data["TransactionDT"],
        unit="s",
        origin="unix"
    )

    # Sort chronologically
    data = data.sort_values(
        ["card_key", "TransactionDT"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Historical transaction counts
    # Current transaction excluded using closed="left"
    print("Calculating 1-hour transaction velocity...")

    count_1h = (
        data.groupby("card_key", sort=False)
        .rolling(
            "1h",
            on="transaction_time",
            closed="left"
        )["TransactionID"]
        .count()
    )

    data["card_txn_count_1h"] = (
        count_1h.fillna(0).to_numpy()
    )

    print("Calculating 24-hour transaction velocity...")

    count_24h = (
        data.groupby("card_key", sort=False)
        .rolling(
            "24h",
            on="transaction_time",
            closed="left"
        )["TransactionID"]
        .count()
    )

    data["card_txn_count_24h"] = (
        count_24h.fillna(0).to_numpy()
    )

    # Historical average amount over previous 30 days
    print("Calculating historical average amount...")

    avg_amount = (
        data.groupby("card_key", sort=False)
        .rolling(
            "30D",
            on="transaction_time",
            closed="left"
        )["TransactionAmt"]
        .mean()
    )

    data["card_avg_amount_30d"] = (
        avg_amount.to_numpy()
    )

    data["amount_vs_card_average"] = (
        data["TransactionAmt"]
        / data["card_avg_amount_30d"].replace(
            0, float("nan")
        )
    )

    # Device and address change indicators
    print("Creating device and address change features...")

    previous_device = (
        data.groupby("card_key")["DeviceInfo"].shift(1)
    )

    previous_address = (
        data.groupby("card_key")["addr1"].shift(1)
    )

    data["device_changed"] = (
        data["DeviceInfo"].notna()
        & previous_device.notna()
        & (data["DeviceInfo"] != previous_device)
    ).astype(int)

    data["address_changed"] = (
        data["addr1"].notna()
        & previous_address.notna()
        & (data["addr1"] != previous_address)
    ).astype(int)

    data["device_missing"] = (
        data["DeviceInfo"].isna().astype(int)
    )

    # Remove helper columns
    data = data.drop(
        columns=["card_key", "transaction_time"]
    )

    return data


def main():
    data = load_data()

    data = create_features(data)

    # Separate train and test datasets
    train_features = data[
        data["dataset_type"] == "train"
    ].drop(columns=["dataset_type"])

    test_features = data[
        data["dataset_type"] == "test"
    ].drop(columns=["dataset_type"])

    train_path = OUTPUT_DIR / "train_features.csv"
    test_path = OUTPUT_DIR / "test_features.csv"

    print("Saving processed features...")

    train_features.to_csv(
        train_path,
        index=False
    )

    test_features.to_csv(
        test_path,
        index=False
    )

    print("\nFeature engineering completed!")

    print("Train records:", len(train_features))
    print("Test records:", len(test_features))

    print("Train features:", train_path)
    print("Test features:", test_path)

    # Validate important generated features
    feature_columns = [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_avg_amount_30d",
        "amount_vs_card_average",
        "device_changed",
        "address_changed",
        "device_missing"
    ]

    print("\nFeature validation:")
    print(
        train_features[feature_columns]
        .describe()
        .round(2)
    )

    print("\nMissing values in generated features:")
    print(
        train_features[feature_columns]
        .isna()
        .sum()
    )


if __name__ == "__main__":
    main()
