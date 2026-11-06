from pathlib import Path

import pandas as pd

DATA_PATH=Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"

df=pd.read_csv(DATA_PATH)

print("\n--- FIRST 5 TRANSACTIONS ---")
print(df.head())
print("\n--- DATASET SIZE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())
print("\n--- FRAUD DISTRIBUTION ---")
print(df["Class"].value_counts())
print("\n--- CLASS PERCENTAGE ---")
print((df["Class"].value_counts(normalize=True) * 100).round(4))
print("\n--- TRANSACTION AMOUNT SUMMARY ---")
print(df["Amount"].describe())
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())
