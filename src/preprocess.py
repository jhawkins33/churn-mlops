"""
Preprocessing script for the Telco Customer Churn dataset.

Reads raw data (from local disk or S3), cleans and encodes it into
model-ready features, and writes the result back out (local disk or S3).

Usage (local):
    python preprocess.py --input data/raw.csv --output data/processed.csv

Usage (S3, once boto3 credentials are configured):
    python preprocess.py \
        --input s3://churn-mlops-raw-dev/WA_Fn-UseC_-Telco-Customer-Churn.csv \
        --output s3://churn-mlops-processed-dev/processed.csv
"""

import argparse
import pandas as pd


# Columns that are Yes/No and can be mapped straight to 0/1
BINARY_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

# Multi-category columns that need one-hot encoding
CATEGORICAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from local path or S3 URI."""
    return pd.read_csv(path)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is stored as a string and has 11 blank values for
    customers with tenure == 0 (brand new, not yet billed). Coerce to
    numeric and fill those with 0, which is the correct value here —
    not an unknown/missing value.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No columns to 1/0, and gender to 1/0."""
    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-category columns."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # pd.get_dummies produces bool dtype columns; cast to int (0/1) since
    # some downstream tools (e.g. SageMaker built-in algorithms) expect
    # numeric rather than boolean features.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # customerID is a unique identifier, not a predictive feature
    df = df.drop(columns=["customerID"])

    df = clean_total_charges(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Write CSV to local path or S3 URI."""
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Telco churn data.")
    parser.add_argument("--input", required=True, help="Path or S3 URI to raw CSV")
    parser.add_argument("--output", required=True, help="Path or S3 URI to write processed CSV")
    args = parser.parse_args()

    print(f"Loading raw data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    print("Preprocessing...")
    processed = preprocess(df)
    print(f"Processed shape: {processed.shape[0]} rows, {processed.shape[1]} columns.")

    print(f"Saving processed data to {args.output}...")
    save_data(processed, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
