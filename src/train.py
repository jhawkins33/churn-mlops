"""
Training script for churn prediction.

Trains two baseline models (Logistic Regression and Random Forest),
reports metrics, and saves the better one. Works two ways with no
code changes needed:

  1. Locally (what we've been doing):
     python train.py --input data/processed.csv --model-dir .

  2. Inside a SageMaker training job (script mode):
     SageMaker automatically sets --model-dir and the --train channel
     path via environment variables (SM_MODEL_DIR, SM_CHANNEL_TRAIN),
     so no arguments need to be passed manually.
"""

import argparse
import glob
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def resolve_input_csv(input_path: str) -> str:
    """
    Accept either a direct CSV path (local usage) or a directory
    (SageMaker channel usage, e.g. /opt/ml/input/data/train/) and
    find the CSV inside it.
    """
    if os.path.isdir(input_path):
        csvs = glob.glob(os.path.join(input_path, "*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in directory {input_path}")
        return csvs[0]
    return input_path


def load_data(path: str):
    csv_path = resolve_input_csv(path)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("confusion matrix:")
    print(confusion_matrix(y_test, preds))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline churn models.")
    # SageMaker sets SM_CHANNEL_TRAIN to the local path where it downloaded
    # the "train" channel data. Locally, default to data/processed.csv.
    parser.add_argument(
        "--input",
        default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed.csv"),
        help="Path to processed CSV, or a directory containing it (SageMaker channel)",
    )
    # SageMaker sets SM_MODEL_DIR to /opt/ml/model — anything saved there
    # is automatically packaged into model.tar.gz and uploaded to S3.
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "."),
        help="Directory to save the trained model into",
    )
    args = parser.parse_args()

    print(f"Loading processed data from {args.input}...")
    X, y = load_data(args.input)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features. Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Baseline 1: Logistic Regression (scaled — LR is sensitive to feature
    # magnitude, e.g. TotalCharges in the thousands vs 0/1 dummy columns.
    # Scaling both fixes the convergence warning and tends to improve AUC.)
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    logreg.fit(X_train, y_train)
    logreg_metrics = evaluate("Logistic Regression (scaled)", logreg, X_test, y_test)

    # Baseline 2: Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    # Pick the better model by ROC AUC
    if rf_metrics["roc_auc"] >= logreg_metrics["roc_auc"]:
        best_name, best_model = "Random Forest", rf
    else:
        best_name, best_model = "Logistic Regression", logreg

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    print(f"\nBest model: {best_name} (saved to {model_path})")
    joblib.dump(best_model, model_path)


if __name__ == "__main__":
    main()
