"""
Computes a statistical baseline from the training data — per-feature
bin edges and bin proportions — used later to detect drift in new data
via Population Stability Index (PSI).

Usage:
    python src/compute_baseline.py --input data/processed.csv --output data/baseline.json
"""

import argparse
import json
import numpy as np
import pandas as pd

N_BINS = 10


def compute_feature_baseline(series: pd.Series):
    """
    Bin a numeric feature into N_BINS quantile-based buckets and record
    the bin edges plus the proportion of training data in each bin.
    Quantile-based binning (rather than equal-width) keeps bins
    meaningfully populated even for skewed features like MonthlyCharges.
    """
    # Binary/near-binary features (0/1 dummies) don't need quantile
    # binning — just use their two natural values directly.
    unique_vals = series.nunique()
    if unique_vals <= 2:
        edges = [-np.inf, 0.5, np.inf]
    else:
        edges = np.quantile(series, np.linspace(0, 1, N_BINS + 1))
        edges = np.unique(edges)  # collapse duplicate edges from ties
        edges[0] = -np.inf
        edges[-1] = np.inf

    binned = pd.cut(series, bins=edges, include_lowest=True)
    proportions = binned.value_counts(normalize=True, sort=False)

    return {
        "edges": [float(e) for e in edges],
        "proportions": [float(p) for p in proportions.values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed training CSV")
    parser.add_argument("--output", required=True, help="Where to save the baseline JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    features = df.drop(columns=["Churn"])

    print(f"Computing baseline for {features.shape[1]} features from {features.shape[0]} rows...")

    baseline = {}
    for col in features.columns:
        baseline[col] = compute_feature_baseline(features[col])

    with open(args.output, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to {args.output}")


if __name__ == "__main__":
    main()