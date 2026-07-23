"""
Creates a synthetically drifted copy of the processed data to validate
that detect_drift.py actually catches real distribution shifts, not
just reports "stable" by default.

Simulates a scenario: a marketing campaign brings in a wave of new,
short-tenure customers on month-to-month-style contracts — the classic
high-churn-risk profile becoming more common in the customer base.

Usage:
    python src/simulate_drift.py --input data/processed.csv --output data/drifted_data.csv
"""

import argparse
import numpy as np
import pandas as pd

np.random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Shift tenure sharply lower — simulates an influx of brand-new customers
    df["tenure"] = (df["tenure"] * 0.25).round().clip(lower=0)

    # Shift MonthlyCharges up — simulates a pricing change or upsell push
    df["MonthlyCharges"] = df["MonthlyCharges"] * 1.3

    # Recompute TotalCharges to stay internally consistent with the new tenure/charges
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)

    # Shift contract mix — simulate more customers on two-year contracts
    # (flip a random 20% of non-two-year customers to two-year)
    flip_mask = (df["Contract_Two year"] == 0) & (np.random.rand(len(df)) < 0.2)
    df.loc[flip_mask, "Contract_Two year"] = 1
    df.loc[flip_mask, "Contract_One year"] = 0

    df.to_csv(args.output, index=False)
    print(f"Drifted data saved to {args.output}")
    print(f"tenure mean: {df['tenure'].mean():.1f} (was likely ~32 in original)")
    print(f"MonthlyCharges mean: {df['MonthlyCharges'].mean():.1f}")


if __name__ == "__main__":
    main()