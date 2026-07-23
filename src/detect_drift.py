"""
Detects feature drift in new data by comparing it against the training
baseline using Population Stability Index (PSI).

PSI interpretation (industry-standard thresholds):
    < 0.1  : no significant drift
    0.1-0.2: moderate drift, worth investigating
    > 0.2  : significant drift, model may need retraining

Usage:
    python src/detect_drift.py --input data/new_data.csv --baseline data/baseline.json
"""

import argparse
import json
import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.2
EPSILON = 1e-6  # avoids divide-by-zero / log(0) for empty bins


def compute_psi(new_series: pd.Series, feature_baseline: dict) -> float:
    edges = feature_baseline["edges"]
    baseline_proportions = np.array(feature_baseline["proportions"])

    binned = pd.cut(new_series, bins=edges, include_lowest=True)
    new_proportions = binned.value_counts(normalize=True, sort=False).values

    # Guard against bin count mismatch (shouldn't happen, but fail loudly if it does)
    if len(new_proportions) != len(baseline_proportions):
        raise ValueError(
            f"Bin count mismatch: baseline has {len(baseline_proportions)}, "
            f"new data produced {len(new_proportions)}"
        )

    baseline_safe = np.clip(baseline_proportions, EPSILON, None)
    new_safe = np.clip(new_proportions, EPSILON, None)

    psi = np.sum((new_safe - baseline_safe) * np.log(new_safe / baseline_safe))
    return float(psi)


def classify(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "SIGNIFICANT DRIFT"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate drift"
    return "stable"

def generate_report(results, input_path, baseline_path, report_path):
    from datetime import datetime

    drifted = [r for r in results if r[2] != "stable"]
    significant = [r for r in results if r[2] == "SIGNIFICANT DRIFT"]

    lines = [
        "# Drift Detection Report",
        "",
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Input data**: `{input_path}`",
        f"- **Baseline**: `{baseline_path}`",
        f"- **Features checked**: {len(results)}",
        f"- **Features with drift**: {len(drifted)} ({len(significant)} significant)",
        "",
        "## Summary",
        "",
        "| Feature | PSI | Status |",
        "|---|---|---|",
    ]
    for col, psi, status in results:
        lines.append(f"| `{col}` | {psi:.4f} | {status} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "PSI thresholds (industry standard):",
        f"- **< {PSI_MODERATE_THRESHOLD}**: stable, no action needed",
        f"- **{PSI_MODERATE_THRESHOLD}–{PSI_SIGNIFICANT_THRESHOLD}**: moderate drift, worth investigating",
        f"- **> {PSI_SIGNIFICANT_THRESHOLD}**: significant drift, model retraining should be considered",
        "",
    ]

    if significant:
        lines.append("## Recommended action")
        lines.append("")
        feature_list = ", ".join(f"`{r[0]}`" for r in significant)
        lines.append(
            f"Significant drift detected in: {feature_list}. "
            "Investigate the source of this shift (data pipeline change, "
            "genuine population shift, upstream bug) before trusting current "
            "model predictions, and consider retraining on more recent data."
        )
    else:
        lines.append("## Recommended action")
        lines.append("")
        lines.append("No significant drift detected. No action needed at this time.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to new data CSV to check for drift")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON from compute_baseline.py")
    parser.add_argument("--report", default=None, help="Optional path to save a markdown drift report")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with a non-zero code if any significant drift is found (for CI use)",
    )
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)

    df = pd.read_csv(args.input)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    print(f"Checking {df.shape[1]} features across {df.shape[0]} rows for drift...\n")

    results = []
    for col in df.columns:
        if col not in baseline:
            continue
        psi = compute_psi(df[col], baseline[col])
        status = classify(psi)
        results.append((col, psi, status))

    results.sort(key=lambda r: r[1], reverse=True)

    print(f"{'Feature':<45} {'PSI':>8}  Status")
    print("-" * 70)
    for col, psi, status in results:
        flag = " ⚠" if status != "stable" else ""
        print(f"{col:<45} {psi:>8.4f}  {status}{flag}")

    drifted = [r for r in results if r[2] != "stable"]
    significant = [r for r in results if r[2] == "SIGNIFICANT DRIFT"]
    print(f"\n{len(drifted)}/{len(results)} features show drift (PSI >= {PSI_MODERATE_THRESHOLD}).")

    if args.report:
        generate_report(results, args.input, args.baseline, args.report)

    if args.fail_on_drift and significant:
        print(f"\nFAILING: {len(significant)} feature(s) show significant drift.")
        exit(1)


if __name__ == "__main__":
    main()
