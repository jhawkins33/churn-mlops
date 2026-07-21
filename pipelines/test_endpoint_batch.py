"""
Sends multiple test prediction requests to the deployed endpoint,
comparing predictions against known actual labels.

Usage:
    python pipelines/test_endpoint_batch.py
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")
ENDPOINT_NAME = "churn-mlops-endpoint"

# Three real rows sampled from processed.csv, with known actual labels.
SAMPLES = [
    {
        "row": [0.0, 0.0, 0.0, 0.0, 41.0, 1.0, 1.0, 79.85, 3320.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "actual": 0,
    },
    {
        "row": [0.0, 1.0, 0.0, 0.0, 66.0, 1.0, 1.0, 102.4, 6471.85, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "actual": 0,
    },
    {
        "row": [0.0, 0.0, 0.0, 0.0, 12.0, 1.0, 1.0, 45.0, 524.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "actual": 0,
    },
]


def main():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    runtime = session.client("sagemaker-runtime")

    correct = 0
    for i, sample in enumerate(SAMPLES):
        body = ",".join(str(v) for v in sample["row"])
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=body,
        )
        result = response["Body"].read().decode().strip()
        predicted = int(result.strip("[]"))
        match = "✓" if predicted == sample["actual"] else "✗"
        if predicted == sample["actual"]:
            correct += 1
        print(f"Row {i+1}: predicted={predicted}  actual={sample['actual']}  {match}")

    print(f"\n{correct}/{len(SAMPLES)} correct")


if __name__ == "__main__":
    main()