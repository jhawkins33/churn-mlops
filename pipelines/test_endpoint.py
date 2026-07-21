"""
Sends a test prediction request to the deployed endpoint.

Usage:
    python pipelines/test_endpoint.py
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")
ENDPOINT_NAME = "churn-mlops-endpoint"

# Real first row from processed.csv (customer 7590-VHVEG), 30 features,
# in the exact column order pandas produced. Known actual label: not churned.
SAMPLE_ROW = "0.0,0.0,1.0,0.0,1.0,0.0,1.0,29.85,29.85,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0"


def main():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    runtime = session.client("sagemaker-runtime")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=SAMPLE_ROW,
    )

    result = response["Body"].read().decode()
    print(f"Prediction: {result.strip()}")
    print("(0 = not churned, 1 = churned)")
    print("Actual label for this customer: 0 (not churned)")


if __name__ == "__main__":
    main()