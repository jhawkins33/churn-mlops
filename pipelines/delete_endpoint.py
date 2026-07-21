"""
Deletes the SageMaker endpoint, its config, and the model — stops all
hourly billing. Run this after you're done testing/demoing.

Usage:
    python pipelines/delete_endpoint.py
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")
ENDPOINT_NAME = "churn-mlops-endpoint"


def main():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    sm = session.client("sagemaker")

    print(f"Deleting endpoint '{ENDPOINT_NAME}'...")
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)

    print(f"Deleting endpoint config '{ENDPOINT_NAME}'...")
    sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)

    print("Done. Endpoint and config deleted — billing stopped.")


if __name__ == "__main__":
    main()