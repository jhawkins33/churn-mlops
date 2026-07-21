"""
Deploys the trained model as a live SageMaker endpoint.

Reads the model artifact path from .last_model_data (written by
launch_training_job.py), or accepts an override via --model-data.

Usage:
    python pipelines/deploy_endpoint.py
    python pipelines/deploy_endpoint.py --model-data s3://.../model.tar.gz
"""

import argparse
import os
import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.sklearn.model import SKLearnModel

load_dotenv()

ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")

ENDPOINT_NAME = "churn-mlops-endpoint"
INSTANCE_TYPE = "ml.m5.large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-data",
        default=None,
        help="S3 URI to model.tar.gz. Defaults to reading .last_model_data.",
    )
    args = parser.parse_args()

    model_data = args.model_data
    if model_data is None:
        with open(".last_model_data") as f:
            model_data = f.read().strip()

    print(f"Deploying model: {model_data}")

    boto_session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    session = sagemaker.Session(boto_session=boto_session)

    model = SKLearnModel(
        model_data=model_data,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir="src",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
    )

    print(f"Deploying endpoint '{ENDPOINT_NAME}' on {INSTANCE_TYPE}...")
    print("This takes a few minutes...")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    print(f"\nEndpoint deployed: {predictor.endpoint_name}")
    print("Remember: this endpoint bills per hour until deleted.")
    print("Run pipelines/delete_endpoint.py when you're done testing.")


if __name__ == "__main__":
    main()