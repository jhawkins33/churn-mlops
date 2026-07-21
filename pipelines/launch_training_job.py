"""
Submits src/train.py as a managed SageMaker training job.

Run this locally (not inside SageMaker) — it uses the SageMaker Python SDK
to package up src/train.py, launch a training container in AWS, and point
it at the processed data in S3. The trained model artifact lands in
churn-mlops-artifacts-dev automatically.

Usage:
    python pipelines/launch_training_job.py
"""

import os
import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.sklearn.estimator import SKLearn

load_dotenv()

# --- Configuration ---
# Set these in a local .env file (gitignored) — see .env.example for the format.
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")

TRAIN_DATA_S3 = "s3://churn-mlops-processed-dev/"
OUTPUT_S3 = "s3://churn-mlops-artifacts-dev/"

INSTANCE_TYPE = "ml.m5.large"  # small, cheap CPU instance — plenty for this dataset size


def main():
    boto_session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    session = sagemaker.Session(boto_session=boto_session)

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        output_path=OUTPUT_S3,
        base_job_name="churn-mlops-training",
        sagemaker_session=session,
    )

    print("Submitting training job to SageMaker...")
    estimator.fit({"train": TRAIN_DATA_S3})
    
    with open(".last_model_data", "w") as f:
        f.write(estimator.model_data)

    print(f"\nTraining complete.")
    print(f"Model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()
