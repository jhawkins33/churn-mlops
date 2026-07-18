# Churn MLOps Pipeline

An end-to-end MLOps pipeline for customer churn prediction on AWS — infrastructure as code, a data pipeline, and model training that runs identically on a laptop or as a managed SageMaker training job.

Built as a portfolio project to pair infrastructure/DevOps experience with practical ML engineering, using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Architecture

```
                    ┌─────────────────────┐
                    │   Terraform (IaC)    │
                    │  S3 buckets + IAM    │
                    └──────────┬───────────┘
                               │
   ┌───────────────┐   ┌──────▼───────┐   ┌───────────────────┐
   │  raw data      │──▶│ preprocessing│──▶│  processed data    │
   │  (S3, raw-dev) │   │ (src/*.py)   │   │ (S3, processed-dev)│
   └───────────────┘   └──────────────┘   └──────────┬─────────┘
                                                       │
                              ┌────────────────────────▼───────────────────────┐
                              │  Training (src/train.py)                        │
                              │  — runs locally OR as a SageMaker training job — │
                              └────────────────────────┬───────────────────────┘
                                                        │
                                              ┌─────────▼──────────┐
                                              │  Model artifact     │
                                              │  (S3, artifacts-dev)│
                                              └──────────────────────┘
```

All infrastructure is provisioned via Terraform, with remote state stored in S3.

## What's here

| Path | Purpose |
|---|---|
| `infrastructure/` | Terraform config — S3 buckets (raw / processed / artifacts), IAM execution role for SageMaker |
| `src/preprocess.py` | Cleans and encodes the raw dataset into model-ready features |
| `src/train.py` | Trains baseline models (Logistic Regression, Random Forest); runs unmodified locally or inside a SageMaker container |
| `src/inference.py` | Model-loading logic for a deployed SageMaker endpoint |
| `pipelines/launch_training_job.py` | Submits `train.py` as a managed SageMaker training job |

## Infrastructure

Three S3 buckets, each with public access blocked, AES256 encryption, and versioning where it matters (raw data and model artifacts — not the derived/regenerable processed layer):

- `churn-mlops-raw-dev` — landing zone for raw data
- `churn-mlops-processed-dev` — cleaned, feature-engineered data
- `churn-mlops-artifacts-dev` — trained model artifacts

Plus an IAM role scoped for SageMaker to assume, with S3 and SageMaker access.

## Data pipeline

The Telco churn dataset needs a few specific fixes before it's model-ready:

- `TotalCharges` is stored as a string, with 11 blank values for brand-new customers (`tenure == 0`) — coerced to numeric and filled with `0`, the mathematically correct value rather than an imputed guess.
- Binary Yes/No and gender columns mapped to 0/1.
- Multi-category columns (`Contract`, `PaymentMethod`, `InternetService`, etc.) one-hot encoded, with the first category dropped to avoid multicollinearity.
- Output: 7,043 rows × 31 columns, fully numeric, zero nulls.

## Training

`train.py` trains two baseline models and keeps the better one by ROC AUC:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression (scaled)** | 80.6% | 65.7% | 56.4% | 60.7% | **0.842** |
| Random Forest | 79.1% | 63.4% | 50.0% | 55.9% | 0.825 |

Logistic Regression wins and is saved as the model artifact. Both are in line with published benchmarks for this dataset (typically 0.82–0.86 AUC).

The same script runs two ways with no code changes:

```bash
# Locally
python src/train.py --input data/processed.csv --model-dir .

# As a managed SageMaker training job
python pipelines/launch_training_job.py
```

This works because the script reads SageMaker's environment variables (`SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`) when present, and falls back to sensible local defaults otherwise.

## Setup

**Prerequisites:** Python 3.12, an AWS account, Terraform ≥ 1.5, AWS CLI configured with a named profile.

```bash
# 1. Provision infrastructure
cd infrastructure
terraform init
terraform apply

# 2. Set up Python environment
cd ..
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install pandas scikit-learn joblib boto3 "sagemaker<3" python-dotenv

# 3. Configure environment variables
cp .env.example .env
# edit .env with your SAGEMAKER_ROLE_ARN (from `terraform output sagemaker_role_arn`)

# 4. Get the raw dataset into S3
aws s3 cp WA_Fn-UseC_-Telco-Customer-Churn.csv s3://churn-mlops-raw-dev/ --profile <your-profile>

# 5. Run the pipeline
aws s3 cp s3://churn-mlops-raw-dev/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw.csv --profile <your-profile>
python src/preprocess.py --input data/raw.csv --output data/processed.csv
aws s3 cp data/processed.csv s3://churn-mlops-processed-dev/processed.csv --profile <your-profile>
python pipelines/launch_training_job.py
```

## Notes on the SageMaker SDK

This project pins `sagemaker<3`. The v3 SDK (released Nov 2025) removed the `Estimator`/`SKLearn` classes used here in favor of a new `ModelTrainer` API — pinning to the last stable v2.x release avoids a moving target while that API settles.

## Roadmap

- [x] Infrastructure as code (Terraform)
- [x] Data pipeline (raw → processed)
- [x] Local baseline training
- [x] Managed SageMaker training job
- [ ] SageMaker endpoint deployment
- [ ] CI/CD (GitHub Actions for plan/apply, retraining triggers)
- [ ] Model monitoring / drift detection
