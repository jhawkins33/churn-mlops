"""
Inference entry point for the deployed SageMaker endpoint.
"""

import os
import joblib
import numpy as np


def model_fn(model_dir):
    """Load the trained model from the directory SageMaker extracts model.tar.gz into."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def predict_fn(input_data, model):
    """
    Reshape input to 2D before predicting. The container's default CSV
    parser can hand us a flat 1D array for a single row — sklearn's
    Pipeline (StandardScaler + LogisticRegression) requires 2D input,
    one row per sample, even for a single sample.
    """
    input_data = np.array(input_data)
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    return model.predict(input_data)