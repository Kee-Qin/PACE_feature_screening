"""Minimal feature-screening example using a public scikit-learn dataset."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.screening_utilities import compute_scores, aggregation_for_method

METHOD = "sis"  # Change to "pace_sis" after configuring the PACE checkpoint.
DEVICE = "cpu"
AGGREGATION = "mean"


def screen_table(X, y, method=METHOD, device=DEVICE, aggregation=AGGREGATION):
    """Return one importance score per input column, preserving column order."""
    options = {"aggregation": aggregation_for_method(method, aggregation)}
    scores = compute_scores(X.to_numpy(dtype=np.float32), np.asarray(y),
                            method=method, task="classification", device=device,
                            **options)
    return pd.DataFrame({"feature": X.columns, "importance": scores})

if __name__ == "__main__":
    data = load_breast_cancer(as_frame=True)
    feature_scores = screen_table(data.data, data.target)
    print(f"Scored all {len(data.data)} rows and {len(feature_scores)} features using {METHOD}.")
    print(feature_scores.sort_values("importance", ascending=False).to_string(index=False))
