from __future__ import annotations

"""SHAP helpers for the random-forest pipeline.
"""

from typing import Any
import numpy as np
import pandas as pd
import shap

#depends on a fixed number of maxrows and this adjusts the df accordingly
def sample_rows(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows: 
        return df
    return df.sample(n=max_rows, random_state=random_state)


def extract_binary_shap_values(values: Any) -> np.ndarray:
    """Normalize SHAP output into `(n_samples, n_features)` for class 1."""
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        if len(values) == 2:
            return np.asarray(values[1])
        return np.asarray(values[0])
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[2] == 2:
        return arr[:, :, 1]
    return arr


def to_dense(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def feature_importance_from_shap_matrix(shap_matrix: np.ndarray) -> np.ndarray:
    """Compute mean absolute SHAP importance per transformed feature."""
    arr = np.asarray(shap_matrix)
    if arr.ndim < 2:
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    reduce_axes = tuple(ax for ax in range(arr.ndim) if ax != 1)
    mean_abs = np.mean(np.abs(arr), axis=reduce_axes)
    return np.ravel(mean_abs)


def shap_summary_for_random_forest_pipeline(
    pipe: Any,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    random_state: int = 42,
    max_background: int = 300,
    max_explain: int = 400,
    top_k: int = 15,
) -> list[dict[str, float | str]]:
    """Return top-k global SHAP importances for a fitted RF pipeline."""
    if not hasattr(pipe, "named_steps"):
        raise ValueError("Expected an imblearn pipeline with named_steps.")
    if "preprocess" not in pipe.named_steps or "clf" not in pipe.named_steps:
        raise ValueError("Pipeline must include 'preprocess' and 'clf' steps.")

    preprocess = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]

    X_bg_df = sample_rows(X_train, max_background, random_state)
    X_exp_df = sample_rows(X_explain, max_explain, random_state + 1)

    X_bg = to_dense(preprocess.transform(X_bg_df))
    X_exp = to_dense(preprocess.transform(X_exp_df))
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(clf, data=X_bg, model_output="raw")
    shap_values = explainer.shap_values(X_exp)
    shap_matrix = extract_binary_shap_values(shap_values)
    mean_abs = feature_importance_from_shap_matrix(shap_matrix)

    k = min(top_k, len(feature_names))
    order = np.argsort(mean_abs)[::-1][:k]
    return [
        {"feature": str(feature_names[i]), "mean_abs_shap": float(mean_abs[i])}
        for i in order
    ]
