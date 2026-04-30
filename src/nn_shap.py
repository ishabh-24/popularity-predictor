from __future__ import annotations

"""SHAP helpers for the neural-network (HitNet) bundle.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
import torch


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


def feature_importance_from_shap_matrix(shap_matrix: np.ndarray) -> np.ndarray:
    """Compute mean absolute SHAP importance per transformed feature."""
    arr = np.asarray(shap_matrix)
    if arr.ndim < 2:
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    reduce_axes = tuple(ax for ax in range(arr.ndim) if ax != 1)
    mean_abs = np.mean(np.abs(arr), axis=reduce_axes)
    return np.ravel(mean_abs)


def shap_summary_for_hitnet_bundle(
    bundle: Any,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    random_state: int = 42,
    max_background: int = 200,
    max_explain: int = 250,
    top_k: int = 15,
) -> list[dict[str, float | str]]:
    """Return top-k global SHAP importances for a trained HitNet bundle."""
    if not hasattr(bundle, "preprocessor") or not hasattr(bundle, "ensure_model"):
        raise ValueError("Expected a HitNetClassifierBundle-like object.")

    preprocessor = bundle.preprocessor
    model = bundle.ensure_model()
    model.eval()

    X_bg_df = sample_rows(X_train, max_background, random_state)
    X_exp_df = sample_rows(X_explain, max_explain, random_state + 1)

    X_bg_np = np.asarray(preprocessor.transform(X_bg_df), dtype=np.float32)
    X_exp_np = np.asarray(preprocessor.transform(X_exp_df), dtype=np.float32)
    feature_names = preprocessor.get_feature_names_out()

    X_bg_t = torch.tensor(X_bg_np, dtype=torch.float32)
    X_exp_t = torch.tensor(X_exp_np, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, X_bg_t)
    shap_values = explainer.shap_values(X_exp_t)
    shap_matrix = extract_binary_shap_values(shap_values)
    mean_abs = feature_importance_from_shap_matrix(shap_matrix)

    k = min(top_k, len(feature_names))
    order = np.argsort(mean_abs)[::-1][:k]
    return [
        {"feature": str(feature_names[i]), "mean_abs_shap": float(mean_abs[i])}
        for i in order
    ]
