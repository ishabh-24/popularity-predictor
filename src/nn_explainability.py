from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .nn_modeling import HitNetClassifierBundle

try:
    from captum.attr import IntegratedGradients
except ImportError:  # pragma: no cover - exercised only when optional dependency is missing
    IntegratedGradients = None


@dataclass(frozen=True)
class NNExplainabilityConfig:
    max_samples: int = 512
    permutation_repeats: int = 5
    ig_steps: int = 64
    baseline_strategy: str = "median"
    random_state: int = 42
    top_k_local: int = 15


def _validate_input_schema(bundle: HitNetClassifierBundle, X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Explainability input must be a pandas DataFrame.")

    required = list(getattr(bundle.preprocessor, "feature_names_in_", []))
    if required:
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for preprocessing: {missing}")
        return X[required]
    return X


def _predict_proba_from_transformed(bundle: HitNetClassifierBundle, Xt: np.ndarray) -> np.ndarray:
    xb = torch.tensor(np.asarray(Xt, dtype=np.float32), dtype=torch.float32)
    model = bundle._ensure_model()
    with torch.no_grad():
        logits = model(xb)
        p1 = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
    return np.clip(np.atleast_1d(p1), 0.0, 1.0)


def get_transformed_feature_names(bundle: HitNetClassifierBundle) -> list[str]:
    pre = bundle.preprocessor
    if hasattr(pre, "get_feature_names_out"):
        return [str(x) for x in pre.get_feature_names_out()]
    return [f"x{i}" for i in range(bundle.input_dim)]


def _column_groups(bundle: HitNetClassifierBundle, transformed_names: list[str]) -> dict[str, list[int]]:
    cat_columns: list[str] = []
    for name, _transformer, columns in getattr(bundle.preprocessor, "transformers_", []):
        if name == "cat":
            cat_columns = [str(c) for c in columns]

    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(transformed_names):
        group = name
        if name.startswith("num__"):
            group = name[len("num__") :]
        elif name.startswith("cat__"):
            raw = name[len("cat__") :]
            group = raw
            for col in cat_columns:
                if raw == col or raw.startswith(f"{col}_"):
                    group = col
                    break

        groups.setdefault(group, []).append(idx)
    return groups


def compute_permutation_importance(
    bundle: HitNetClassifierBundle,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cfg: NNExplainabilityConfig,
) -> pd.DataFrame:
    X_in = _validate_input_schema(bundle, X)
    y_arr = np.asarray(y, dtype=np.int64)
    if y_arr.ndim != 1:
        raise ValueError("Expected y to be one-dimensional for permutation importance.")

    if cfg.max_samples > 0 and len(X_in) > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_state)
        idx = np.sort(rng.choice(len(X_in), size=cfg.max_samples, replace=False))
        X_in = X_in.iloc[idx].copy()
        y_arr = y_arr[idx]

    Xt = np.asarray(bundle.preprocessor.transform(X_in), dtype=np.float32)
    feature_names = get_transformed_feature_names(bundle)

    p_base = _predict_proba_from_transformed(bundle, Xt)
    pred_base = (p_base >= bundle.decision_threshold).astype(np.int64)
    baseline_score = float(f1_score(y_arr, pred_base, zero_division=0))

    rng = np.random.default_rng(cfg.random_state)
    per_feature_scores: list[dict[str, Any]] = []
    for j, feature in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(cfg.permutation_repeats):
            Xt_perm = Xt.copy()
            Xt_perm[:, j] = Xt_perm[rng.permutation(Xt_perm.shape[0]), j]
            p_perm = _predict_proba_from_transformed(bundle, Xt_perm)
            pred_perm = (p_perm >= bundle.decision_threshold).astype(np.int64)
            perm_score = float(f1_score(y_arr, pred_perm, zero_division=0))
            drops.append(baseline_score - perm_score)

        per_feature_scores.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
                "baseline_f1": baseline_score,
            }
        )

    out = pd.DataFrame(per_feature_scores).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def compute_local_integrated_gradients(
    bundle: HitNetClassifierBundle,
    X_rows: pd.DataFrame,
    X_reference: pd.DataFrame,
    *,
    cfg: NNExplainabilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if IntegratedGradients is None:
        raise ImportError("captum is required for Integrated Gradients. Install it with `pip install captum`.")

    X_rows_in = _validate_input_schema(bundle, X_rows)
    X_ref_in = _validate_input_schema(bundle, X_reference)
    if X_rows_in.empty:
        raise ValueError("At least one row is required for local integrated gradients.")

    if cfg.max_samples > 0 and len(X_ref_in) > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_state)
        idx = np.sort(rng.choice(len(X_ref_in), size=cfg.max_samples, replace=False))
        X_ref_in = X_ref_in.iloc[idx].copy()

    Xt_rows = np.asarray(bundle.preprocessor.transform(X_rows_in), dtype=np.float32)
    Xt_ref = np.asarray(bundle.preprocessor.transform(X_ref_in), dtype=np.float32)
    transformed_names = get_transformed_feature_names(bundle)

    if cfg.baseline_strategy != "median":
        raise ValueError("Unsupported baseline strategy. Use 'median'.")

    baseline = np.nanmedian(Xt_ref, axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, 0.0).astype(np.float32)

    rows_tensor = torch.tensor(Xt_rows, dtype=torch.float32)
    baseline_tensor = torch.tensor(np.repeat(baseline.reshape(1, -1), Xt_rows.shape[0], axis=0), dtype=torch.float32)

    model = bundle._ensure_model()
    model.eval()
    ig = IntegratedGradients(model)
    attrs_tensor = ig.attribute(rows_tensor, baselines=baseline_tensor, n_steps=cfg.ig_steps)
    attrs = attrs_tensor.detach().cpu().numpy()

    raw_df = pd.DataFrame(attrs, columns=transformed_names)
    raw_df.insert(0, "row_index", X_rows_in.index.to_numpy())

    groups = _column_groups(bundle, transformed_names)
    grouped_rows: list[dict[str, Any]] = []
    for row_offset, row_idx in enumerate(X_rows_in.index.to_numpy()):
        row_vals = attrs[row_offset]
        for group, indices in groups.items():
            value = float(np.sum(row_vals[indices]))
            grouped_rows.append(
                {
                    "row_index": int(row_idx),
                    "feature_group": group,
                    "attribution": value,
                    "abs_attribution": abs(value),
                }
            )

    grouped_df = pd.DataFrame(grouped_rows)
    grouped_df["rank"] = grouped_df.groupby("row_index")["abs_attribution"].rank(method="first", ascending=False).astype(int)
    grouped_df = grouped_df.sort_values(["row_index", "rank"]).reset_index(drop=True)

    raw_sums = raw_df.drop(columns=["row_index"]).sum(axis=1).to_numpy()
    grouped_sums = grouped_df.groupby("row_index")["attribution"].sum().reindex(X_rows_in.index).to_numpy()
    max_abs_diff = float(np.max(np.abs(raw_sums - grouped_sums))) if len(raw_sums) else 0.0

    checks = {
        "raw_attribution_width": int(raw_df.shape[1] - 1),
        "transformed_input_dim": int(Xt_rows.shape[1]),
        "max_abs_grouped_sum_diff": max_abs_diff,
    }
    return raw_df, grouped_df, checks


def run_nn_explainability(
    bundle: HitNetClassifierBundle,
    *,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    X_local: pd.DataFrame,
    out_dir: str | Path,
    cfg: NNExplainabilityConfig | None = None,
) -> dict[str, str]:
    cfg = cfg or NNExplainabilityConfig()

    permutation_df = compute_permutation_importance(bundle, X_eval, y_eval, cfg=cfg)
    raw_ig_df, grouped_ig_df, checks = compute_local_integrated_gradients(bundle, X_local, X_eval, cfg=cfg)
    local_top_df = grouped_ig_df[grouped_ig_df["rank"] <= cfg.top_k_local].copy()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_path = out / "nn_global_permutation_importance.csv"
    local_raw_path = out / "nn_local_integrated_gradients_raw.csv"
    local_grouped_path = out / "nn_local_integrated_gradients_grouped.csv"
    local_top_path = out / "nn_local_integrated_gradients.csv"
    meta_path = out / "nn_explainability_metadata.json"

    permutation_df.to_csv(global_path, index=False)
    raw_ig_df.to_csv(local_raw_path, index=False)
    grouped_ig_df.to_csv(local_grouped_path, index=False)
    local_top_df.to_csv(local_top_path, index=False)

    metadata = {
        "method_global": "permutation_importance",
        "method_local": "integrated_gradients",
        "config": asdict(cfg),
        "n_eval_rows": int(len(X_eval)),
        "n_local_rows": int(len(X_local)),
        "checks": checks,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "global": str(global_path),
        "local_raw": str(local_raw_path),
        "local_grouped": str(local_grouped_path),
        "local_top": str(local_top_path),
        "metadata": str(meta_path),
    }
