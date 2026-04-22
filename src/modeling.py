from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DatasetSpec, add_time_features, coerce_types, validate_dataset

# Spotify-style audio features only (excludes e.g. spotify_popularity, duration_ms).
SPOTIFY_AUDIO_FEATURE_COLUMNS: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    model_max_iter: int = 2000
    max_audio_missing_frac: float | None = 0.5
    # If set, keep rows with release_year >= max(release_year) - recent_years_window. None = no year filter.
    recent_years_window: int | None = 4


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    use_smote: bool,
    random_state: int,
    model_max_iter: int,
    smote_k_neighbors: int = 5,
) -> ImbPipeline:
    numeric_transformer = PipelineSteps.numeric()
    categorical_transformer = PipelineSteps.categorical()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=model_max_iter, class_weight=None)

    steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)))
    steps.append(("clf", clf))
    return ImbPipeline(steps=steps)


class PipelineSteps:
    @staticmethod
    def numeric():
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    @staticmethod
    def numeric_trees():
        """Trees do not need scaling; imputation only."""
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    @staticmethod
    def categorical():
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )


def _prepare_xy_for_training(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: TrainConfig | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray, dict[str, Any]]:
    spec = spec or DatasetSpec()
    cfg = cfg or TrainConfig()

    df = add_time_features(coerce_types(df_raw, spec), spec)
    issues = validate_dataset(df, spec)
    if issues:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(issues))

    if cfg.recent_years_window is not None and "release_year" in df.columns:
        yr = pd.to_numeric(df["release_year"], errors="coerce")
        max_y = yr.max()
        if pd.notna(max_y):
            cutoff = int(max_y) - cfg.recent_years_window
            df = df.loc[yr >= cutoff].copy()

    audio_cols = [c for c in SPOTIFY_AUDIO_FEATURE_COLUMNS if c in df.columns]
    n_rows_dropped_audio_missing = 0
    if cfg.max_audio_missing_frac is not None and audio_cols:
        missing_frac = df[audio_cols].isna().mean(axis=1)
        keep = missing_frac <= cfg.max_audio_missing_frac
        n_rows_dropped_audio_missing = int((~keep).sum())
        df = df.loc[keep].copy()

    base_numeric = [c for c in spec.numeric_cols if c in df.columns]
    time_numeric = [c for c in ["release_year", "release_month", "release_dow"] if c in df.columns]
    numeric_features = base_numeric + time_numeric
    categorical_features = [c for c in spec.categorical_cols if c in df.columns]

    y = df[spec.target_col].astype(int).to_numpy()

    meta = {
        "audio_feature_columns": audio_cols,
        "max_audio_missing_frac": (
            cfg.max_audio_missing_frac
            if cfg.max_audio_missing_frac is not None and audio_cols
            else None
        ),
        "n_rows_dropped_audio_missing": n_rows_dropped_audio_missing,
        "recent_years_window": cfg.recent_years_window,
    }
    return df, numeric_features, categorical_features, y, meta


def _smote_settings(y_train: np.ndarray, *, use_smote: bool, random_state: int) -> tuple[bool, int]:
    use = use_smote
    smote_k = 5
    if use:
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist(), strict=False))
        minority = min(class_counts.values()) if class_counts else 0
        if minority <= 1:
            use = False
        else:
            smote_k = max(1, min(5, minority - 1))
    return use, smote_k


def build_rf_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    use_smote: bool,
    random_state: int,
    smote_k_neighbors: int,
    n_estimators: int = 200,
    max_depth: int | None = 20,
) -> ImbPipeline:
    numeric_transformer = PipelineSteps.numeric_trees()
    categorical_transformer = PipelineSteps.categorical()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)))
    steps.append(("clf", clf))
    return ImbPipeline(steps=steps)


def train_evaluate_baseline(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    spec = spec or DatasetSpec()
    cfg = cfg or TrainConfig()

    df, numeric_features, categorical_features, y, prep_meta = _prepare_xy_for_training(df_raw, spec=spec, cfg=cfg)
    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    use_smote, smote_k = _smote_settings(y_train, use_smote=cfg.use_smote, random_state=cfg.random_state)

    pipe = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_smote=use_smote,
        random_state=cfg.random_state,
        model_max_iter=cfg.model_max_iter,
        smote_k_neighbors=smote_k,
    )
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    auc = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            auc = float(roc_auc_score(y_test, proba))

    metrics = {
        "model": "logistic_regression",
        "n_rows": int(df.shape[0]),
        "recent_years_window": prep_meta["recent_years_window"],
        "audio_feature_columns": prep_meta["audio_feature_columns"],
        "max_audio_missing_frac": prep_meta["max_audio_missing_frac"],
        "n_rows_dropped_audio_missing": prep_meta["n_rows_dropped_audio_missing"],
        "n_features_numeric": int(len(numeric_features)),
        "n_features_categorical": int(len(categorical_features)),
        "test_size": cfg.test_size,
        "use_smote": use_smote,
        "smote_k_neighbors": smote_k if use_smote else None,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "feature_columns": {"numeric": numeric_features, "categorical": categorical_features},
    }

    return {"pipeline": pipe, "metrics": metrics, "config": asdict(cfg)}


@dataclass(frozen=True)
class RandomForestTrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    max_audio_missing_frac: float | None = 0.5
    n_estimators: int = 200
    max_depth: int | None = 20
    # None = use all years (more positives for train/test); 4 matches logistic regression window.
    recent_years_window: int | None = None


def train_evaluate_random_forest(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: RandomForestTrainConfig | None = None,
) -> dict[str, Any]:
    spec = spec or DatasetSpec()
    cfg = cfg or RandomForestTrainConfig()

    # Reuse TrainConfig fields that overlap with filtering logic.
    base_cfg = TrainConfig(
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        use_smote=cfg.use_smote,
        max_audio_missing_frac=cfg.max_audio_missing_frac,
        recent_years_window=cfg.recent_years_window,
    )

    df, numeric_features, categorical_features, y, prep_meta = _prepare_xy_for_training(df_raw, spec=spec, cfg=base_cfg)
    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    use_smote, smote_k = _smote_settings(y_train, use_smote=cfg.use_smote, random_state=cfg.random_state)

    pipe = build_rf_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_smote=use_smote,
        random_state=cfg.random_state,
        smote_k_neighbors=smote_k,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
    )
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    auc = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            auc = float(roc_auc_score(y_test, proba))

    metrics = {
        "model": "random_forest",
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "n_rows": int(df.shape[0]),
        "recent_years_window": prep_meta["recent_years_window"],
        "audio_feature_columns": prep_meta["audio_feature_columns"],
        "max_audio_missing_frac": prep_meta["max_audio_missing_frac"],
        "n_rows_dropped_audio_missing": prep_meta["n_rows_dropped_audio_missing"],
        "n_features_numeric": int(len(numeric_features)),
        "n_features_categorical": int(len(categorical_features)),
        "test_size": cfg.test_size,
        "use_smote": use_smote,
        "smote_k_neighbors": smote_k if use_smote else None,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "feature_columns": {"numeric": numeric_features, "categorical": categorical_features},
    }

    return {"pipeline": pipe, "metrics": metrics, "config": asdict(cfg)}


def save_artifacts(
    out_dir: str | Path,
    *,
    pipeline,
    metrics: dict[str, Any],
    config: dict[str, Any],
    pipeline_filename: str = "baseline_pipeline.joblib",
    metrics_filename: str = "metrics.json",
    config_filename: str = "train_config.json",
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / pipeline_filename
    metrics_path = out / metrics_filename
    config_path = out / config_filename

    dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    config_path.write_text(json.dumps(config, indent=2))

    return {"model": str(model_path), "metrics": str(metrics_path), "config": str(config_path)}

