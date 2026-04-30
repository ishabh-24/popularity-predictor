from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass
from collections.abc import Callable
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


""" We need to build the pipeline, train the model, and evaluate the model.
We did this using scikit-learn.

Then we implemented a Logistic Regression model to use as a baseline for comparison.
Justification for Logistic Regression: Logistic Regression is a simple model that is easy to interpret
and can help us understand a basic relationship between the features and the target variable. While it
is not expected to perform as well as the other models we implement, it serves as a useful baseline
to compare the prediction capabilities of the other models, and how they improve upon different metrics
like AUC or recall (significant improvement happened). A simple model also guards against overfitting
in comparison to more complex models - this is important with our limited dataset.

This file is also where we have our implementation of Ensemble Models - our random forest 
implementation is composed of 200 different decision trees that are trained on different subsets of
the data. The method build_rf_pipeline builds the pipeline for the random forest ensemble models.

Justification for ensemble models (RF): We know from our EDA analysis that predicting a hit or non hit
is not a linear problem - this makes sense because we have a high number of features to consider (e.g.
danceability, energy, genre, etc.), that all have an effect on the target variable, so a more complex
model is needed to capture this complex relationship. Random Forest decision trees are high performing
for binary classification tasks, and can help reduce overfitting through bagging which makes them 
more robust for this task.

Justification for SMOTE: We know from our EDA analysis that our dataset is imbalanced - we have a 
much higher number of non hits than hits naturally, since most songs never hit the Billboard Hot 100.
SMOTE is a technique that oversamples the minority class to help balance the dataset. It is a good 
choice because it simply and effectively balances the dataset without introducing too much noise.
"""

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
    max_audio_missing_frac: float | None = None
    recent_years_window: int | None = None


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
        # Numeric preprocessing pipeline for linear models:
        # - `SimpleImputer(strategy='median')` handles missing numeric values and
        #   better handles outliers compared to mean imputation.
        # - `StandardScaler` scales features to zero mean / unit variance which
        #   improves convergence and interpretability for linear models like
        #   `LogisticRegression`.
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    @staticmethod
    def numeric_trees():
        # Numeric preprocessing for tree-based models:
        # - We only impute missing values (median) and do not apply standard scaling. Median imputation
        #   also decreases the direct effect of outliers without aggressively taking them away which could
        #   remove informative relationships for trees.
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


def prepare_xy_for_training(
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
    # The `max_audio_missing_frac` threshold is a preprocessing step
    # that drops rows where a large fraction of audio features are missing
    # (e.g., incomplete Spotify records). 

    base_numeric = [c for c in spec.numeric_cols if c in df.columns]
    time_numeric = [c for c in ["release_year", "release_month"] if c in df.columns]
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


def smote_settings(y_train: np.ndarray, *, use_smote: bool) -> tuple[bool, int]:
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
    max_depth: int | None = 10,
    min_samples_leaf: int = 5,
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
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        random_state = random_state,
        class_weight = "balanced_subsample",
        n_jobs = -1,
    )

    steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state = random_state, k_neighbors = smote_k_neighbors)))
    steps.append(("clf", clf))
    return ImbPipeline(steps=steps)


def train_evaluate(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec,
    base_cfg: TrainConfig,
    build_pipeline_fn: Callable[[list[str], list[str], bool, int], ImbPipeline],
    extra_metrics: dict[str, Any],
) -> dict[str, Any]:
    df, numeric_features, categorical_features, y, prep_meta = prepare_xy_for_training(
        df_raw, spec=spec, cfg=base_cfg
    )
    X = df[numeric_features + categorical_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=base_cfg.test_size,
        random_state=base_cfg.random_state,
        stratify=y,
    )
    use_smote, smote_k = smote_settings(y_train, use_smote=base_cfg.use_smote)
    pipe = build_pipeline_fn(numeric_features, categorical_features, use_smote, smote_k)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    auc = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            auc = float(roc_auc_score(y_test, proba))

    prep_keys = (
        "recent_years_window",
        "audio_feature_columns",
        "max_audio_missing_frac",
        "n_rows_dropped_audio_missing",
    )
    metrics = {
        **extra_metrics,
        "n_rows": int(df.shape[0]),
        **{k: prep_meta[k] for k in prep_keys},
        "n_features_numeric": len(numeric_features),
        "n_features_categorical": len(categorical_features),
        "test_size": base_cfg.test_size,
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
    return {"pipeline": pipe, "metrics": metrics, "X_train": X_train, "X_test": X_test}


def train_evaluate_baseline(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    spec = spec or DatasetSpec()
    cfg = cfg or TrainConfig()

    def build_pipe(
        nf: list[str], cf: list[str], use_smote: bool, smote_k: int
    ) -> ImbPipeline:
        return build_pipeline(
            numeric_features=nf,
            categorical_features=cf,
            use_smote=use_smote,
            random_state=cfg.random_state,
            model_max_iter=cfg.model_max_iter,
            smote_k_neighbors=smote_k,
        )

    out = train_evaluate(
        df_raw,
        spec=spec,
        base_cfg=cfg,
        build_pipeline_fn=build_pipe,
        extra_metrics={"model": "logistic_regression"},
    )
    return {**out, "config": asdict(cfg)}


@dataclass(frozen=True)
class RandomForestTrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    max_audio_missing_frac: float | None = None
    n_estimators: int = 200
    max_depth: int | None = 10
    min_samples_leaf: int = 5
    recent_years_window: int | None = None


def train_evaluate_random_forest(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: RandomForestTrainConfig | None = None,
) -> dict[str, Any]:
    spec = spec or DatasetSpec()
    cfg = cfg or RandomForestTrainConfig()

    base_cfg = TrainConfig(
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        use_smote=cfg.use_smote,
        max_audio_missing_frac=cfg.max_audio_missing_frac,
        recent_years_window=cfg.recent_years_window,
    )

    def build_pipe(
        nf: list[str], cf: list[str], use_smote: bool, smote_k: int
    ) -> ImbPipeline:
        return build_rf_pipeline(
            numeric_features=nf,
            categorical_features=cf,
            use_smote=use_smote,
            random_state=cfg.random_state,
            smote_k_neighbors=smote_k,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
        )

    out = train_evaluate(
        df_raw,
        spec=spec,
        base_cfg=base_cfg,
        build_pipeline_fn=build_pipe,
        extra_metrics={
            "model": "random_forest",
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_samples_leaf": cfg.min_samples_leaf,
        },
    )
    return {**out, "config": asdict(cfg)}


def save_artifacts(
    out_dir: str | os.PathLike[str],
    *,
    pipeline,
    metrics: dict[str, Any],
    config: dict[str, Any],
    pipeline_filename: str = "baseline_pipeline.joblib",
    metrics_filename: str = "metrics.json",
    config_filename: str = "train_config.json",
) -> dict[str, str]:
    out = os.fspath(out_dir)
    os.makedirs(out, exist_ok=True)

    model_path = os.path.join(out, pipeline_filename)
    metrics_path = os.path.join(out, metrics_filename)
    config_path = os.path.join(out, config_filename)

    dump(pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2))
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2))

    return {"model": model_path, "metrics": metrics_path, "config": config_path}

