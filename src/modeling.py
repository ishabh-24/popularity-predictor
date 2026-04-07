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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DatasetSpec, add_time_features, coerce_types, validate_dataset


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    model_max_iter: int = 2000


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
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    @staticmethod
    def categorical():
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )


def train_evaluate_baseline(
    df_raw: pd.DataFrame,
    *,
    spec: DatasetSpec | None = None,
    cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    spec = spec or DatasetSpec()
    cfg = cfg or TrainConfig()

    df = add_time_features(coerce_types(df_raw, spec), spec)
    issues = validate_dataset(df, spec)
    if issues:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(issues))

    # Build feature lists from what's actually present.
    base_numeric = [c for c in spec.numeric_cols if c in df.columns]
    time_numeric = [c for c in ["release_year", "release_month", "release_dow"] if c in df.columns]
    numeric_features = base_numeric + time_numeric

    categorical_features = [c for c in spec.categorical_cols if c in df.columns]

    y = df[spec.target_col].astype(int).to_numpy()
    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # SMOTE needs at least (k_neighbors + 1) minority samples in the *training* split.
    # If the dataset is small (like our sample), auto-tune k or disable SMOTE.
    use_smote = cfg.use_smote
    smote_k = 5
    if use_smote:
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist(), strict=False))
        minority = min(class_counts.values()) if class_counts else 0
        if minority <= 1:
            use_smote = False
        else:
            smote_k = max(1, min(5, minority - 1))

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

    # AUC: only if predict_proba exists and both classes present
    auc = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            auc = float(roc_auc_score(y_test, proba))

    metrics = {
        "n_rows": int(df.shape[0]),
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
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "baseline_pipeline.joblib"
    metrics_path = out / "metrics.json"
    config_path = out / "train_config.json"

    dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    config_path.write_text(json.dumps(config, indent=2))

    return {"model": str(model_path), "metrics": str(metrics_path), "config": str(config_path)}

