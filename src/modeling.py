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
from sklearn.pipeline import Pipeline

# Importing from your existing data module
from .data import DatasetSpec, add_time_features, coerce_types, validate_dataset

@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    model_max_iter: int = 2000

class PipelineSteps:
    @staticmethod
    def numeric() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    @staticmethod
    def categorical() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    use_smote: bool,
    random_state: int,
    model_max_iter: int,
    smote_k_neighbors: int = 5,
) -> ImbPipeline:
    """
    Constructs an imbalanced-learn pipeline including preprocessing, 
    oversampling, and the classifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", PipelineSteps.numeric(), numeric_features),
            ("cat", PipelineSteps.categorical(), categorical_features),
        ],
        remainder="drop",
    )

    # Baseline: Logistic Regression
    clf = LogisticRegression(max_iter=model_max_iter, random_state=random_state)

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

    # 1. Preprocess & Validate
    df = add_time_features(coerce_types(df_raw, spec), spec)
    issues = validate_dataset(df, spec)
    if issues:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(issues))

    # 2. Feature Selection
    numeric_features = [c for c in spec.numeric_cols if c in df.columns]
    # Add time features created by add_time_features
    for time_feat in ["release_year", "release_month", "release_dow"]:
        if time_feat in df.columns:
            numeric_features.append(time_feat)
            
    categorical_features = [c for c in spec.categorical_cols if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df[spec.target_col].astype(int)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # 4. Handle SMOTE constraints
    use_smote = cfg.use_smote
    smote_k = 5
    if use_smote:
        minority_count = y_train.value_counts().min()
        if minority_count <= 1:
            use_smote = False # Cannot oversample with only 1 sample
        else:
            smote_k = min(5, minority_count - 1)

    # 5. Fit
    pipe = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_smote=use_smote,
        random_state=cfg.random_state,
        model_max_iter=cfg.model_max_iter,
        smote_k_neighbors=smote_k
    )
    
    pipe.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return {
        "pipeline": pipe,
        "metrics": metrics,
        "config": asdict(cfg),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }

def save_artifacts(
    out_dir: str | Path,
    *,
    pipeline: ImbPipeline,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": out / "baseline_model.joblib",
        "metrics": out / "metrics.json",
        "config": out / "train_config.json",
    }

    dump(pipeline, paths["model"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2))
    paths["config"].write_text(json.dumps(config, indent=2))

    return {k: str(v) for k, v in paths.items()}