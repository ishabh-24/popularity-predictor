from __future__ import annotations
import argparse
import random
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from ..data import DatasetSpec, load_dataset
from ..modeling import (
    RandomForestTrainConfig,
    TrainConfig,
    prepare_xy_for_training,
    smote_settings,
    build_pipeline,
    build_rf_pipeline,
    save_artifacts,
    train_evaluate_baseline,
    train_evaluate_random_forest,
)
from ..nn_explainability import NNExplainabilityConfig, run_nn_explainability
from ..nn_modeling import NeuralNetTrainConfig, save_nn_artifacts, train_evaluate_neural_net

""" This file contains the implementation of the training script for the Kaggle Billboard dataset.
Part of our training includes hyperparameter tuning for Logistic Regression and Random Forest models.

Justification for RandomizedSearchCV: RandomizedSearchCV is a technique that randomly samples a subset
of the hyperparameters to tune, and then evaluates the model on the validation set. This is a good 
technique because it is more efficient than evaluating all possible combinations of the hyperparameters.
"""

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train binary hit classifier from kaggle_billboard_songs.csv (hit=Billboard match)."
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to kaggle_billboard_songs.csv (defaults to data/kaggle_billboard_songs.csv).",
    )
    p.add_argument("--out", type=str, default="", help="Output artifacts directory (defaults to artifacts/).")
    p.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling.")
    p.add_argument(
        "--label-scheme",
        type=str,
        choices=["0/1", "1/2"],
        default="0/1",
        help="How to display labels in printed output. Model always trains on 0/1 internally.",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=["logreg", "rf", "nn"],
        default="logreg",
        help="logreg = logistic regression (baseline); rf = random forest; nn = PyTorch Lightning HitNet.",
    )
    p.add_argument("--tune", action="store_true", help="Enable simple hyperparameter tuning for selected model.")
    p.add_argument("--tune-n-iter", type=int, default=20, help="Random search iterations for --tune.")
    p.add_argument("--tune-cv", type=int, default=5, help="CV folds for --tune (logreg/rf only).")
    p.add_argument(
        "--tune-scoring",
        type=str,
        default="f1",
        choices=["f1", "roc_auc"],
        help="Selection metric for --tune.",
    )
    p.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators (only --model rf).")
    p.add_argument("--max-depth", type=int, default=10, help="RandomForest max_depth; use 0 for None (only --model rf).")
    p.add_argument(
        "--rf-recent-years-window",
        type=int,
        default=None,
        nargs="?",
        const=4,
        metavar="N",
        help=(
            "Only --model rf: optional year filter release_year >= max_year - N. "
            "Default (omit): no year filter (full CSV, more hits for train/test). "
            "Pass with no value (--rf-recent-years-window) to use N=4 like logistic regression."
        ),
    )
    p.add_argument(
        "--run-explainability",
        action="store_true",
        help="Only --model nn: run post-training global + local explainability and save artifacts.",
    )
    p.add_argument(
        "--xai-local-rows",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated row offsets from the filtered training frame for local IG (only --model nn).",
    )
    p.add_argument(
        "--xai-max-samples",
        type=int,
        default=512,
        help="Max rows sampled for permutation and IG baseline reference.",
    )
    p.add_argument(
        "--xai-permutation-repeats",
        type=int,
        default=5,
        help="Permutation repeats per transformed feature.",
    )
    p.add_argument(
        "--xai-ig-steps",
        type=int,
        default=64,
        help="Integrated Gradients interpolation steps.",
    )
    p.add_argument(
        "--xai-top-k-local",
        type=int,
        default=15,
        help="Top-k grouped local attributions saved to nn_local_integrated_gradients.csv.",
    )
    return p


def parse_local_row_offsets(raw: str, n_rows: int) -> list[int]:
    if not raw.strip():
        return [0] if n_rows > 0 else []
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    uniq = sorted(set(vals))
    for v in uniq:
        if v < 0 or v >= n_rows:
            raise ValueError(f"Invalid --xai-local-rows index {v}; valid range is 0..{max(0, n_rows - 1)}")
    return uniq


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define hit label from this dataset.

    We treat `billboard_matched==1` as a 'hit' (made Hot 100 in any sampled weeks),
    and `billboard_matched==0` as 'miss'.
    """
    out = df.copy()
    if "billboard_matched" not in out.columns:
        raise ValueError("Expected column 'billboard_matched' in dataset.")
    out["is_hit"] = pd.to_numeric(out["billboard_matched"], errors="coerce").fillna(0).astype(int)
    out["is_hit"] = out["is_hit"].clip(0, 1)

    # Provide a real date column expected by the shared preprocessing code.
    # This Kaggle CSV has `release_year` but typically not `release_date`.
    if "release_date" not in out.columns and "release_year" in out.columns:
        yr = pd.to_numeric(out["release_year"], errors="coerce")
        out["release_date"] = pd.to_datetime(yr.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return out


def metrics_from_predictions(
    *,
    model_name: str,
    pred: np.ndarray,
    y_test: np.ndarray,
    proba_1: np.ndarray | None,
    n_rows: int,
    n_features_numeric: int,
    n_features_categorical: int,
    test_size: float,
    use_smote: bool,
    smote_k_neighbors: int | None,
    feature_columns: dict[str, list[str]],
    prep_meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auc = None
    if proba_1 is not None and len(np.unique(y_test)) == 2:
        auc = float(roc_auc_score(y_test, proba_1))
    out = {
        "model": model_name,
        "n_rows": int(n_rows),
        "recent_years_window": prep_meta["recent_years_window"],
        "audio_feature_columns": prep_meta["audio_feature_columns"],
        "max_audio_missing_frac": prep_meta["max_audio_missing_frac"],
        "n_rows_dropped_audio_missing": prep_meta["n_rows_dropped_audio_missing"],
        "n_features_numeric": int(n_features_numeric),
        "n_features_categorical": int(n_features_categorical),
        "test_size": float(test_size),
        "use_smote": bool(use_smote),
        "smote_k_neighbors": smote_k_neighbors if use_smote else None,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "feature_columns": feature_columns,
    }
    if extra:
        out.update(extra)
    return out


def train_evaluate_logreg_tuned(
    df: pd.DataFrame,
    *,
    spec: DatasetSpec,
    cfg: TrainConfig,
    tune_n_iter: int,
    tune_cv: int,
    tune_scoring: str,
) -> dict[str, Any]:
    dfp, numeric_features, categorical_features, y, prep_meta = prepare_xy_for_training(df, spec=spec, cfg=cfg)
    X = dfp[numeric_features + categorical_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    use_smote, smote_k = smote_settings(y_train, use_smote=cfg.use_smote)
    pipe = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_smote=use_smote,
        random_state=cfg.random_state,
        model_max_iter=cfg.model_max_iter,
        smote_k_neighbors=smote_k,
    )

    #This uses RandomizedSearchCV to tune the hyperparameters of the logistic regression model!
    param_dist: dict[str, list[Any]] = {
        "clf__C": np.logspace(-3, 2, num=20).tolist(),
        "clf__solver": ["lbfgs", "liblinear"],
    }
    if use_smote:
        param_dist["smote__k_neighbors"] = [max(1, min(k, smote_k)) for k in [3, 5, 7]]
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=max(1, int(tune_n_iter)),
        scoring=tune_scoring,
        cv=max(2, int(tune_cv)),
        random_state=cfg.random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    pred = best_pipe.predict(X_test)
    proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, "predict_proba") else None
    metrics = metrics_from_predictions(
        model_name="logistic_regression",
        pred=pred,
        y_test=y_test,
        proba_1=proba,
        n_rows=dfp.shape[0],
        n_features_numeric=len(numeric_features),
        n_features_categorical=len(categorical_features),
        test_size=cfg.test_size,
        use_smote=use_smote,
        smote_k_neighbors=smote_k,
        feature_columns={"numeric": numeric_features, "categorical": categorical_features},
        prep_meta=prep_meta,
        extra={
            "tuned": True,
            "tune_method": "RandomizedSearchCV",
            "tune_scoring": tune_scoring,
            "tune_cv": max(2, int(tune_cv)),
            "tune_n_iter": max(1, int(tune_n_iter)),
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
    )
    return {"pipeline": best_pipe, "metrics": metrics, "config": {"base": cfg.__dict__, "tune": {"n_iter": tune_n_iter, "cv": tune_cv, "scoring": tune_scoring}}}


def train_evaluate_rf_tuned(
    df: pd.DataFrame,
    *,
    spec: DatasetSpec,
    cfg: RandomForestTrainConfig,
    tune_n_iter: int,
    tune_cv: int,
    tune_scoring: str,
) -> dict[str, Any]:
    base_cfg = TrainConfig(
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        use_smote=cfg.use_smote,
        max_audio_missing_frac=cfg.max_audio_missing_frac,
        recent_years_window=cfg.recent_years_window,
    )
    dfp, numeric_features, categorical_features, y, prep_meta = prepare_xy_for_training(df, spec=spec, cfg=base_cfg)
    X = dfp[numeric_features + categorical_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    use_smote, smote_k = smote_settings(y_train, use_smote=cfg.use_smote)
    pipe = build_rf_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_smote=use_smote,
        random_state=cfg.random_state,
        smote_k_neighbors=smote_k,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
    )

    #RandomizedSearchCV to tune the hyperparameters of the random forest!
    param_dist: dict[str, list[Any]] = {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 6, 10, 14],
        "clf__min_samples_leaf": [1, 3, 5, 10],
    }
    if use_smote:
        param_dist["smote__k_neighbors"] = [max(1, min(k, smote_k)) for k in [3, 5, 7]]
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=max(1, int(tune_n_iter)),
        scoring=tune_scoring,
        cv=max(2, int(tune_cv)),
        random_state=cfg.random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    pred = best_pipe.predict(X_test)
    proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, "predict_proba") else None
    best_clf = best_pipe.named_steps["clf"]
    metrics = metrics_from_predictions(
        model_name="random_forest",
        pred=pred,
        y_test=y_test,
        proba_1=proba,
        n_rows=dfp.shape[0],
        n_features_numeric=len(numeric_features),
        n_features_categorical=len(categorical_features),
        test_size=cfg.test_size,
        use_smote=use_smote,
        smote_k_neighbors=smote_k,
        feature_columns={"numeric": numeric_features, "categorical": categorical_features},
        prep_meta=prep_meta,
        extra={
            "n_estimators": int(best_clf.n_estimators),
            "max_depth": best_clf.max_depth,
            "min_samples_leaf": int(best_clf.min_samples_leaf),
            "tuned": True,
            "tune_method": "RandomizedSearchCV",
            "tune_scoring": tune_scoring,
            "tune_cv": max(2, int(tune_cv)),
            "tune_n_iter": max(1, int(tune_n_iter)),
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
    )
    return {"pipeline": best_pipe, "metrics": metrics, "config": {"base": cfg.__dict__, "tune": {"n_iter": tune_n_iter, "cv": tune_cv, "scoring": tune_scoring}}}


def train_evaluate_nn_tuned_simple(
    df: pd.DataFrame,
    *,
    spec: DatasetSpec,
    tune_n_iter: int,
    random_state: int,
) -> dict[str, Any]:
    rng = random.Random(random_state)
    candidates: list[NeuralNetTrainConfig] = []
    for _ in range(max(1, int(tune_n_iter))):
        candidates.append(
            NeuralNetTrainConfig(
                random_state=random_state,
                batch_size=rng.choice([16, 32, 64]),
                epochs=rng.choice([40, 80, 120]),
                lr=rng.choice([1e-4, 5e-4, 1e-3, 5e-3]),
                val_fraction=0.1,
            )
        )
    best_result = None
    best_score = -1.0
    best_cfg = None
    for cand in candidates:
        res = train_evaluate_neural_net(df, spec=spec, cfg=cand)
        score = float(res["metrics"].get("f1", 0.0))
        if score > best_score:
            best_score = score
            best_result = res
            best_cfg = cand
    assert best_result is not None and best_cfg is not None
    best_result["metrics"].update(
        {
            "tuned": True,
            "tune_method": "simple_random_search",
            "tune_scoring": "f1",
            "tune_n_iter": max(1, int(tune_n_iter)),
            "best_score_simple_search": float(best_score),
            "best_params": {
                "batch_size": int(best_cfg.batch_size),
                "epochs": int(best_cfg.epochs),
                "lr": float(best_cfg.lr),
                "val_fraction": float(best_cfg.val_fraction),
            },
        }
    )
    return best_result


def main() -> int:
    args = build_arg_parser().parse_args()

    data_path = args.data if args.data else "data/kaggle_billboard_songs.csv"
    out_dir = args.out if args.out else "artifacts"

    df = load_dataset(data_path)
    df = prepare_target(df)

    spec = DatasetSpec(
        target_col="is_hit",
        id_cols=("track_name", "artist_name"),
        date_col="release_date",
        numeric_cols=(
            #Kaggle audio features
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
        ),
        categorical_cols=("genre",),
    )

    if args.model == "logreg":
        if args.tune:
            result = train_evaluate_logreg_tuned(
                df,
                spec=spec,
                cfg=TrainConfig(use_smote=not args.no_smote),
                tune_n_iter=args.tune_n_iter,
                tune_cv=args.tune_cv,
                tune_scoring=args.tune_scoring,
            )
        else:
            result = train_evaluate_baseline(df, spec=spec, cfg=TrainConfig(use_smote=not args.no_smote))
        saved = save_artifacts(
            out_dir,
            pipeline=result["pipeline"],
            metrics=result["metrics"],
            config=result["config"],
            pipeline_filename="baseline_pipeline.joblib",
        )
    elif args.model == "rf":
        md = None if args.max_depth == 0 else args.max_depth
        rf_cfg = RandomForestTrainConfig(
            use_smote=not args.no_smote,
            n_estimators=args.n_estimators,
            max_depth=md,
            recent_years_window=args.rf_recent_years_window,
        )
        
        if args.tune:
            result = train_evaluate_rf_tuned(
                df,
                spec=spec,
                cfg=rf_cfg,
                tune_n_iter=args.tune_n_iter,
                tune_cv=args.tune_cv,
                tune_scoring=args.tune_scoring,
            )
        else:
            result = train_evaluate_random_forest(df, spec=spec, cfg=rf_cfg)
        saved = save_artifacts(
            out_dir,
            pipeline=result["pipeline"],
            metrics=result["metrics"],
            config=result["config"],
            pipeline_filename="random_forest_pipeline.joblib",
            metrics_filename="metrics_random_forest.json",
            config_filename="train_config_random_forest.json",
        )
    else:
        if args.tune:
            result = train_evaluate_nn_tuned_simple(
                df,
                spec=spec,
                tune_n_iter=args.tune_n_iter,
                random_state=42,
            )
        else:
            result = train_evaluate_neural_net(df, spec=spec, cfg=NeuralNetTrainConfig())
        saved = save_nn_artifacts(
            out_dir,
            bundle=result["pipeline"],
            metrics=result["metrics"],
            config=result["config"],
        )
        if args.run_explainability:
            local_rows = parse_local_row_offsets(args.xai_local_rows, len(result["X_all"]))
            xai_cfg = NNExplainabilityConfig(
                max_samples=args.xai_max_samples,
                permutation_repeats=args.xai_permutation_repeats,
                ig_steps=args.xai_ig_steps,
                top_k_local=args.xai_top_k_local,
                random_state=int(result["config"].get("random_state", 42)),
            )
            xai_saved = run_nn_explainability(
                result["pipeline"],
                X_eval=result["X_test"],
                y_eval=np.asarray(result["y_test"], dtype=np.int64),
                X_local=result["X_all"].iloc[local_rows],
                out_dir=out_dir,
                cfg=xai_cfg,
            )
            saved.update({f"xai_{k}": v for k, v in xai_saved.items()})

    print("Saved artifacts:")
    for k, v in saved.items():
        print(f"- {k}: {v}")

    m = result["metrics"]
    print("\nLabel definition:")
    if args.label_scheme == "1/2":
        print("1 = hit (billboard_matched==1)")
        print("2 = miss (billboard_matched==0)")
        print("(trained as hit=1, miss=0)")
    else:
        print("1 = hit (billboard_matched==1)")
        print("0 = miss (billboard_matched==0)")

    print("\nKey metrics:")
    for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        print(f"- {k}: {m.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

